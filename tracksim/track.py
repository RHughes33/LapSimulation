"""
Track geometry from a real driven lap.

FastF1's X/Y telemetry channels are in decimetres (tenths of a metre), not
metres, confirmed by checking that the path length built from X/Y matches
the telemetry's own Distance channel only after dividing by 10.

FastF1 samples telemetry at fixed time intervals, not fixed distance, so at
high speed (where the car covers more ground per sample) the raw points can
be 15-20 m apart, versus a few metres apart in slow corners. An earlier
version of this module resampled onto an even grid with linear
interpolation, then smoothed with a moving average. That failed exactly in
the sparse, high-speed sections: a 15 m smoothing window doesn't span
enough real samples to average out small position noise when the raw
points themselves are 15-20 m apart, and linear interpolation between
sparse points creates small kinks at every real sample, which a second
derivative (needed for curvature) amplifies badly. In testing, this
produced a "corner" tight enough to need over 100g at a point where the
real car was doing 300 km/h in a straight line, obviously not real.

This version fits a smoothing spline directly to the raw (x, y) points
(scipy.interpolate.splprep, periodic since a lap is a closed loop), which
handles the smoothing and the resampling together and is the standard tool
for exactly this kind of noisy, unevenly-sampled path data. Curvature comes
from the spline's own analytic first and second derivatives rather than
finite differences on resampled points, which is both more accurate and
not sensitive to the resampling spacing.

A note on what "the track" means here: there is no public centreline data,
so the fastest recorded lap in the session is used as a stand-in for the
track geometry. That means this project does not solve for the racing
line, only for the speed along a line a real driver already chose. This
is a real modelling simplification and is treated as one in the write-up,
not hidden.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep


@dataclass
class Track:
    s: np.ndarray          # arc length from the start/finish line, metres
    x: np.ndarray          # position, metres
    y: np.ndarray
    curvature: np.ndarray  # signed curvature, 1/metres (1/radius)
    length: float

    @property
    def n_points(self):
        return len(self.s)


def track_from_xy(x_raw, y_raw, point_spacing_m: float = 2.0, smoothing: float = 3000.0) -> Track:
    """Build track geometry from raw (x, y) points in metres, ordered around the lap.

    smoothing: scipy's spline smoothing factor (larger = smoother, less
    faithful to raw noise).
    """
    x_raw, y_raw = np.asarray(x_raw, float), np.asarray(y_raw, float)

    # Close the loop: lap telemetry starts just after the line and ends just
    # before it, so join the end back to the start
    if np.hypot(x_raw[-1] - x_raw[0], y_raw[-1] - y_raw[0]) > 1.0:
        x_raw = np.append(x_raw, x_raw[0])
        y_raw = np.append(y_raw, y_raw[0])

    raw_s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_raw), np.diff(y_raw)))])
    length = raw_s[-1]

    # Periodic smoothing spline, parameterised by arc length. per=True
    # enforces matching position and derivatives at the join, so the lap is
    # treated as the closed loop it is.
    u = raw_s[:-1] / length
    tck, _ = splprep([x_raw[:-1], y_raw[:-1]], u=u, per=True, s=smoothing)

    n = int(length / point_spacing_m)
    u_even = np.linspace(0, 1, n, endpoint=False)
    x, y = splev(u_even, tck)
    dx, dy = splev(u_even, tck, der=1)
    ddx, ddy = splev(u_even, tck, der=2)

    # Curvature of a parametric curve. The formula is invariant to how the
    # curve is parameterised, so the spline's own u-derivatives can be used
    # directly without converting to arc-length derivatives first.
    denom = (dx**2 + dy**2) ** 1.5
    curvature = (dx * ddy - dy * ddx) / np.where(denom < 1e-9, np.nan, denom)
    curvature = np.nan_to_num(curvature)

    return Track(s=u_even * length, x=x, y=y, curvature=curvature, length=length)


def read_xy(csv_path):
    """Raw X/Y from a FastF1 telemetry file, converted from decimetres to metres."""
    df = pd.read_csv(csv_path)
    return df["X"].to_numpy() / 10.0, df["Y"].to_numpy() / 10.0, df


def load_track(csv_path, point_spacing_m: float = 2.0, smoothing: float = 3000.0) -> Track:
    """Track geometry from a single driver's recorded line."""
    x, y, _ = read_xy(csv_path)
    return track_from_xy(x, y, point_spacing_m, smoothing)


def consensus_track(csv_paths, n_samples: int = 3000, **kwargs) -> Track:
    """Track geometry from the average of several drivers' recorded lines.

    Each driver's line is resampled at the same fractions of lap distance,
    then the positions are averaged point by point. Position noise in each
    recorded line is largely independent from driver to driver, so
    averaging cancels much of it, while the shape all drivers share (the
    track itself) survives. This is a much steadier basis for curvature,
    which comes from a second derivative and so amplifies noise, than any
    single recorded lap.
    """
    frac = np.linspace(0, 1, n_samples, endpoint=False)
    xs, ys = [], []
    for path in csv_paths:
        x, y, _ = read_xy(path)
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        f = s / s[-1]
        xs.append(np.interp(frac, f, x))
        ys.append(np.interp(frac, f, y))
    return track_from_xy(np.mean(xs, axis=0), np.mean(ys, axis=0), **kwargs)
