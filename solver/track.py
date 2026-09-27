"""
Builds a smooth, closed-loop track geometry from one driver's real X/Y
position trace, and computes curvature analytically from a periodic
smoothing spline (rather than by differencing the raw points, which are
noisy and would give a wildly jumpy curvature signal).

This uses the fastest driver's own racing line as "the track", not the
official painted track centreline (which isn't in this data). That's a
deliberate, stated simplification: see the README.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep

XY_UNITS_TO_METRES = 0.1  # FastF1 X/Y are in decimetres; confirmed against the Distance channel


@dataclass
class Track:
    s: np.ndarray          # arc length grid, metres, s[0] = 0
    x: np.ndarray          # metres
    y: np.ndarray
    kappa: np.ndarray      # curvature, 1/metres (signed: positive = left-hand turn)
    length: float
    ds: float


def load_reference_line(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["X"] *= XY_UNITS_TO_METRES
    df["Y"] *= XY_UNITS_TO_METRES
    return df


def build_track(csv_path, n_points: int = 1200, smoothing: float | None = None) -> Track:
    df = load_reference_line(csv_path)
    x, y = df["X"].to_numpy(), df["Y"].to_numpy()

    # Fit a periodic (closed-loop) smoothing spline parameterised by
    # normalised arc length u in [0, 1). `per=True` forces the spline and
    # its derivatives to match up smoothly at the start/finish line.
    # `s=smoothing` controls how much the noisy driven line is smoothed;
    # too little leaves jittery, unphysical curvature spikes, too much
    # rounds off real corners. The default below was chosen by checking
    # the curvature plot against Silverstone's known corner sequence.
    if smoothing is None:
        smoothing = 0.05 * len(x)
    tck, _ = splprep([x, y], per=True, s=smoothing, k=5)

    u = np.linspace(0, 1, n_points, endpoint=False)
    xs, ys = splev(u, tck)
    dxs, dys = splev(u, tck, der=1)
    ddxs, ddys = splev(u, tck, der=2)

    speed_param = np.hypot(dxs, dys)  # d(arc length)/du
    kappa = (dxs * ddys - dys * ddxs) / speed_param**3

    # Resample onto a uniform arc-length grid (splprep's parameter u is not
    # arc length itself, so points above are not evenly spaced in metres yet)
    ds_cumulative = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    length = ds_cumulative[-1] + np.hypot(xs[0] - xs[-1], ys[0] - ys[-1])

    n_uniform = n_points
    s_uniform = np.linspace(0, length, n_uniform, endpoint=False)
    x_u = np.interp(s_uniform, ds_cumulative, xs, period=length)
    y_u = np.interp(s_uniform, ds_cumulative, ys, period=length)
    kappa_u = np.interp(s_uniform, ds_cumulative, kappa, period=length)

    return Track(s=s_uniform, x=x_u, y=y_u, kappa=kappa_u, length=length,
                 ds=s_uniform[1] - s_uniform[0])
