"""
Calibration and validation of the vehicle model against real telemetry.

What is calibrated, and from what:
  - Drag area (CdA): from the session's real top speed, assuming the car is
    at full power in a straight line at that point.
  - Tyre friction (mu): from the tightest corner on the lap, where the car
    is slowest, so downforce (which grows with speed squared) contributes
    least and the result is least sensitive to the downforce estimate.
    The apex speed used is the median of the top 10 qualifiers, not one
    driver, so a single driver's mistake or unusually good corner doesn't
    set the car's grip level.
  - Downforce area (ClA) is NOT fitted. See vehicle.py for why.

Every other genuine corner on the lap is then used as an independent check.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .track import Track
from .vehicle import AIR_DENSITY, G


def load_session(data_dir):
    data_dir = Path(data_dir)
    summary = sorted(json.loads((data_dir / "laps_summary.json").read_text()),
                     key=lambda d: d["lap_time_seconds"])
    for d in summary:
        d["path"] = data_dir / d["telemetry_file"]
    return summary


def apex_speeds(summary, frac_centre, frac_halfwidth=0.015, top_n=10):
    """Minimum speed near a point on the lap, for each of the top_n qualifiers."""
    out = []
    for d in summary[:top_n]:
        df = pd.read_csv(d["path"])
        f = df["Distance"] / df["Distance"].max()
        zone = df[(f > frac_centre - frac_halfwidth) & (f < frac_centre + frac_halfwidth)]
        out.append(zone["Speed"].min())
    return np.array(out)


def detect_corners(reference_csv, order=8, min_gap_m=100.0):
    """Candidate corners: local minima in a reference lap's speed trace,
    returned as fractions of the lap so they can be located on any line."""
    df = pd.read_csv(reference_csv)
    v = df["Speed"].to_numpy()
    idx = argrelextrema(v, np.less_equal, order=order)[0]
    rows = []
    for i in idx:
        d = df["Distance"].iloc[i]
        if not rows or d - rows[-1]["distance"] > min_gap_m:
            rows.append({"distance": d, "fraction": d / df["Distance"].max()})
    return pd.DataFrame(rows)


def corner_curvature(track: Track, fraction, window_m=30.0):
    """Peak curvature near a point: the recorded apex (slowest point) can sit
    a few metres from the geometric point of tightest curvature."""
    i = np.argmin(np.abs(track.s / track.length - fraction))
    w = int(window_m / (track.s[1] - track.s[0]))
    idx = np.arange(i - w, i + w + 1) % track.n_points
    return np.abs(track.curvature[idx]).max()


def corner_table(track: Track, summary, max_radius_m=300.0, flat_out_throttle=90.0, top_n=10):
    """Every candidate corner with its geometry, top-10 median apex speed,
    and whether it is grip-limited.

    Two filters, both decided from the data before any fitting:
      - radius > max_radius_m: a lift on a kink or straight, not a corner
      - median apex throttle >= flat_out_throttle: taken flat out, so the car
        is not at its grip limit there. It only shows the car can do AT
        LEAST that speed, which says nothing about where the limit is.
    """
    rows = []
    for _, c in detect_corners(summary[0]["path"]).iterrows():
        k = corner_curvature(track, c["fraction"])
        if k < 1e-6 or 1 / k > max_radius_m:
            continue
        speeds, throttles = [], []
        for d in summary[:top_n]:
            df = pd.read_csv(d["path"])
            f = df["Distance"] / df["Distance"].max()
            zone = df[(f > c["fraction"] - 0.015) & (f < c["fraction"] + 0.015)]
            i = zone["Speed"].idxmin()
            speeds.append(df["Speed"][i])
            throttles.append(df["Throttle"][i])
        rows.append({"distance_m": round(c["fraction"] * track.length),
                     "fraction": c["fraction"], "curvature": k, "radius_m": 1 / k,
                     "apex_kmh": float(np.median(speeds)),
                     "apex_throttle_pct": float(np.median(throttles)),
                     "grip_limited": float(np.median(throttles)) < flat_out_throttle})
    return pd.DataFrame(rows)


def calibrate(track: Track, summary, mass=768.0, power_w=750_000.0):
    """Fit tyre grip (mu) and downforce area (ClA) across every grip-limited
    corner at once, using the fact that at the grip limit

        kappa * v^2 = mu*g + (mu * 0.5*rho*ClA / m) * v^2

    is a straight line in v^2 (intercept mu*g, slope mu*0.5*rho*ClA/m).
    Drag area comes from the session's top speed.
    """
    v_top = max(pd.read_csv(d["path"])["Speed"].max() for d in summary) / 3.6
    drag_area = 2 * power_w / (AIR_DENSITY * v_top**3)

    corners = corner_table(track, summary)
    fit = corners[corners["grip_limited"]]
    v2 = (fit["apex_kmh"].to_numpy() / 3.6) ** 2
    lat = fit["curvature"].to_numpy() * v2

    slope, intercept = np.polyfit(v2, lat, 1)
    mu = intercept / G
    lift_area = slope * mass / (0.5 * AIR_DENSITY * mu)
    pred = intercept + slope * v2
    r2 = 1 - np.sum((lat - pred) ** 2) / np.sum((lat - lat.mean()) ** 2)

    # Leave-one-out: refit without each corner in turn and predict it, a
    # fairer test than in-sample fit with only a handful of points
    loo = []
    for i in range(len(v2)):
        m = np.arange(len(v2)) != i
        s_, b_ = np.polyfit(v2[m], lat[m], 1)
        loo.append((lat[i] - (b_ + s_ * v2[i])) / lat[i])

    return {"mu": float(mu), "lift_area": float(lift_area), "drag_area": float(drag_area),
            "v_top_kmh": float(v_top * 3.6), "fit_r_squared": float(r2),
            "n_corners_fitted": int(len(fit)), "n_corners_excluded_flat_out": int((~corners["grip_limited"]).sum()),
            "loo_mean_abs_error_pct": float(np.mean(np.abs(loo)) * 100),
            "corners": corners, "loo_errors": loo}


def _accel_series(path):
    """Speed and longitudinal acceleration from one lap, smoothed enough to
    damp the 1 km/h quantisation of the speed channel."""
    from scipy.ndimage import median_filter
    df = pd.read_csv(path).drop_duplicates("Time").reset_index(drop=True)
    v = median_filter(df["Speed"].to_numpy() / 3.6, 5)
    a = np.gradient(v, df["Time"].to_numpy())
    a = pd.Series(a).rolling(5, center=True, min_periods=1).mean().to_numpy()
    return df, v, a


def calibrate_power_drag(summary, mass=768.0, top_n=10):
    """Effective power and drag area from full-throttle acceleration.

    At full throttle on a straight, drive force = power / v, so
        m * a * v = P - 0.5 * rho * CdA * v^3
    a straight line in v^3 with intercept P and slope -0.5*rho*CdA.
    Only samples above 150 km/h are used: below that the car can be
    traction-limited rather than power-limited.
    """
    x, y = [], []
    for d in summary[:top_n]:
        df, v, a = _accel_series(d["path"])
        m = (df["Throttle"].to_numpy() >= 98) & ~df["Brake"].to_numpy().astype(bool) & (v > 150 / 3.6) & (a > 0)
        x.append(v[m] ** 3)
        y.append(mass * a[m] * v[m])
    x, y = np.concatenate(x), np.concatenate(y)
    slope, power = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (power + slope * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"power_w": float(power), "drag_area": float(-2 * slope / AIR_DENSITY),
            "power_fit_r_squared": float(r2), "power_fit_samples": int(len(x)),
            "_x": x, "_y": y}


def calibrate_braking(summary, drag_area, mass=768.0, top_n=10, min_kmh=200.0):
    """Peak braking from grip: for each top-n lap, the hardest deceleration
    held over the smoothing window while braking above min_kmh, with the
    drag contribution at that speed taken off. Median across drivers.

    This is an effective value: the speed channel is sampled a few times a
    second, which rounds off the sharpest peaks, so true peak braking is
    likely somewhat higher.
    """
    peaks = []
    for d in summary[:top_n]:
        df, v, a = _accel_series(d["path"])
        m = df["Brake"].to_numpy().astype(bool) & (v > min_kmh / 3.6)
        if not m.any():
            continue
        i = np.where(m)[0][np.argmin(a[m])]
        drag_decel = 0.5 * AIR_DENSITY * drag_area * v[i] ** 2 / mass
        peaks.append((-a[i] - drag_decel) / G)
    return {"brake_limit_g": float(np.median(peaks)), "brake_peaks_g": [float(p) for p in peaks]}


def calibrate_all(track: Track, summary, mass=768.0):
    """Every fitted vehicle parameter, in one place."""
    lat = calibrate(track, summary, mass=mass)
    pwr = calibrate_power_drag(summary, mass=mass)
    brk = calibrate_braking(summary, pwr["drag_area"], mass=mass)
    return {"mu": lat["mu"], "lift_area": lat["lift_area"], "power_w": pwr["power_w"],
            "drag_area": pwr["drag_area"], "brake_limit_g": brk["brake_limit_g"],
            "lateral": lat, "power": pwr, "braking": brk}
