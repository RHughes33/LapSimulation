"""
Runs the full lap simulation study and saves every chart and number used
in the write-up.

    python analysis/run_analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tracksim.calibrate import calibrate_all, load_session  # noqa: E402
from tracksim.track import consensus_track, load_track  # noqa: E402
from tracksim.vehicle import G, Vehicle  # noqa: E402
from tracksim.solver import simulate_lap  # noqa: E402

CHARTS = ROOT / "charts"
DATA = ROOT / "data"
BLUE, RED, GREY, INK, GOLD, GREEN = "#1f5fa8", "#b8452f", "#8a8f98", "#14213d", "#c9962b", "#3f7d4e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BOOTSTRAP_SAMPLES = 30
RNG = np.random.default_rng(42)


def real_profiles(track, summary, top_n=10):
    """Top-n recorded speeds on the track's distance grid (matched by fraction of lap)."""
    frac = track.s / track.length
    out = []
    for d in summary[:top_n]:
        df = pd.read_csv(d["path"])
        out.append(np.interp(frac, df["Distance"] / df["Distance"].max(), df["Speed"] / 3.6))
    return np.array(out)


def vehicle_from(c):
    return Vehicle(mu=c["mu"], lift_area=c["lift_area"], power_w=c["power_w"],
                   drag_area=c["drag_area"], brake_limit_g=c["brake_limit_g"])


def track_map_chart(track, res):
    fig, ax = plt.subplots(figsize=(7.5, 8))
    sc = ax.scatter(track.x, track.y, c=res.v_final * 3.6, cmap="RdYlBu_r", s=7, vmin=80, vmax=330)
    ax.plot(track.x[0], track.y[0], "k|", ms=18, mew=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Simulated speed (km/h)")
    ax.set_title(f"Silverstone, simulated 2026 lap: {res.lap_time:.2f} s", fontsize=12)
    fig.tight_layout()
    fig.savefig(CHARTS / "1_track_map.png", dpi=170)
    plt.close(fig)


def speed_trace_chart(track, res, real):
    s = track.s / 1000
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                    gridspec_kw={"height_ratios": [6, 0.5]})
    ax.fill_between(s, real.min(0) * 3.6, real.max(0) * 3.6, color=GREY, alpha=0.25, label="Top 10 range")
    ax.plot(s, np.median(real, 0) * 3.6, color=INK, lw=1.3, label="Top 10 median (real)")
    ax.plot(s, res.v_final * 3.6, color=RED, lw=1.6, label="Simulation")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Simulated vs real speed around the lap")
    ax.legend(frameon=False, loc="lower right", ncol=3)

    mode = res.limiting_mode()
    colours = {"accelerating": GREEN, "braking": RED, "corner": GOLD}
    for name, col in colours.items():
        strip.fill_between(s, 0, 1, where=(mode == name), color=col, step="mid", label=name)
    strip.set_yticks([])
    strip.set_xlabel("Distance (km)")
    strip.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -1.6), fontsize=9,
                 title="What limits the simulated car", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "2_speed_trace.png", dpi=170)
    plt.close(fig)


def corner_fit_chart(cal):
    lat = cal["lateral"]
    c = lat["corners"]
    v2 = (c["apex_kmh"] / 3.6) ** 2
    lat_g = c["curvature"] * v2 / G
    fit = c["grip_limited"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(v2[fit], lat_g[fit], color=BLUE, s=45, label="Grip-limited corners (fitted)", zorder=3)
    ax.scatter(v2[~fit], lat_g[~fit], facecolors="none", edgecolors=RED, s=60, lw=1.5,
               label="Taken flat out (excluded: not at the grip limit)", zorder=3)
    xs = np.linspace(0, v2.max() * 1.05, 50)
    ax.plot(xs, cal["mu"] * (1 + 0.5 * 1.225 * cal["lift_area"] * xs / (768 * G)), color=INK, lw=1.2,
            label=f"Fit: grip {cal['mu']:.2f}, downforce area {cal['lift_area']:.1f} m²")
    ax.set_xlabel("Apex speed squared (m²/s²)")
    ax.set_ylabel("Lateral acceleration at apex (g)")
    ax.set_title(f"Cornering grip rises with speed as downforce builds (R² = {lat['fit_r_squared']:.2f})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "3_corner_fit.png", dpi=170)
    plt.close(fig)


def power_fit_chart(cal):
    p = cal["power"]
    x, y = p["_x"], p["_y"] / 1000
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x / 1e5, y, s=3, color=BLUE, alpha=0.25)
    xs = np.linspace(0, x.max(), 50)
    ax.plot(xs / 1e5, (p["power_w"] - 0.5 * 1.225 * p["drag_area"] * xs) / 1000, color=INK, lw=1.4,
            label=f"Fit: {p['power_w'] / 1000:.0f} kW effective, drag area {p['drag_area']:.2f} m²")
    ax.axhline(750, color=RED, ls="--", lw=1)
    ax.text(0.2, 765, "2026 headline power, 750 kW", color=RED, fontsize=9)
    ax.set_xlabel("Speed cubed (×10⁵ m³/s³)")
    ax.set_ylabel("Power going into acceleration (kW)")
    ax.set_title("Full-throttle data: effective power sits well below the headline figure")
    ax.set_ylim(0, 900)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS / "4_power_fit.png", dpi=170)
    plt.close(fig)


def apex_error(track, res, real, cal):
    """Mean absolute error in minimum corner speed across every detected corner."""
    frac = track.s / track.length
    med = np.median(real, 0)
    errs = []
    for _, c in cal["lateral"]["corners"].iterrows():
        i = np.argmin(np.abs(frac - c["fraction"]))
        idx = np.arange(i - 25, i + 26) % len(frac)
        errs.append(res.v_final[idx].min() - med[idx].min())
    return float(np.mean(np.abs(errs)) * 3.6)


def development_chart(track, cal, summary, results, real):
    """Lap time at each modelling stage, alongside apex speed error, so the
    write-up can show what each change did, and that a matching lap time on
    its own isn't evidence the model is right."""
    literature = Vehicle(mu=1.664, lift_area=3.5, power_w=750_000, drag_area=1.793, brake_limit_g=99)
    corners_fitted = Vehicle(mu=cal["mu"], lift_area=cal["lift_area"], power_w=750_000,
                             drag_area=1.793, brake_limit_g=99)
    runs = [
        ("Published power\n& downforce", simulate_lap(track, literature, combined_grip=False)),
        ("+ grip & downforce\nfitted to corners", simulate_lap(track, corners_fitted, combined_grip=False)),
        ("+ power, drag & braking\nfitted to telemetry", simulate_lap(track, vehicle_from(cal), combined_grip=False)),
        ("+ combined grip\n(friction circle)", simulate_lap(track, vehicle_from(cal))),
    ]
    stages = [(n, r.lap_time, apex_error(track, r, real, cal)) for n, r in runs]
    pole = summary[0]["lap_time_seconds"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names, times, apex = zip(*stages)
    bars = ax.bar(names, times, color=[GREY, GREY, GREY, BLUE])
    for b, t, a in zip(bars, times, apex):
        ax.text(b.get_x() + b.get_width() / 2, t + 0.3, f"{t:.1f} s", ha="center")
        ax.text(b.get_x() + b.get_width() / 2, 71.5, f"apex error\n{a:.0f} km/h", ha="center",
                color="white", fontsize=9)
    ax.axhline(pole, color=RED, ls="--", lw=1.2)
    ax.text(3.45, pole + 0.3, f"Real pole {pole:.2f} s", color=RED, ha="right")
    ax.set_ylim(70, max(times) + 4)
    ax.set_ylabel("Simulated lap time (s)")
    ax.set_title("How each modelling step changed the answer\n"
                 "(the first model matches pole only because its errors cancel out)")
    fig.tight_layout()
    fig.savefig(CHARTS / "5_model_development.png", dpi=170)
    plt.close(fig)
    results["development_stages"] = {n.replace("\n", " "): {"lap_s": round(t, 3), "apex_error_kmh": round(a, 1)}
                                     for n, t, a in stages}


def uncertainty(summary, veh, results):
    """Two views of how much the answer depends on the track geometry:
    the lap time on each driver's own recorded line, and a bootstrap over
    which drivers' lines go into the consensus."""
    own = {d["driver"]: simulate_lap(load_track(d["path"]), veh).lap_time for d in summary}
    boot = []
    paths = [d["path"] for d in summary]
    for _ in range(BOOTSTRAP_SAMPLES):
        pick = RNG.choice(len(paths), len(paths), replace=True)
        boot.append(simulate_lap(consensus_track([paths[i] for i in pick]), veh).lap_time)
    results["own_line_lap_times"] = {k: round(v, 3) for k, v in own.items()}
    results["own_line_spread_s"] = round(max(own.values()) - min(own.values()), 3)
    results["bootstrap_mean_s"] = round(float(np.mean(boot)), 3)
    results["bootstrap_95_interval_s"] = [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])]
    return own, np.array(boot)


def field_chart(summary, sim_time, boot):
    df = pd.DataFrame(summary).sort_values("lap_time_seconds", ascending=False)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.barh(df["driver"], df["lap_time_seconds"], color=GREY)
    ax.axvspan(lo, hi, color=BLUE, alpha=0.2, label=f"Simulation, 95% interval {lo:.2f}-{hi:.2f} s")
    ax.axvline(sim_time, color=BLUE, lw=1.6)
    ax.set_xlim(83, 94)
    ax.set_xlabel("Lap time (s)")
    ax.set_title("Real 2026 Silverstone qualifying laps vs the simulated 'perfect lap'")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "6_field_comparison.png", dpi=170)
    plt.close(fig)


def line_noise_chart(summary, own, boot, sim_time):
    order = [d["driver"] for d in summary]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(order, [own[d] for d in order], color=GREY)
    ax.axhline(sim_time, color=BLUE, lw=1.6, label=f"Consensus line of all 22: {sim_time:.2f} s")
    ax.set_ylim(min(own.values()) - 1, max(own.values()) + 1)
    ax.set_ylabel("Simulated lap time (s)")
    ax.set_title("Same car, same model, each driver's own recorded line\n"
                 "(drivers ordered by real qualifying position)")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS / "7_line_noise.png", dpi=170)
    plt.close(fig)


def sensitivity_chart(track, cal, results):
    """Lap time change from a 10% change in each parameter: the basic
    question a lap simulator exists to answer ('what is this worth?')."""
    base_veh = vehicle_from(cal)
    base = simulate_lap(track, base_veh).lap_time
    params = {"mu": "Tyre grip", "lift_area": "Downforce", "power_w": "Power",
              "drag_area": "Drag", "mass": "Mass", "brake_limit_g": "Braking limit"}
    rows = []
    for key, label in params.items():
        deltas = []
        for f in (0.9, 1.1):
            v = vehicle_from(cal)
            setattr(v, key, getattr(v, key) * f)
            deltas.append(simulate_lap(track, v).lap_time - base)
        rows.append((label, deltas[0], deltas[1]))
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=GREY, label="10% less")
    ax.barh(y, [r[2] for r in rows], color=BLUE, label="10% more")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Change in lap time (s)")
    ax.set_title("What each car parameter is worth at Silverstone")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "8_sensitivity.png", dpi=170)
    plt.close(fig)
    results["sensitivity_10pct"] = {r[0]: {"minus_10pct_s": round(r[1], 3), "plus_10pct_s": round(r[2], 3)} for r in rows}


def main():
    CHARTS.mkdir(exist_ok=True)
    summary = load_session(DATA)
    track = consensus_track([d["path"] for d in summary])
    cal = calibrate_all(track, summary)
    veh = vehicle_from(cal)
    res = simulate_lap(track, veh)
    real = real_profiles(track, summary)
    ds = track.s[1] - track.s[0]

    results = {
        "calibrated": {k: round(cal[k], 4) for k in ("mu", "lift_area", "power_w", "drag_area", "brake_limit_g")},
        "lateral_fit_r_squared": round(cal["lateral"]["fit_r_squared"], 3),
        "lateral_fit_loo_error_pct": round(cal["lateral"]["loo_mean_abs_error_pct"], 1),
        "corners_fitted": cal["lateral"]["n_corners_fitted"],
        "corners_excluded_flat_out": cal["lateral"]["n_corners_excluded_flat_out"],
        "power_fit_r_squared": round(cal["power"]["power_fit_r_squared"], 3),
        "brake_peaks_g": [round(p, 2) for p in cal["braking"]["brake_peaks_g"]],
        "sim_lap_s": round(res.lap_time, 3),
        "pole_s": summary[0]["lap_time_seconds"],
        "top10_median_profile_lap_s": round(float(np.sum(ds / np.median(real, 0))), 3),
        "speed_mean_abs_error_kmh": round(float(np.mean(np.abs(res.v_final - np.median(real, 0))) * 3.6), 1),
        "sim_top_speed_kmh": round(float(res.v_final.max() * 3.6), 1),
        "real_top_speed_kmh": round(cal["lateral"]["v_top_kmh"], 1),
    }
    results["sim_vs_pole_pct"] = round((res.lap_time / results["pole_s"] - 1) * 100, 2)

    track_map_chart(track, res)
    speed_trace_chart(track, res, real)
    corner_fit_chart(cal)
    power_fit_chart(cal)
    development_chart(track, cal, summary, results, real)
    own, boot = uncertainty(summary, veh, results)
    field_chart(summary, res.lap_time, boot)
    line_noise_chart(summary, own, boot, res.lap_time)
    sensitivity_chart(track, cal, results)

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
