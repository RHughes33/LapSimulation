"""
Part 2: 2026 energy deployment. Adds the battery and electric motor to the
lap simulator, finds the fastest legal deployment strategy, and works out
what extra energy is worth. Takes several minutes: every optimisation
searches 1,800 strategies.

    python analysis/energy_analysis.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tracksim.calibrate import calibrate_all, load_session  # noqa: E402
from tracksim.energy import PowerUnit, optimise_deployment, unconstrained_lap  # noqa: E402
from tracksim.solver import simulate_lap  # noqa: E402
from tracksim.track import consensus_track  # noqa: E402
from tracksim.vehicle import Vehicle  # noqa: E402

CHARTS = ROOT / "charts"
BLUE, RED, GREY, INK, GOLD, GREEN = "#1f5fa8", "#b8452f", "#8a8f98", "#14213d", "#c9962b", "#3f7d4e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
DRAG_GRID = [0.9, 0.95, 1.0, 1.05, 1.1]


def real_traces(track, summary, top_n=10):
    frac = track.s / track.length
    dfs = [pd.read_csv(d["path"]) for d in summary[:top_n]]
    f = [df["Distance"] / df["Distance"].max() for df in dfs]
    speed = np.median([np.interp(frac, fi, df["Speed"] / 3.6) for fi, df in zip(f, dfs)], 0)
    throttle = np.median([np.interp(frac, fi, df["Throttle"]) for fi, df in zip(f, dfs)], 0)
    brake = np.median([np.interp(frac, fi, df["Brake"].astype(float)) for fi, df in zip(f, dfs)], 0)
    return speed, throttle, brake


def fit_drag(track, cal, pu, real, straight):
    """The earlier drag value (0.92 m^2) was fitted alongside a constant
    'effective' power, so it absorbed the missing energy limits. With the
    real power unit modelled, drag is refitted: for each candidate, find
    that car's best strategy, then compare its speed on the straights
    (where the top 10 were at full throttle above 200 km/h) with reality.
    Fitting to speeds rather than to accelerations avoids differentiating
    a speed channel sampled only a few times a second."""
    rows = []
    for cda in DRAG_GRID:
        veh = Vehicle(mu=cal["mu"], lift_area=cal["lift_area"], drag_area=cda,
                      brake_limit_g=cal["brake_limit_g"])
        best, _ = optimise_deployment(track, veh, pu, step_kmh=10.0)
        rmse = float(np.sqrt(np.mean((best.v_final[straight] - real[straight]) ** 2)) * 3.6)
        rows.append((cda, rmse))
        print(f"  drag {cda}: straight-line speed error {rmse:.1f} km/h", flush=True)
    return min(rows, key=lambda r: r[1])[0], rows


def speed_power_chart(track, best, base, real, pu):
    s = track.s / 1000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.4]})
    a1.plot(s, real * 3.6, color=INK, lw=1.3, label="Top 10 median (real)")
    a1.plot(s, base.v_final * 3.6, color=GREY, lw=1.2, ls="--", label="Constant power (previous model)")
    a1.plot(s, best.v_final * 3.6, color=RED, lw=1.6, label="With 2026 energy rules")
    a1.set_ylabel("Speed (km/h)")
    a1.set_title("Energy rules reproduce the real cars slowing at full throttle on the straights")
    a1.legend(frameon=False, loc="lower right", ncol=3, fontsize=9)

    k = best.k_power / 1000
    a2.fill_between(s, 0, k, where=k > 0, color=GREEN, step="mid", label="Deploying")
    a2.fill_between(s, 0, k, where=(k < 0) & (best.v_final < best.v_curvature) & (k > -pu.k_clip_w / 1000 - 1),
                    color=GOLD, step="mid", label="Harvesting")
    a2.axhline(0, color=INK, lw=0.6)
    a2.set_ylabel("Motor power (kW)")
    a2.set_xlabel("Distance (km)")
    a2.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "9_energy_speed_trace.png", dpi=170)
    plt.close(fig)


def strategy_map_chart(tried, best, pu):
    vd = np.array([t.v_deploy for t in tried]) * 3.6
    vc = np.array([t.v_clip for t in tried]) * 3.6
    lt = np.array([t.lap_time for t in tried])
    legal = np.array([t.legal(pu) for t in tried])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(vd[~legal], vc[~legal], c="#e3e5ea", s=14, marker="s", label="Breaks the energy rules")
    sc = ax.scatter(vd[legal], vc[legal], c=lt[legal], cmap="viridis_r", s=14, marker="s",
                    vmax=np.percentile(lt[legal], 90))
    ax.scatter(best.v_deploy * 3.6, best.v_clip * 3.6, marker="*", s=260, color=RED,
               edgecolor="white", zorder=3, label=f"Fastest legal: {best.lap_time:.2f} s")
    fig.colorbar(sc, ax=ax, label="Lap time (s)")
    ax.set_xlabel("Deploy the motor below (km/h)")
    ax.set_ylabel("Superclip above (km/h)")
    ax.set_title("Every deployment strategy tried")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "10_strategy_map.png", dpi=170)
    plt.close(fig)


def energy_value_chart(values, base_time):
    labels = list(values)
    deltas = [values[k] - base_time for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [BLUE if d < -0.005 else RED if d > 0.005 else GREY for d in deltas]
    ax.barh(labels, deltas, color=cols)
    for i, d in enumerate(deltas):
        ax.text(d - 0.08 if d < 0 else d + 0.08, i, f"{d:+.2f} s", va="center",
                ha="right" if d < 0 else "left", fontsize=9)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Change in lap time vs the 2026 qualifying rules (s)")
    ax.set_title("What extra energy is worth at Silverstone")
    xmin = min(deltas)
    ax.set_xlim(xmin * 1.25, 1.2)
    fig.tight_layout()
    fig.savefig(CHARTS / "11_energy_value.png", dpi=170)
    plt.close(fig)


def time_gap_by_phase(track, best, real, throttle, brake):
    ds = track.s[1] - track.s[0]
    gain = ds / real - ds / best.v_final
    phases = {"full throttle": throttle >= 98, "braking": brake > 0.5}
    phases["part throttle / coasting"] = ~phases["full throttle"] & ~phases["braking"]
    return {k: {"share_of_lap_pct": round(float(m.mean() * 100), 1),
                "sim_time_gain_s": round(float(gain[m].sum()), 2)} for k, m in phases.items()}


def main():
    CHARTS.mkdir(exist_ok=True)
    summary = load_session(ROOT / "data")
    track = consensus_track([d["path"] for d in summary])
    cal = calibrate_all(track, summary)
    real, throttle, brake = real_traces(track, summary)
    straight = (throttle >= 98) & (real * 3.6 > 200)
    pu = PowerUnit()

    print("Fitting drag...")
    cda, drag_rows = fit_drag(track, cal, pu, real, straight)
    veh = Vehicle(mu=cal["mu"], lift_area=cal["lift_area"], drag_area=cda, brake_limit_g=cal["brake_limit_g"])

    print("Optimising deployment...")
    best, tried = optimise_deployment(track, veh, pu)
    base = simulate_lap(track, Vehicle(mu=cal["mu"], lift_area=cal["lift_area"], power_w=cal["power_w"],
                                       drag_area=cal["drag_area"], brake_limit_g=cal["brake_limit_g"]))

    print("Valuing extra energy...")
    values = {}
    for label, change in [("Harvest limit 6 MJ", {"harvest_limit_j": 6e6}),
                          ("Harvest limit 8 MJ", {"harvest_limit_j": 8e6}),
                          ("Battery runs down 1 MJ", {"battery_drawdown_j": 1e6}),
                          ("Battery runs down 2 MJ", {"battery_drawdown_j": 2e6})]:
        values[label] = optimise_deployment(track, veh, replace(pu, **change))[0].lap_time
        print(f"  {label}: {values[label]:.3f}", flush=True)
    values["No energy limits"] = unconstrained_lap(track, veh, pu).lap_time
    values = dict(sorted(values.items(), key=lambda kv: kv[1], reverse=True))

    speed_power_chart(track, best, base, real, pu)
    strategy_map_chart(tried, best, pu)
    energy_value_chart(values, best.lap_time)

    rmse = lambda v: round(float(np.sqrt(np.mean((v[straight] - real[straight]) ** 2)) * 3.6), 1)
    out = {
        "drag_area_refit": cda,
        "drag_fit": {str(c): r for c, r in drag_rows},
        "best_lap_s": round(best.lap_time, 3),
        "constant_power_lap_s": round(base.lap_time, 3),
        "deploy_below_kmh": round(best.v_deploy * 3.6, 1),
        "superclip_above_kmh": round(best.v_clip * 3.6, 1),
        "harvested_braking_mj": round(best.harvested_brake_j / 1e6, 2),
        "harvested_clipping_mj": round(best.harvested_clip_j / 1e6, 2),
        "deployed_mj": round(best.deployed_j / 1e6, 2),
        "straight_speed_error_kmh": {"constant_power": rmse(base.v_final), "energy_model": rmse(best.v_final)},
        "lap_time_options_s": {k: round(v, 3) for k, v in values.items()},
        "remaining_gap_by_phase": time_gap_by_phase(track, best, real, throttle, brake),
    }
    results_path = ROOT / "results.json"
    data = json.loads(results_path.read_text()) if results_path.exists() else {}
    data["energy"] = out
    results_path.write_text(json.dumps(data, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
