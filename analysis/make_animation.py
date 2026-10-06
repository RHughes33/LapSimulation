"""
Animated lap: the energy-optimised simulation racing a ghost of the real
pole lap round Silverstone, with speed, motor power and battery level.

Saves an MP4 (for sharing) and a smaller GIF (plays inside the README).

    python analysis/make_animation.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tracksim.calibrate import calibrate_all, load_session  # noqa: E402
from tracksim.energy import PowerUnit, optimise_deployment  # noqa: E402
from tracksim.track import consensus_track  # noqa: E402
from tracksim.vehicle import Vehicle  # noqa: E402

CHARTS = ROOT / "charts"
RED, INK, GREY, GREEN, GOLD = "#d0452f", "#14213d", "#9aa0a8", "#3f9d5a", "#d9a32b"
BG = "#0f1724"


def build_lap():
    summary = load_session(ROOT / "data")
    track = consensus_track([d["path"] for d in summary])
    cal = calibrate_all(track, summary)
    results = json.loads((ROOT / "results.json").read_text()) if (ROOT / "results.json").exists() else {}
    drag = results.get("energy", {}).get("drag_area_refit", 1.0)
    veh = Vehicle(mu=cal["mu"], lift_area=cal["lift_area"], drag_area=drag,
                  brake_limit_g=cal["brake_limit_g"])
    best, _ = optimise_deployment(track, veh, PowerUnit())

    pole = summary[0]
    df = pd.read_csv(pole["path"])
    frac = track.s / track.length
    v_pole = np.interp(frac, df["Distance"] / df["Distance"].max(), df["Speed"] / 3.6)
    return track, best, pole, v_pole


def timeline(track, v, official_time=None):
    """Elapsed time at each point. For the real lap, scaled so the finish
    matches the official lap time exactly (the speed trace is re-sampled
    onto the consensus line, so integrating it is only approximately right)."""
    ds = track.s[1] - track.s[0]
    t = np.concatenate([[0], np.cumsum(ds / v)])[:-1]
    total = t[-1] + ds / v[-1]
    if official_time is not None:
        t, total = t * official_time / total, official_time
    return t, total


def position_at(time, t, track, total):
    if time >= total:
        return len(track.s) - 1
    return int(np.searchsorted(t, time, side="right") - 1)


def main(speedup_mp4=3.0, speedup_gif=5.0):
    CHARTS.mkdir(exist_ok=True)
    track, best, pole, v_pole = build_lap()
    t_sim, T_sim = timeline(track, best.v_final)
    t_pole, T_pole = timeline(track, v_pole, pole["lap_time_seconds"])
    dt = (track.s[1] - track.s[0]) / best.v_final
    battery = np.cumsum(-best.k_power * dt) / 1e6   # MJ relative to the start line
    km = track.s / 1000
    k = best.k_power / 1000
    # Braking slows the car hard; superclipping only gently (the engine is
    # still pushing). Use deceleration to tell the two kinds of harvest apart.
    ds = track.s[1] - track.s[0]
    decel = -(np.roll(best.v_final, -1) ** 2 - best.v_final ** 2) / (2 * ds) / 9.81
    braking = decel > 1.0

    plt.rcParams.update({"font.size": 11, "text.color": "white", "axes.labelcolor": "#c8ccd4",
                         "xtick.color": "#c8ccd4", "ytick.color": "#c8ccd4", "axes.edgecolor": "#3a4456"})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.25], height_ratios=[1.5, 0.8, 1.0],
                          left=0.03, right=0.97, top=0.9, bottom=0.08, wspace=0.12, hspace=0.45)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_k = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_b = fig.add_subplot(gs[2, 1], sharex=ax_v)
    for ax in (ax_map, ax_v, ax_k, ax_b):
        ax.set_facecolor(BG)

    fig.text(0.03, 0.95, "2026 Silverstone qualifying: simulated lap vs real pole lap",
             fontsize=16, weight="bold")
    fig.text(0.03, 0.915, "Simulation runs the 2026 energy rules: deploy below 270 km/h, "
             "superclip above 305 km/h, 7 MJ harvest limit", fontsize=10.5, color="#c8ccd4")

    # Track map
    ax_map.plot(track.x, track.y, color="#2b3546", lw=9, solid_capstyle="round")
    ax_map.plot(track.x, track.y, color="#4a5568", lw=1)
    ax_map.set_aspect("equal")
    ax_map.margins(x=0.08, y=0.14)
    ax_map.set_axis_off()
    trail, = ax_map.plot([], [], color=RED, lw=3, alpha=0.6)
    car_sim, = ax_map.plot([], [], "o", ms=13, color=RED, mec="white", mew=1.5, zorder=5)
    car_pole, = ax_map.plot([], [], "o", ms=11, color="#e6e8ec", mec=INK, mew=1, alpha=0.9, zorder=4)
    timer = ax_map.text(0.02, 0.02, "", transform=ax_map.transAxes, fontsize=12, family="monospace",
                        va="bottom")
    mode_txt = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes, fontsize=13, weight="bold",
                           ha="left", va="top")

    # Speed
    ax_v.plot(km, v_pole * 3.6, color=GREY, lw=1, alpha=0.7, label=f"Pole lap, {pole['driver']} (real)")
    ax_v.plot(km, best.v_final * 3.6, color=RED, lw=1.2, alpha=0.45, label="Simulation")
    v_sim_dot, = ax_v.plot([], [], "o", color=RED, ms=8)
    v_pole_dot, = ax_v.plot([], [], "o", color=GREY, ms=7)
    v_line_sim, = ax_v.plot([], [], color=RED, lw=2)
    ax_v.set_ylabel("Speed (km/h)")
    ax_v.set_ylim(70, 340)
    ax_v.legend(frameon=False, loc="lower right", fontsize=9, labelcolor="white")

    # Motor power
    ax_k.fill_between(km, 0, k, where=k > 0, color=GREEN, alpha=0.25, step="mid")
    ax_k.fill_between(km, 0, k, where=k < 0, color=GOLD, alpha=0.25, step="mid")
    k_line, = ax_k.plot([], [], color="white", lw=1.4, drawstyle="steps-mid")
    ax_k.axhline(0, color="#3a4456", lw=0.8)
    ax_k.set_ylabel("Motor (kW)")
    ax_k.set_ylim(-400, 400)
    ax_k.text(0.005, 0.86, "deploying", transform=ax_k.transAxes, color=GREEN, fontsize=9)
    ax_k.text(0.005, 0.04, "harvesting", transform=ax_k.transAxes, color=GOLD, fontsize=9)

    # Battery
    ax_b.plot(km, battery, color="#3a4456", lw=1)
    b_line, = ax_b.plot([], [], color="#5fb0ff", lw=2)
    b_dot, = ax_b.plot([], [], "o", color="#5fb0ff", ms=7)
    ax_b.axhline(0, color="#3a4456", lw=0.8, ls=":")
    ax_b.set_ylabel("Battery (MJ)")
    ax_b.set_xlabel("Distance (km)")
    ax_b.set_ylim(battery.min() - 0.4, battery.max() + 0.4)
    ax_b.text(0.005, 0.85, "change in stored energy since the start line", transform=ax_b.transAxes,
              fontsize=9, color="#c8ccd4")
    for ax in (ax_v, ax_k):
        plt.setp(ax.get_xticklabels(), visible=False)

    def frames(speedup, fps):
        end = max(T_sim, T_pole) + 2.0
        return np.arange(0, end * fps / speedup) * speedup / fps

    def draw(time):
        i = position_at(time, t_sim, track, T_sim)
        j = position_at(time, t_pole, track, T_pole)
        car_sim.set_data([track.x[i]], [track.y[i]])
        car_pole.set_data([track.x[j]], [track.y[j]])
        lo = max(0, i - 60)
        trail.set_data(track.x[lo:i + 1], track.y[lo:i + 1])

        v_sim_dot.set_data([km[i]], [best.v_final[i] * 3.6])
        v_pole_dot.set_data([km[j]], [v_pole[j] * 3.6])
        v_line_sim.set_data(km[:i + 1], best.v_final[:i + 1] * 3.6)
        k_line.set_data(km[:i + 1], k[:i + 1])
        b_line.set_data(km[:i + 1], battery[:i + 1])
        b_dot.set_data([km[i]], [battery[i]])

        ts = min(time, T_sim)
        tp = min(time, T_pole)
        done_s = " FINISH" if time >= T_sim else ""
        done_p = " FINISH" if time >= T_pole else ""
        timer.set_text(f"Simulation {ts:6.2f} s{done_s}\nPole lap   {tp:6.2f} s{done_p}")
        if time >= T_sim:
            mode_txt.set_text(f"{T_pole - T_sim:.2f} s quicker\nthan pole")
            mode_txt.set_color("white")
        elif k[i] > 1:
            mode_txt.set_text("DEPLOYING +350 kW")
            mode_txt.set_color(GREEN)
        elif k[i] < -1:
            label = "BRAKING, HARVESTING" if braking[i] else "SUPERCLIPPING"
            mode_txt.set_text(f"{label}\n{k[i]:.0f} kW")
            mode_txt.set_color(GOLD)
        else:
            mode_txt.set_text("ENGINE ONLY")
            mode_txt.set_color("#c8ccd4")
        return ()

    print("Rendering MP4...", flush=True)
    anim = FuncAnimation(fig, draw, frames=frames(speedup_mp4, 30), blit=False)
    anim.save(CHARTS / "lap_animation.mp4", writer=FFMpegWriter(fps=30, bitrate=3500), dpi=100)

    print("Rendering GIF...", flush=True)
    fig.set_size_inches(9.6, 5.4)
    anim = FuncAnimation(fig, draw, frames=frames(speedup_gif, 12), blit=False)
    anim.save(CHARTS / "lap_animation.gif", writer=PillowWriter(fps=12), dpi=75)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
