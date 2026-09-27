"""
Fetches lap telemetry (including track position X, Y) for every driver's
fastest lap in a session, for the lap time simulator.

One driver's line is used as the track geometry (see analysis code), and
every driver's actual lap time and speed trace is used to see how close
each got to the simulator's theoretical optimal lap.

Run this on your own machine, then upload data/ back.
"""
import json
from pathlib import Path

import fastf1
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = Path(__file__).parent.parent / "cache"

# Add more (year, race, session) tuples here to simulate other tracks
SESSIONS_TO_FETCH = [
    (2026, "Silverstone", "Q"),
]


def fetch_session(year, race_name, session_type):
    print(f"\nFetching {year} {race_name} {session_type}")
    session = fastf1.get_session(year, race_name, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    event_slug = f"{year}_{race_name.replace(' ', '')}_{session_type}"
    summaries = []
    drivers = session.laps["Driver"].unique()
    print(f"  Found {len(drivers)} drivers: {', '.join(sorted(drivers))}")

    for driver_code in sorted(drivers):
        try:
            lap = session.laps.pick_drivers(driver_code).pick_fastest()
        except Exception as e:
            print(f"  {driver_code}: could not get fastest lap ({e})")
            continue

        if lap is None or pd.isna(lap["LapTime"]):
            print(f"  {driver_code}: no valid lap, skipping")
            continue

        telemetry = lap.get_telemetry()
        keep_cols = ["Distance", "X", "Y", "Speed", "Throttle", "Brake", "nGear", "RPM", "Time"]
        telemetry = telemetry[[c for c in keep_cols if c in telemetry.columns]].copy()
        telemetry["Time"] = telemetry["Time"].dt.total_seconds()

        out_path = DATA_DIR / f"{event_slug}_{driver_code}_track.csv"
        telemetry.to_csv(out_path, index=False)
        print(f"  {driver_code}: saved ({len(telemetry)} rows), lap time {lap['LapTime']}")

        summaries.append({
            "event": event_slug, "year": year, "race": race_name, "session": session_type,
            "driver": driver_code, "lap_time_seconds": lap["LapTime"].total_seconds(),
            "telemetry_file": out_path.name,
        })

    return summaries


def main():
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    all_summaries = []
    for args in SESSIONS_TO_FETCH:
        all_summaries.extend(fetch_session(*args))

    with open(DATA_DIR / "laps_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nDone: {len(all_summaries)} drivers fetched. Upload everything in {DATA_DIR} back.")


if __name__ == "__main__":
    main()
