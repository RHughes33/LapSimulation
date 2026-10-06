"""
Energy-aware lap simulation for 2026-spec power units.

The car has a petrol engine (ICE) and an electric motor (MGU-K). The motor
can push the car (deploying battery energy) or charge the battery
(harvesting), so its contribution at each point is one of:

  - Deploy:     +K_max, adding to the engine's power
  - Superclip:  -K_clip, recharging the battery at full throttle, so the
                car's net drive drops and it can slow down with the pedal flat
  - Neutral:    0, engine only
  - Braking:    harvests up to K_max from the energy the brakes would
                otherwise turn into heat

Which one applies depends on speed, set by two switch-over speeds:
deploy below v_deploy, superclip once above v_clip, engine only in between.
Superclipping latches: once it starts it carries on until the car slows
for the next corner (speed back below v_deploy). The telemetry shows real
cars doing exactly this, losing speed steadily all the way to the
braking point, and an un-latched version instead held a flat speed plateau
at the switch-over point, which real cars never show.
Deploying at low speed and harvesting at high speed is the sensible
pattern: a joule spent at low speed buys more time, because the car
spends longer covering each metre there.

Rules (2026, as amended before the Miami GP, so in force at Silverstone):
  - ICE about 400 kW, MGU-K up to 350 kW (deploy and superclip)
  - Qualifying harvest limit: 7 MJ per lap
A lap is legal if the harvest stays under 7 MJ and the energy deployed
doesn't exceed the energy harvested plus whatever the battery is allowed
to run down over the lap (battery_drawdown_j, zero by default: the
battery ends the lap where it started).

Not modelled: the 250 kW deployment limit outside "key acceleration
zones" (on a qualifying lap almost all deployment is from corner exit to
braking point, where 350 kW applies), electrical losses, and battery
state-of-charge limits within the lap.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from .solver import TOP_SPEED_CAP_MS, _brake
from .track import Track
from .vehicle import G, Vehicle


@dataclass
class PowerUnit:
    ice_w: float = 400_000.0
    k_max_w: float = 350_000.0     # MGU-K deploy and braking-harvest limit
    k_clip_w: float = 350_000.0    # superclip harvest power
    harvest_limit_j: float = 7.0e6 # qualifying, per lap
    battery_drawdown_j: float = 0.0


@dataclass
class EnergyLap:
    v_final: np.ndarray
    v_curvature: np.ndarray
    lap_time: float
    deployed_j: float
    harvested_brake_j: float
    harvested_clip_j: float
    k_power: np.ndarray        # MGU-K power at each point (W): + deploy, - harvest
    v_deploy: float
    v_clip: float

    @property
    def harvested_j(self):
        return self.harvested_brake_j + self.harvested_clip_j

    def legal(self, pu: PowerUnit, tol=1.0e3):
        return (self.harvested_j <= pu.harvest_limit_j + tol and
                self.deployed_j <= self.harvested_j + pu.battery_drawdown_j + tol)


def _k_policy(v, pu, v_deploy, v_clip, clipping=False):
    if v < v_deploy:
        return pu.k_max_w
    if clipping or v > v_clip:
        return -pu.k_clip_w
    return 0.0


@njit(cache=True)
def _passes(kappa, v_curv, ds, mass, mu, q_lift, q_drag, max_lat_g, brake_limit_g,
            ice, k_max, k_clip, v_deploy, v_clip, laps):
    """Forward and backward passes, compiled with Numba. Same logic as the
    plain-Python solver: q_lift = 0.5*rho*ClA and q_drag = 0.5*rho*CdA."""
    g = 9.81
    n = kappa.shape[0]
    v_fwd = v_curv.copy()
    clip_state = np.zeros(n, dtype=np.bool_)
    v = v_curv.min()
    clipping = False
    for _ in range(laps):
        for i in range(n):
            if v > v_clip:
                clipping = True
            elif v < v_deploy:
                clipping = False
            clip_state[i] = clipping
            vv = max(v, 1.0)
            if vv < v_deploy:
                k = k_max
            elif clipping or vv > v_clip:
                k = -k_clip
            else:
                k = 0.0
            lat_max = min(mu * (g + q_lift * vv * vv / mass), max_lat_g * g)
            used = min(kappa[i] * vv * vv / lat_max, 1.0)
            grip = mu * (mass * g + q_lift * vv * vv) * np.sqrt(1.0 - used * used)
            a = (min((ice + k) / vv, grip) - q_drag * vv * vv) / mass
            v = min(np.sqrt(max(v * v + 2.0 * a * ds, 1.0)), v_curv[i])
            v_fwd[i] = v
    v_bwd = v_fwd.copy()
    v = v_fwd.min()
    for _ in range(laps):
        for i in range(n - 1, -1, -1):
            lat_max = min(mu * (g + q_lift * v * v / mass), max_lat_g * g)
            used = min(kappa[i] * v * v / lat_max, 1.0)
            dec = (min(mu * (g + q_lift * v * v / mass), brake_limit_g * g) * np.sqrt(1.0 - used * used)
                   + q_drag * v * v / mass)
            v = min(np.sqrt(max(v * v + 2.0 * dec * ds, 0.0)), v_fwd[i])
            v_bwd[i] = v
    return v_fwd, v_bwd, clip_state


def simulate_energy_lap(track: Track, vehicle: Vehicle, pu: PowerUnit,
                        v_deploy: float, v_clip: float, laps: int = 2) -> EnergyLap:
    ds = track.s[1] - track.s[0]
    n = track.n_points
    kappa = np.abs(track.curvature)
    v_curv = np.minimum(vehicle.max_corner_speed(track.curvature), TOP_SPEED_CAP_MS)

    rho = 1.225
    v_fwd, v_bwd, clip_state = _passes(
        kappa, v_curv, ds, vehicle.mass, vehicle.mu, 0.5 * rho * vehicle.lift_area,
        0.5 * rho * vehicle.drag_area, vehicle.max_lateral_g, vehicle.brake_limit_g,
        pu.ice_w, pu.k_max_w, pu.k_clip_w, v_deploy, v_clip, laps)

    v_final = np.minimum(v_curv, np.minimum(v_fwd, v_bwd))
    dt = ds / v_final

    # Energy accounting on the final speed profile
    k_power = np.zeros(n)
    braking = v_final < v_fwd - 0.05
    cornering = np.abs(v_final - v_curv) < 0.05
    on_throttle = ~braking & ~cornering
    for i in np.where(on_throttle)[0]:
        k_power[i] = _k_policy(v_final[i], pu, v_deploy, v_clip, clip_state[i])
    # Braking harvest: the motor takes up to k_max of the braking power
    idx = np.where(braking)[0]
    decel = np.array([_brake(vehicle, kappa[i], v_final[i]) for i in idx])
    braking_power = np.zeros(n)
    braking_power[idx] = vehicle.mass * decel * v_final[idx]
    k_power[braking] = -np.minimum(pu.k_max_w, braking_power[braking])

    deployed = float(np.sum(np.clip(k_power, 0, None) * dt))
    harvest_clip = float(np.sum(-k_power[on_throttle & (k_power < 0)] * dt[on_throttle & (k_power < 0)]))
    harvest_brake = float(np.sum(-k_power[braking] * dt[braking]))

    return EnergyLap(v_final=v_final, v_curvature=v_curv, lap_time=float(np.sum(dt)),
                     deployed_j=deployed, harvested_brake_j=harvest_brake,
                     harvested_clip_j=harvest_clip, k_power=k_power,
                     v_deploy=v_deploy, v_clip=v_clip)


def _search(track, vehicle, pu, vd_grid, vc_grid):
    tried, best = [], None
    for vd in vd_grid:
        for vc in vc_grid:
            if vc < vd:
                continue
            lap = simulate_energy_lap(track, vehicle, pu, vd, vc)
            tried.append(lap)
            if lap.legal(pu) and (best is None or lap.lap_time < best.lap_time):
                best = lap
    return best, tried


def optimise_deployment(track: Track, vehicle: Vehicle, pu: PowerUnit, step_kmh=5.0):
    """Find the fastest legal lap over the two switch-over speeds by a full
    grid search. A coarse-then-fine search was tried first and missed the
    true optimum: the lap time landscape has several separate dips, so
    refining around the best coarse point can settle in the wrong one.
    Returns the best lap and every lap tried (for the strategy map)."""
    vd = np.arange(100, 361, step_kmh) / 3.6
    vc = np.arange(150, 401, step_kmh) / 3.6
    return _search(track, vehicle, pu, vd, vc)


def unconstrained_lap(track, vehicle, pu):
    """No energy limits at all: full deployment everywhere, no clipping.
    Shows what the 2026 energy rules cost over a lap."""
    return simulate_energy_lap(track, vehicle, pu, v_deploy=1e3, v_clip=2e3)
