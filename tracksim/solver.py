"""
Quasi-steady-state speed profile solver with combined grip.

At every point around the lap, the achievable speed is the smallest of:
  1. The cornering limit: as fast as the corner allows from grip alone
  2. A forward pass: accelerating as hard as possible from behind
  3. A backward pass: braking as hard as possible for what's ahead

Combined grip (the friction circle): a tyre has one grip budget shared
between cornering and braking or accelerating. So in the forward and
backward passes, the grip available for accelerating or braking at each
point is reduced by how much of the budget cornering is already using:

    available longitudinal grip = full grip * sqrt(1 - (lateral used / lateral max)^2)

An earlier version let the tyres use full grip in both directions at
once. That let the car exit every corner too fast and was the main reason
it beat the real lap time by several seconds.
"""
from dataclasses import dataclass

import numpy as np

from .track import Track
from .vehicle import G, Vehicle

TOP_SPEED_CAP_MS = 130.0  # ~468 km/h, only so dead-straight points have a finite limit


@dataclass
class LapResult:
    track: Track
    v_curvature: np.ndarray
    v_forward: np.ndarray
    v_backward: np.ndarray
    v_final: np.ndarray
    lap_time: float
    ds: float

    @property
    def s(self):
        return self.track.s

    def limiting_mode(self, tol=0.05):
        """What limits the car at each point: 'corner', 'braking' or 'accelerating'."""
        mode = np.full(len(self.v_final), "accelerating", dtype=object)
        mode[self.v_final < self.v_forward - tol] = "braking"
        mode[np.abs(self.v_final - self.v_curvature) < tol] = "corner"
        return mode


def _grip_share(vehicle: Vehicle, kappa, v, combined=True):
    """Fraction of the grip budget left for braking/accelerating."""
    if not combined:
        return 1.0
    lat_max = min(vehicle.mu * (G + vehicle.downforce(v) / vehicle.mass), vehicle.max_lateral_g * G)
    used = min(kappa * v**2 / lat_max, 1.0)
    return np.sqrt(1.0 - used**2)


def _accel(vehicle: Vehicle, kappa, v, combined=True):
    v = max(v, 1.0)
    grip = vehicle.mu * (vehicle.mass * G + vehicle.downforce(v)) * _grip_share(vehicle, kappa, v, combined)
    drive = min(vehicle.power_w / v, grip)
    return (drive - vehicle.drag_force(v)) / vehicle.mass


def _brake(vehicle: Vehicle, kappa, v, combined=True):
    grip = min(vehicle.mu * (G + vehicle.downforce(v) / vehicle.mass), vehicle.brake_limit_g * G)
    return grip * _grip_share(vehicle, kappa, v, combined) + vehicle.drag_force(v) / vehicle.mass


def simulate_lap(track: Track, vehicle: Vehicle, laps: int = 3, combined_grip: bool = True) -> LapResult:
    ds = track.s[1] - track.s[0]
    n = track.n_points
    kappa = np.abs(track.curvature)
    v_curv = np.minimum(vehicle.max_corner_speed(track.curvature), TOP_SPEED_CAP_MS)

    # Passes run several times round the closed lap, so the speed at the
    # start/finish line is consistent rather than an arbitrary starting value
    v_fwd = v_curv.copy()
    v = v_curv.min()
    for _ in range(laps):
        for i in range(n):
            v = min(np.sqrt(max(v**2 + 2 * _accel(vehicle, kappa[i], v, combined_grip) * ds, 0.0)), v_curv[i])
            v_fwd[i] = v

    v_bwd = v_fwd.copy()
    v = v_fwd.min()
    for _ in range(laps):
        for i in range(n - 1, -1, -1):
            v = min(np.sqrt(max(v**2 + 2 * _brake(vehicle, kappa[i], v, combined_grip) * ds, 0.0)), v_fwd[i])
            v_bwd[i] = v

    v_final = np.minimum(v_curv, np.minimum(v_fwd, v_bwd))
    return LapResult(track=track, v_curvature=v_curv, v_forward=v_fwd, v_backward=v_bwd,
                     v_final=v_final, lap_time=float(np.sum(ds / v_final)), ds=ds)
