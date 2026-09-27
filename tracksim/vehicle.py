"""
Point-mass vehicle model for a 2026-spec F1 car.

Mass comes from the FIA's 2026 minimum weight. Every other parameter is
calibrated from real 2026 Silverstone qualifying telemetry (see
calibrate.py), because the published figures didn't hold up when tested:

  - Cornering grip (mu) and downforce area (lift_area): fitted across every
    grip-limited corner at once.
  - Effective power and drag area: fitted from full-throttle acceleration
    on the straights. Effective power comes out well below the 2026
    headline figure (~750 kW), which matches how the 2026 rules work: the
    electric motor's output is limited by how much battery energy is
    available per lap, so it can't run at full power all lap.
  - Braking limit: fitted so simulated braking zones match real ones.

These are "effective" values: each absorbs things the model doesn't
represent separately (energy management, tyre load sensitivity, the
limits of 4 Hz telemetry). They reproduce how the car behaves, which is
what a lap simulator needs, but they shouldn't be read as the car's true
aerodynamic coefficients.
"""
from dataclasses import dataclass

import numpy as np

AIR_DENSITY = 1.225  # kg/m^3
G = 9.81             # m/s^2


@dataclass
class Vehicle:
    mass: float = 768.0          # kg, FIA 2026 minimum (car + driver + tyres)
    mu: float = 1.707            # cornering grip, fitted
    lift_area: float = 6.52      # m^2, effective downforce area (ClA), fitted
    power_w: float = 455_000.0   # W, effective power, fitted
    drag_area: float = 0.92      # m^2, drag area (CdA), fitted
    brake_limit_g: float = 2.5   # peak braking from grip, fitted (drag adds on top)
    max_lateral_g: float = 8.0   # backstop only: never reached at the fitted values
                                 # on this track, stops gentle kinks giving unbounded speed

    def downforce(self, v):
        return 0.5 * AIR_DENSITY * self.lift_area * v**2

    def drag_force(self, v):
        return 0.5 * AIR_DENSITY * self.drag_area * v**2

    def max_corner_speed(self, curvature) -> np.ndarray:
        """Speed at which the cornering force needed equals the grip available.

        Solving mu*(m*g + 0.5*rho*ClA*v^2) = m*kappa*v^2 for v gives
            v^2 = mu*m*g / (m*kappa - 0.5*mu*rho*ClA)
        If the denominator is <= 0, downforce alone covers the corner at any
        speed; max_lateral_g then acts as a physical ceiling.
        """
        kappa = np.maximum(np.abs(curvature), 1e-6)
        denom = self.mass * kappa - 0.5 * self.mu * AIR_DENSITY * self.lift_area
        v2_grip = np.where(denom > 1e-9, self.mu * self.mass * G / np.where(denom > 1e-9, denom, 1.0), np.inf)
        v2 = np.minimum(v2_grip, self.max_lateral_g * G / kappa)
        return np.sqrt(v2)

    def max_traction_accel(self, v):
        """Forward acceleration: power-limited or grip-limited, whichever is
        lower, minus drag. Grip isn't shared with cornering (a standard
        simplification in point-mass lap simulators, see README)."""
        v = np.maximum(v, 1.0)
        drive = np.minimum(self.power_w / v, self.mu * (self.mass * G + self.downforce(v)))
        return (drive - self.drag_force(v)) / self.mass

    def max_brake_decel(self, v):
        """Deceleration: grip-based braking up to the fitted limit, plus drag."""
        grip = np.minimum(self.mu * (G + self.downforce(v) / self.mass), self.brake_limit_g * G)
        return grip + self.drag_force(v) / self.mass
