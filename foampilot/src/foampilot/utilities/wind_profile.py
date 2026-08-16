"""Wind profile utilities for building_aero and atmospheric boundary-layer workflows.

This module provides:
- logarithmic wind profile
- friction velocity from reference wind speed
- turbulence quantities (TKE, epsilon) from the log profile
- wind-direction rotation helpers
"""

from __future__ import annotations

import math

KAPPA = 0.41
Z_REF = 10.0


def log_wind_profile(z, z0, u_star, kappa=KAPPA):
    """Compute the logarithmic wind profile at height *z*."""
    return u_star / kappa * math.log(max(z, z0) / z0)


def friction_velocity(u_ref, z_ref, z0, kappa=KAPPA):
    """Compute the friction velocity from a reference wind speed."""
    return u_ref * kappa / math.log(z_ref / z0)


def turbulence_quantities(u_star, z, z0, kappa=KAPPA, intensity=0.1):
    """Return (TKE, epsilon) for a log-wind-profile inlet condition."""
    u = log_wind_profile(z, z0, u_star, kappa)
    k = 1.5 * (intensity * u) ** 2
    eps = (u_star ** 3) / (kappa * max(z, z0))
    return k, eps


def rotation_angle_for_wind_direction(direction_deg):
    """Return the rotation angle corresponding to a wind direction.

    Currently this is an identity mapping; override for custom conventions.
    """
    return direction_deg
