"""
Minimal wind profile utilities for building_aero example.
"""

KAPPA = 0.41
Z_REF = 10.0


def log_wind_profile(z, z0, u_star, kappa=KAPPA):
    return u_star / kappa * __import__("math").log(max(z, z0) / z0)


def friction_velocity(u_ref, z_ref, z0, kappa=KAPPA):
    return u_ref * kappa / __import__("math").log(z_ref / z0)


def turbulence_quantities(u_star, z, z0, kappa=KAPPA, intensity=0.1):
    u = log_wind_profile(z, z0, u_star, kappa)
    k = 1.5 * (intensity * u) ** 2
    eps = (u_star ** 3) / (kappa * max(z, z0))
    return k, eps


def rotation_angle_for_wind_direction(direction_deg):
    return direction_deg
