import numpy as np
from typing import Optional


def calc_region_heat_flux(
    T_field: np.ndarray,
    thermal_conductivity: float,
    dx: float = 1.0,
) -> np.ndarray:
    """Calculate heat flux magnitude from a temperature field using finite differences.

    Parameters
    ----------
    T_field : np.ndarray
        Temperature field (1D, 2D, or 3D).
    thermal_conductivity : float
        Thermal conductivity k in W/(m·K).
    dx : float, optional
        Grid spacing in meters.

    Returns
    -------
    np.ndarray
        Heat flux magnitude field (same shape as T_field).
    """
    grad_T = np.gradient(T_field, dx)
    if np.ndim(grad_T) == 0:
        return np.array([0.0])
    grad_T_arr = np.array(grad_T)
    if grad_T_arr.ndim == 1:
        q_mag = np.abs(thermal_conductivity * grad_T_arr)
        return q_mag
    q_mag = np.abs(thermal_conductivity * np.linalg.norm(grad_T, axis=0))
    return q_mag


def calc_interface_heat_flux(
    T_fluid: np.ndarray,
    T_solid: np.ndarray,
    h: float,
    area: float,
    dx_fluid: float = 1.0,
    dx_solid: float = 1.0,
    k_fluid: float = 0.026,
    k_solid: float = 50.0,
) -> dict:
    """Calculate the heat flux across a fluid-solid interface.

    Parameters
    ----------
    T_fluid : np.ndarray
        Temperature field on the fluid side of the interface.
    T_solid : np.ndarray
        Temperature field on the solid side of the interface.
    h : float
        Heat transfer coefficient in W/(m²·K).
    area : float
        Interface area in m².
    dx_fluid : float, optional
        Grid spacing on the fluid side.
    dx_solid : float, optional
        Grid spacing on the solid side.
    k_fluid : float, optional
    k_solid : float, optional

    Returns
    -------
    dict
        Dictionary with keys ``'q_total'``, ``'q_conv'``, ``'q_cond_fluid'``,
        ``'q_cond_solid'``, and ``'T_interface'``.
    """
    T_fluid_avg = float(np.mean(T_fluid))
    T_solid_avg = float(np.mean(T_solid))
    T_interface = (T_fluid_avg + T_solid_avg) / 2.0

    q_conv = h * (T_solid_avg - T_fluid_avg) * area

    grad_T_fluid = float(np.gradient(T_fluid, dx_fluid).mean())
    q_cond_fluid = k_fluid * grad_T_fluid * area

    grad_T_solid = float(np.gradient(T_solid, dx_solid).mean())
    q_cond_solid = k_solid * grad_T_solid * area

    q_total = q_conv + q_cond_fluid + q_cond_solid

    return {
        "q_total": q_total,
        "q_conv": q_conv,
        "q_cond_fluid": q_cond_fluid,
        "q_cond_solid": q_cond_solid,
        "T_interface": T_interface,
    }


def calc_nusselt_number(
    q_wall: float,
    L: float,
    k_fluid: float,
    T_wall: float,
    T_bulk: float,
) -> float:
    """Calculate the Nusselt number from wall heat flux.

    Nu = q_wall * L / (k_fluid * (T_wall - T_bulk))

    Parameters
    ----------
    q_wall : float
        Wall heat flux in W/m².
    L : float
        Characteristic length in m.
    k_fluid : float
        Fluid thermal conductivity in W/(m·K).
    T_wall : float
        Wall temperature in K.
    T_bulk : float
        Bulk fluid temperature in K.

    Returns
    -------
    float
        Nusselt number (dimensionless).
    """
    delta_T = T_wall - T_bulk
    if abs(delta_T) < 1e-10:
        return 0.0
    return abs(q_wall) * L / (k_fluid * abs(delta_T))


def calc_thermal_boundary_layer_thickness(
    T_wall: float,
    T_bulk: float,
    T_field: np.ndarray,
    x_positions: np.ndarray,
    threshold: float = 0.99,
) -> float:
    """Estimate the thermal boundary layer thickness.

    Parameters
    ----------
    T_wall : float
        Wall temperature in K.
    T_bulk : float
        Bulk fluid temperature in K.
    T_field : np.ndarray
        Temperature field along the wall-normal direction.
    x_positions : np.ndarray
        Position array corresponding to T_field.
    threshold : float, optional
        Fraction defining the boundary layer edge (default: 0.99).

    Returns
    -------
    float
        Thermal boundary layer thickness in meters.
    """
    T_edge = T_wall - threshold * (T_wall - T_bulk)
    wall_points = x_positions[T_field > T_edge]
    edge_points = x_positions[T_field <= T_edge]

    if len(wall_points) == 0 or len(edge_points) == 0:
        return 0.0

    delta = float(np.min(np.abs(wall_points[:, np.newaxis] - edge_points[np.newaxis, :])))
    return delta


def calc_heat_transfer_coefficient(
    q_wall: float,
    T_wall: float,
    T_bulk: float,
) -> float:
    """Calculate the convective heat transfer coefficient.

    h = q_wall / (T_wall - T_bulk)

    Parameters
    ----------
    q_wall : float
        Wall heat flux in W/m².
    T_wall : float
        Wall temperature in K.
    T_bulk : float
        Bulk fluid temperature in K.

    Returns
    -------
    float
        Heat transfer coefficient h in W/(m²·K).
    """
    delta_T = T_wall - T_bulk
    if abs(delta_T) < 1e-10:
        return 0.0
    return q_wall / delta_T


def calc_total_heat_balance(
    Q_in: float,
    Q_out: float,
    Q_stored: float,
    tolerance: float = 1e-6,
) -> dict:
    """Verify the global energy conservation (heat balance).

    In a steady-state CHT simulation the heat entering the domain
    must equal the heat leaving plus the rate of energy storage.
    For transient cases, ``Q_stored`` represents the change in stored
    energy over the time step.

    Parameters
    ----------
    Q_in : float
        Total heat influx [W] (or [J] over the time step).
    Q_out : float
        Total heat outflux [W] (or [J] over the time step).
    Q_stored : float
        Rate of change of stored energy [W] (or [J] over the time step).
        Positive means energy is being stored in the domain.
    tolerance : float, optional
        Relative tolerance for the balance check (default 1e-6).

    Returns
    -------
    dict
        Keys: ``'balance'`` (Q_in - Q_out - Q_stored),
        ``'balance_error'`` (relative error),
        ``'is_conserved'`` (bool).
    """
    balance = Q_in - Q_out - Q_stored
    denom = max(abs(Q_in), abs(Q_out), 1e-10)
    balance_error = abs(balance) / denom
    is_conserved = balance_error < tolerance
    return {
        "balance": float(balance),
        "balance_error": float(balance_error),
        "is_conserved": bool(is_conserved),
    }


def calc_temperature_contour(
    T_field: np.ndarray,
    levels: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> dict:
    """Extract isotherms (contour lines) from a 2-D temperature field.

    Parameters
    ----------
    T_field : np.ndarray
        2-D temperature field [K].
    levels : int, optional
        Number of contour levels.  If ``None``, uses 10.
    vmin : float, optional
        Lower temperature bound.  Defaults to ``T_field.min()``.
    vmax : float, optional
        Upper temperature bound.  Defaults to ``T_field.max()``.

    Returns
    -------
    dict
        Keys: ``'levels'`` (list of contour levels),
        ``'T_min'``, ``'T_max'``, ``'delta_T'`` (range).
    """
    levels = levels or 10
    vmin = vmin if vmin is not None else float(np.min(T_field))
    vmax = vmax if vmax is not None else float(np.max(T_field))
    contour_levels = np.linspace(vmin, vmax, levels)
    return {
        "levels": contour_levels.tolist(),
        "T_min": vmin,
        "T_max": vmax,
        "delta_T": float(vmax - vmin),
    }


def calc_thermal_resistance(
    T_hot: float,
    T_cold: float,
    Q_total: float,
) -> float:
    """Calculate the overall thermal resistance between two regions.

    R = ΔT / Q  [K/W]

    Parameters
    ----------
    T_hot : float
        Temperature on the hot side (K).
    T_cold : float
        Temperature on the cold side (K).
    Q_total : float
        Total heat transfer rate between the two sides (W).

    Returns
    -------
    float
        Thermal resistance in K/W.
    """
    delta_T = T_hot - T_cold
    if abs(Q_total) < 1e-10:
        return 0.0
    return delta_T / Q_total