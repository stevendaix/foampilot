# -*- coding: utf-8 -*-
"""
Windkessel lumped-parameter cardiovascular model (2E / 3E / 4E, serial inertance)

Numerically stable implementation using spline-interpolated flow signals.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Union, Optional
import warnings


@dataclass
class WindkesselResult:
    """
    Extended OdeResult with Windkessel-specific pressure fields.
    
    Attributes
    ----------
    t : ndarray
        Time points.
    p1 : ndarray
        Inlet pressure (aortic) [Pa].
    p2 : ndarray
        Capacitor pressure (after Rc) [Pa].
    success : bool
        Whether the solver completed successfully.
    message : str
        Solver termination message.
    """
    t: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    success: bool
    message: str


class Windkessel:  # ✅ Renommé pour correspondre à l'import du validation script
    """
    Windkessel model with serial inertance.

    Supported configurations:
        - 2-element: Rc = 0, L = 0
        - 3-element: L = 0, Rc > 0
        - 4-element: Rc, Rp, C, L > 0

    Governing equations:
        C * dp2/dt + p2/Rp = Q(t)           [Compliance node]
        p1 = p2 + Rc*Q + L*dQ/dt            [Inlet pressure reconstruction]

    State variable:
        p2(t): pressure across the compliance element [Pa]

    Expected units (SI):
        - Time: seconds [s]
        - Flow: cubic meters per second [m³/s]
        - Pressure: Pascals [Pa]
        - Resistance: Pa·s/m³
        - Compliance: m³/Pa
        - Inertance: Pa·s²/m³
    """

    def __init__(
        self,
        t_flow: np.ndarray,
        q_flow: np.ndarray,
        Rc: float,
        Rp: float,
        C: float,
        L: float = 0.0,
        periodic: bool = True,
    ):
        """
        Initialize the Windkessel model.

        Parameters
        ----------
        t_flow : array_like
            Time array for flow signal [s]. Must be monotonically increasing.
        q_flow : array_like
            Flow values Q(t) [m³/s].
        Rc : float
            Proximal/characteristic resistance [Pa·s/m³]. Must be >= 0.
        Rp : float
            Peripheral resistance [Pa·s/m³]. Must be > 0.
        C : float
            Compliance [m³/Pa]. Must be > 0.
        L : float, optional
            Inertance [Pa·s²/m³]. Must be >= 0 (default: 0).
        periodic : bool, optional
            Assume periodic flow signal for spline interpolation (default: True).

        Raises
        ------
        ValueError
            If parameters are physically invalid or arrays mismatch.
        """
        # Validate arrays
        t_flow = np.asarray(t_flow, dtype=float)
        q_flow = np.asarray(q_flow, dtype=float)
        
        if len(t_flow) != len(q_flow):
            raise ValueError(f"t_flow and q_flow must have same length, got {len(t_flow)} and {len(q_flow)}")
        if len(t_flow) < 2:
            raise ValueError("Flow signal must contain at least 2 points")
        if not np.all(np.diff(t_flow) > 0):
            raise ValueError("t_flow must be strictly monotonically increasing")

        # Validate physical parameters
        if Rp <= 0:
            raise ValueError(f"Rp must be positive, got {Rp}")
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if Rc < 0:
            raise ValueError(f"Rc must be non-negative, got {Rc}")
        if L < 0:
            raise ValueError(f"L must be non-negative, got {L}")

        self.Rc = float(Rc)
        self.Rp = float(Rp)
        self.C = float(C)
        self.L = float(L)
        self.periodic = bool(periodic)

        self._t_min = float(t_flow[0])
        self._t_max = float(t_flow[-1])
        self._period = self._t_max - self._t_min

        # Create spline with appropriate boundary conditions
        bc_type = "periodic" if periodic else "natural"
        try:
            self._Q_spline = CubicSpline(t_flow, q_flow, bc_type=bc_type)
        except ValueError as e:
            raise ValueError(f"Failed to create flow spline: {e}")

    # ------------------------------------------------------------------
    # Flow signal and derivatives
    # ------------------------------------------------------------------

    def _wrap_time(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Wrap time to [t_min, t_max] for periodic signals."""
        if not self.periodic:
            return t
        # Handle both scalar and array inputs
        t_wrapped = np.mod(t - self._t_min, self._period) + self._t_min
        return t_wrapped if np.ndim(t) > 0 else float(t_wrapped)

    def Q(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate flow Q(t) at time(s) t [m³/s]."""
        return self._Q_spline(self._wrap_time(t))

    def dQdt(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate flow derivative dQ/dt at time(s) t [m³/s²]."""
        return self._Q_spline(self._wrap_time(t), 1)

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _rhs(self, t: float, y: list[float]) -> list[float]:
        """
        Right-hand side of the ODE system.

        State:
            y[0] = p2 : pressure across compliance [Pa]

        Equation:
            dp2/dt = Q(t)/C - p2/(Rp*C)
        """
        p2 = y[0]
        q_val = self.Q(t)
        
        # Guard against division by zero (should never happen if C > 0)
        if self.C == 0:
            raise RuntimeError("Compliance C is zero - model undefined")
            
        dp2dt = q_val / self.C - p2 / (self.Rp * self.C)
        return [dp2dt]

    # ------------------------------------------------------------------
    # Algebraic reconstruction
    # ------------------------------------------------------------------

    def p1_from_p2(self, t: Union[float, np.ndarray], p2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Reconstruct inlet pressure p1 from p2 using algebraic relation.

        p1 = p2 + Rc*Q + L*dQ/dt

        Parameters
        ----------
        t : float or array_like
            Time point(s) [s].
        p2 : float or array_like
            Capacitor pressure(s) [Pa].

        Returns
        -------
        p1 : float or ndarray
            Inlet pressure(s) [Pa].
        """
        t_arr = np.asarray(t)
        p2_arr = np.asarray(p2)
        
        q_val = self.Q(t_arr)
        dqdt_val = self.dQdt(t_arr)
        
        return p2_arr + self.Rc * q_val + self.L * dqdt_val

    # Alias for backward compatibility with original API
    p1 = p1_from_p2

    # ------------------------------------------------------------------
    # Initial condition estimation
    # ------------------------------------------------------------------

    def estimate_steady_state_p2(self, n_samples: int = 100) -> float:
        """
        Estimate a reasonable initial condition for p2 by averaging
        the quasi-steady-state solution over one cycle.

        Quasi-steady approximation: p2 ≈ Rp * Q (ignoring compliance dynamics)
        """
        t_samples = np.linspace(self._t_min, self._t_max, n_samples, endpoint=False)
        q_samples = self.Q(t_samples)
        # Steady-state approximation: p2 = Rp * Q when dp2/dt ≈ 0
        p2_estimates = self.Rp * q_samples
        return float(np.mean(p2_estimates))

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    def solve(
        self,
        t_start: float = 0.0,
        t_end: float = 1.0,
        n_steps: int = 2000,
        p2_init: Optional[float] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        estimate_ic: bool = True,
    ) -> WindkesselResult:
        """
        Solve the Windkessel model ODE system.

        Parameters
        ----------
        t_start : float, optional
            Start time [s] (default: 0.0).
        t_end : float, optional
            End time [s] (default: 1.0).
        n_steps : int, optional
            Number of output time points (default: 2000).
        p2_init : float, optional
            Initial condition for p2 [Pa]. If None and estimate_ic=True,
            uses estimate_steady_state_p2(). Otherwise defaults to 0.
        method : str, optional
            ODE solver method passed to scipy.integrate.solve_ivp (default: "RK45").
        rtol, atol : float, optional
            Relative and absolute tolerances for the solver.
        estimate_ic : bool, optional
            Whether to estimate a better initial condition if p2_init is None.

        Returns
        -------
        result : WindkesselResult
            Extended OdeResult with fields:
                - t: time points [s]
                - p1: inlet pressure [Pa]
                - p2: capacitor pressure [Pa]
                - success, message: solver status

        Raises
        ------
        RuntimeError
            If the ODE solver fails to converge.
        """
        if t_end <= t_start:
            raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start})")
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")

        # Handle initial condition
        if p2_init is None:
            if estimate_ic:
                p2_init = self.estimate_steady_state_p2()
            else:
                p2_init = 0.0
                warnings.warn(
                    "Using p2_init=0.0. For faster convergence to periodic steady-state, "
                    "set estimate_ic=True or provide a better initial guess.",
                    UserWarning,
                    stacklevel=2
                )

        t_eval = np.linspace(t_start, t_end, n_steps)

        sol = solve_ivp(
            fun=self._rhs,
            t_span=(t_start, t_end),
            y0=[p2_init],
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        p2 = sol.y[0]
        p1 = self.p1_from_p2(sol.t, p2)

        return WindkesselResult(
            t=sol.t,
            p1=p1,
            p2=p2,
            success=sol.success,
            message=sol.message,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation for debugging."""
        model_type = (
            "2-element" if self.Rc == 0 and self.L == 0 else
            "3-element" if self.L == 0 else
            "4-element"
        )
        return (
            f"Windkessel(model='{model_type}', "
            f"Rc={self.Rc:.3e} Pa·s/m³, Rp={self.Rp:.3e} Pa·s/m³, "
            f"C={self.C:.3e} m³/Pa, L={self.L:.3e} Pa·s²/m³, "
            f"periodic={self.periodic})"
        )

    @property
    def time_constant(self) -> float:
        """Diastolic time constant tau = Rp * C [s]."""
        return self.Rp * self.C


# ✅ Alias pour rétro-compatibilité si besoin
WindkesselModel = Windkessel