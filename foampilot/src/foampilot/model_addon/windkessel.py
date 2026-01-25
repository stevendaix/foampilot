# -*- coding: utf-8 -*-
"""
Windkessel lumped-parameter cardiovascular model (2E / 3E / 4E, serial inertance)

Numerically stable implementation using spline-interpolated flow signals.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class WindkesselModel:
    """
    Windkessel model with serial inertance.

    Supported models:
        - 2-element: Rc = 0, L = 0
        - 3-element: L = 0
        - 4-element: Rc, Rp, C, L > 0

    State variable:
        p2(t): pressure after proximal resistance Rc
    """

    def __init__(
        self,
        t_flow,
        q_flow,
        Rc,
        Rp,
        C,
        L=0.0,
        periodic=True,
    ):
        """
        Parameters
        ----------
        t_flow : array_like
            Time array for flow signal.
        q_flow : array_like
            Flow values Q(t).
        Rc : float
            Proximal resistance.
        Rp : float
            Peripheral resistance.
        C : float
            Compliance.
        L : float, optional
            Inertance (default: 0).
        periodic : bool
            Assume periodic flow signal.
        """

        self.Rc = Rc
        self.Rp = Rp
        self.C = C
        self.L = L
        self.periodic = periodic

        self._t_max = t_flow[-1]

        bc = "periodic" if periodic else "natural"
        self._Q_spline = CubicSpline(t_flow, q_flow, bc_type=bc)

    # ------------------------------------------------------------------
    # Flow signal and derivatives
    # ------------------------------------------------------------------

    def _wrap_time(self, t):
        return t % self._t_max if self.periodic else t

    def Q(self, t):
        return self._Q_spline(self._wrap_time(t))

    def dQdt(self, t):
        return self._Q_spline(self._wrap_time(t), 1)

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _rhs(self, t, y):
        """
        Right-hand side of ODE.

        State:
            y[0] = p2
        """
        p2 = y[0]
        Q = self.Q(t)

        dp2dt = Q / self.C - p2 / (self.Rp * self.C)
        return [dp2dt]

    # ------------------------------------------------------------------
    # Algebraic reconstruction
    # ------------------------------------------------------------------

    def p1(self, t, p2):
        """
        Inlet pressure reconstructed algebraically.
        """
        return p2 + self.Rc * self.Q(t) + self.L * self.dQdt(t)

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    def solve(
        self,
        t_start=0.0,
        t_end=1.0,
        n_steps=2000,
        p2_init=0.0,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    ):
        """
        Solve the Windkessel model.

        Returns
        -------
        result : OdeResult
            SciPy OdeResult with additional fields:
                - result.p1
                - result.p2
        """

        t_eval = np.linspace(t_start, t_end, n_steps)

        sol = solve_ivp(
            self._rhs,
            (t_start, t_end),
            [p2_init],
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        p2 = sol.y[0]
        p1 = self.p1(sol.t, p2)

        # Attach for convenience
        sol.p1 = p1
        sol.p2 = p2

        return sol

