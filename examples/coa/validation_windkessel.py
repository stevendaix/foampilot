
# -*- coding: utf-8 -*-
"""
Validation script for Windkessel class against reference waveform
(Morphological / phase / diastolic decay validation)
https://github.com/TS-CUBED/PySeuille/tree/main
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from windkessel import Windkessel


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def affine_match(reference, target):
    """Affine rescaling of reference signal to match target amplitude range."""
    ref = reference - np.min(reference)
    ref /= np.max(ref)
    return ref * (np.max(target) - np.min(target)) + np.min(target)


def normalized_rms_error(ref, target):
    return np.linalg.norm(ref - target) / np.linalg.norm(target)


def diastolic_tau(t, p):
    """Estimate diastolic time constant assuming exponential decay."""
    peak_idx = np.argmax(p)
    t_d = t[peak_idx:] - t[peak_idx]
    p_d = p[peak_idx:] - np.min(p[peak_idx:])

    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)

    popt, _ = curve_fit(exp_decay, t_d, p_d, maxfev=5000)
    return popt[1]


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

# Flow (Csc)
flow_data = np.loadtxt("data_typec_q.csv", delimiter=",", skiprows=1)
t_flow = flow_data[:, 0]
q_flow = flow_data[:, 1] * 1e-6  # ml/s → m³/s

# Pressure (Csvp)
pressure_data = np.loadtxt("data_typec_p.csv", delimiter=",", skiprows=1)
t_p = pressure_data[:, 0]
p_ref = pressure_data[:, 1]  # mmHg


# ---------------------------------------------------------------------
# Windkessel simulation
# ---------------------------------------------------------------------

wk = Windkessel(
    t_flow=t_flow,
    q_flow=q_flow,
    Rc=1.2e7,
    Rp=1.5e8,
    C=1.8e-9,
    L=5e4,
    periodic=True,
)

T = t_flow[-1]

sol = wk.solve(
    t_start=0.0,
    t_end=5 * T,
    n_steps=5000,
)

# Remove transient
startN = int(0.8 * len(sol.t))
t_sim = sol.t[startN:] - sol.t[startN]
P_sim = sol.p1[startN:]


# ---------------------------------------------------------------------
# Reference pressure processing
# ---------------------------------------------------------------------

# Interpolate reference pressure on simulation grid
P_ref_interp = np.interp(t_sim, t_p, p_ref)

# Shape-based comparison
P_ref = affine_match(P_ref_interp, P_sim)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

nrms = normalized_rms_error(P_ref, P_sim)

peak_ref = np.argmax(P_ref)
peak_sim = np.argmax(P_sim)
dt_peak = t_sim[peak_ref] - t_sim[peak_sim]

tau_sim = diastolic_tau(t_sim, P_sim)
tau_ref = diastolic_tau(t_sim, P_ref)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 4))
plt.plot(t_sim, P_sim, label="Windkessel (simulated)", linewidth=2)
plt.plot(t_sim, P_ref, "--", label="Reference (scaled)", color="grey")
plt.xlabel("Time [s]")
plt.ylabel("Pressure [a.u.]")
plt.legend()
plt.grid()
plt.xlim(0.0, T)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("=== Windkessel validation report ===")
print(f"NRMS error           : {nrms * 100:.2f} %")
print(f"Peak time shift      : {dt_peak * 1000:.2f} ms")
print(f"Diastolic tau (sim)  : {tau_sim:.4f} s")
print(f"Diastolic tau (ref)  : {tau_ref:.4f} s")
print(f"Relative tau error   : {abs(tau_sim - tau_ref) / tau_ref * 100:.2f} %")