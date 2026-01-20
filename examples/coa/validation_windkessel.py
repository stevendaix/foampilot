# -*- coding: utf-8 -*-
"""
Validation script for Windkessel class against reference waveform
(e.g. Stergiopulos et al.)

Validation type:
- Morphological (shape-based)
- Phase consistency
- Diastolic decay consistency
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from windkessel import Windkessel

def affine_match(reference, target):
    """
    Affine rescaling of reference signal to match target amplitude range.
    """
    ref = reference - np.min(reference)
    ref /= np.max(ref)
    return ref * (np.max(target) - np.min(target)) + np.min(target)


def normalized_rms_error(ref, target):
    return np.linalg.norm(ref - target) / np.linalg.norm(target)


def diastolic_tau(t, p):
    """
    Estimate diastolic time constant assuming exponential decay.
    """

    # take diastolic part (after systolic peak)
    peak_idx = np.argmax(p)
    t_d = t[peak_idx:]
    p_d = p[peak_idx:]

    # shift for numerical stability
    p_d = p_d - np.min(p_d)

    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)

    popt, _ = curve_fit(
        exp_decay,
        t_d - t_d[0],
        p_d,
        maxfev=5000,
    )

    return popt[1]



# Windkessel parameters (example)
wk = Windkessel(
    t_flow=t_ref,
    q_flow=I_ref,
    Rc=1.2e7,
    Rp=1.5e8,
    C=1.8e-9,
    L=5e4,
)

sol = wk.solve(
    t_start=0.0,
    t_end=10.0,
    n_steps=5000,
)

# Remove transient
startN = int(0.8 * len(sol.t))

t = sol.t[startN:]
P_sim = sol.p1[startN:]


P_ref = affine_match(I_ref[: len(t)], P_sim)

nrms = normalized_rms_error(P_ref, P_sim)

peak_ref = np.argmax(P_ref)
peak_sim = np.argmax(P_sim)

dt_peak = t[peak_ref] - t[peak_sim]

tau_sim = diastolic_tau(t, P_sim)
tau_ref = diastolic_tau(t, P_ref)

plt.figure(figsize=(8, 4))
plt.plot(t, P_sim, label="Windkessel (simulated)", linewidth=2)
plt.plot(t, P_ref, "--", label="Reference (scaled)", color="grey")
plt.xlabel("Time [s]")
plt.ylabel("Pressure [a.u.]")
plt.legend()
plt.grid()
plt.xlim(t[0], t[0] + 1.0)
plt.tight_layout()
plt.show()

print("=== Windkessel validation report ===")
print(f"NRMS error           : {nrms * 100:.2f} %")
print(f"Peak time shift      : {dt_peak * 1000:.2f} ms")
print(f"Diastolic tau (sim)  : {tau_sim:.4f} s")
print(f"Diastolic tau (ref)  : {tau_ref:.4f} s")
print(f"Relative tau error   : {abs(tau_sim - tau_ref) / tau_ref * 100:.2f} %")

