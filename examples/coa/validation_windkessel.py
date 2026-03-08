# -*- coding: utf-8 -*-
"""
Validation script for Windkessel class against reference waveform
(Morphological / phase / diastolic decay validation)
https://github.com/TS-CUBED/PySeuille/tree/main 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import sys
from pathlib import Path

# Add foampilot to path
sys.path.insert(0, '/home/steven/foampilot')

from foampilot.model_addon.windkessel import Windkessel


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def affine_match(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Affine rescaling of reference signal to match target amplitude range."""
    ref = reference - np.min(reference)
    if np.max(ref) == 0:
        return np.zeros_like(reference)
    ref = ref / np.max(ref)
    return ref * (np.max(target) - np.min(target)) + np.min(target)


def normalized_rms_error(ref: np.ndarray, target: np.ndarray) -> float:
    """Compute normalized RMS error between two signals."""
    norm_target = np.linalg.norm(target)
    if norm_target == 0:
        return np.inf
    return np.linalg.norm(ref - target) / norm_target


def diastolic_tau(t: np.ndarray, p: np.ndarray, 
                  systolic_fraction: float = 0.3) -> float:
    """
    Estimate diastolic time constant assuming exponential decay.
    
    Parameters
    ----------
    t : array-like
        Time vector [s]
    p : array-like
        Pressure vector [same units as output]
    systolic_fraction : float, optional
        Fraction of cardiac cycle to skip to avoid systolic phase
        (default: 0.3, meaning start fitting at 30% after peak)
    
    Returns
    -------
    tau : float
        Diastolic time constant [s], or np.nan if fitting fails
    """
    peak_idx = np.argmax(p)
    
    # Start fitting after systolic phase to avoid dicrotic notch interference
    start_idx = peak_idx + int(systolic_fraction * (len(p) - peak_idx))
    start_idx = min(start_idx, len(p) - 10)  # Ensure enough points remain
    
    if start_idx >= len(p) - 10:
        return np.nan
    
    t_d = t[start_idx:] - t[start_idx]
    p_d = p[start_idx:] - np.min(p[start_idx:])
    
    # Avoid log(0) or negative values
    mask = p_d > 1e-6 * np.max(p_d)
    if np.sum(mask) < 5:
        return np.nan
    
    t_d, p_d = t_d[mask], p_d[mask]

    def exp_decay(t_arr, A, tau):
        return A * np.exp(-t_arr / tau)

    try:
        # Provide reasonable initial guess: A~max(pressure), tau~1s
        p0 = [np.max(p_d), 1.0]
        bounds = ([0, 0.01], [np.inf, 10.0])  # tau between 10ms and 10s
        popt, _ = curve_fit(exp_decay, t_d, p_d, p0=p0, bounds=bounds, maxfev=5000)
        return popt[1]
    except (RuntimeError, ValueError, OptimizeWarning):
        return np.nan


def mmhg_to_pa(p_mmhg: np.ndarray) -> np.ndarray:
    """Convert pressure from mmHg to Pascals."""
    return p_mmhg * 133.322


def pa_to_mmhg(p_pa: np.ndarray) -> np.ndarray:
    """Convert pressure from Pascals to mmHg."""
    return p_pa / 133.322


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

# Define paths
data_dir = Path(__file__).parent  # examples/coa/
flow_file = data_dir / "data_typec_q.csv"
pressure_file = data_dir / "data_typec_p.csv"

# Load flow data (Csc)
try:
    flow_data = np.loadtxt(flow_file, delimiter=",", skiprows=1)
    t_flow = flow_data[:, 0]
    q_flow = flow_data[:, 1] * 1e-6  # ml/s → m³/s
except FileNotFoundError:
    print(f"❌ Error: Flow file not found: {flow_file}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading flow data: {e}")
    sys.exit(1)

# Load pressure data (Csvp)
try:
    pressure_data = np.loadtxt(pressure_file, delimiter=",", skiprows=1)
    t_p = pressure_data[:, 0]
    p_ref_mmhg = pressure_data[:, 1]  # mmHg
    # Convert reference pressure to SI units (Pa) for consistent comparison
    p_ref_pa = mmhg_to_pa(p_ref_mmhg)
except FileNotFoundError:
    print(f"❌ Error: Pressure file not found: {pressure_file}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading pressure data: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------
# Windkessel simulation
# ---------------------------------------------------------------------

# Model parameters (SI units: Pa, m³, s)
Rc = 1.2e7    # Characteristic resistance [Pa·s/m³]
Rp = 1.5e8    # Peripheral resistance [Pa·s/m³]
C = 1.8e-9    # Compliance [m³/Pa]
L = 5e4       # Inertance [Pa·s²/m³]

wk = Windkessel(
    t_flow=t_flow,
    q_flow=q_flow,
    Rc=Rc,
    Rp=Rp,
    C=C,
    L=L,
    periodic=True,
)

T = t_flow[-1]  # Cardiac cycle period

sol = wk.solve(
    t_start=0.0,
    t_end=5 * T,
    n_steps=5000,
)

# Remove transient (keep last 20% for steady-state analysis)
startN = int(0.8 * len(sol.t))
t_sim = sol.t[startN:] - sol.t[startN]  # Reset time origin
P_sim_pa = sol.p1[startN:]              # Pressure in Pa (SI)


# ---------------------------------------------------------------------
# Reference pressure processing
# ---------------------------------------------------------------------

# Interpolate reference pressure on simulation grid (already in Pa)
P_ref_interp_pa = np.interp(t_sim, t_p, p_ref_pa)

# Shape-based comparison: affine matching on SI units
P_ref_matched = affine_match(P_ref_interp_pa, P_sim_pa)


# ---------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------

nrms = normalized_rms_error(P_ref_matched, P_sim_pa)

# Peak timing analysis
peak_ref = np.argmax(P_ref_matched)
peak_sim = np.argmax(P_sim_pa)
dt_peak = t_sim[peak_ref] - t_sim[peak_sim]

# Diastolic time constant estimation
tau_sim = diastolic_tau(t_sim, P_sim_pa)
tau_ref = diastolic_tau(t_sim, P_ref_matched)

# Relative tau error (handle NaN cases)
if np.isnan(tau_sim) or np.isnan(tau_ref) or tau_ref == 0:
    tau_error_pct = np.nan
else:
    tau_error_pct = abs(tau_sim - tau_ref) / tau_ref * 100


# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(t_sim, pa_to_mmhg(P_sim_pa), label="Windkessel (simulated)", linewidth=2)
plt.plot(t_sim, pa_to_mmhg(P_ref_matched), "--", label="Reference (scaled)", color="grey")
plt.xlabel("Time [s]")
plt.ylabel("Pressure [mmHg]")
plt.title("Windkessel Model Validation - Pressure Waveform Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0.0, T)
plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "validation_waveform.png", dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------
# Report and validation thresholds
# ---------------------------------------------------------------------

print("\n" + "="*50)
print("=== Windkessel Validation Report ===")
print("="*50)
print(f"NRMS error           : {nrms * 100:6.2f} %")
print(f"Peak time shift      : {dt_peak * 1000:6.2f} ms")
print(f"Diastolic tau (sim)  : {tau_sim:6.4f} s" if not np.isnan(tau_sim) else "Diastolic tau (sim)  : NaN")
print(f"Diastolic tau (ref)  : {tau_ref:6.4f} s" if not np.isnan(tau_ref) else "Diastolic tau (ref)  : NaN")
if not np.isnan(tau_error_pct):
    print(f"Relative tau error   : {tau_error_pct:6.2f} %")
else:
    print("Relative tau error   : NaN")
print("="*50)

# Validation thresholds (adjust based on your requirements)
VALIDATION_THRESHOLDS = {
    "nrms_max": 0.15,           # 15% max normalized RMS error
    "dt_peak_max_ms": 50,       # 50ms max peak timing error
    "tau_error_max_pct": 20.0,  # 20% max relative tau error
}

# Check thresholds and report
validation_passed = True

if nrms > VALIDATION_THRESHOLDS["nrms_max"]:
    print(f"❌ NRMS error exceeds threshold ({nrms*100:.2f}% > {VALIDATION_THRESHOLDS['nrms_max']*100}%)")
    validation_passed = False
else:
    print(f"✅ NRMS error within threshold")

if abs(dt_peak) * 1000 > VALIDATION_THRESHOLDS["dt_peak_max_ms"]:
    print(f"❌ Peak timing error exceeds threshold ({abs(dt_peak)*1000:.2f} ms > {VALIDATION_THRESHOLDS['dt_peak_max_ms']} ms)")
    validation_passed = False
else:
    print(f"✅ Peak timing within threshold")

if not np.isnan(tau_error_pct) and tau_error_pct > VALIDATION_THRESHOLDS["tau_error_max_pct"]:
    print(f"❌ Tau error exceeds threshold ({tau_error_pct:.2f}% > {VALIDATION_THRESHOLDS['tau_error_max_pct']}%)")
    validation_passed = False
elif not np.isnan(tau_error_pct):
    print(f"✅ Diastolic tau within threshold")
else:
    print("⚠️  Could not compute tau error (fitting failed)")

print("="*50)
if validation_passed:
    print("🎉 VALIDATION PASSED ✅")
else:
    print("💥 VALIDATION FAILED ❌")
print("="*50 + "\n")

# Save metrics to JSON for automated tracking
metrics = {
    "nrms": float(nrms),
    "dt_peak_ms": float(dt_peak * 1000),
    "tau_sim_s": float(tau_sim) if not np.isnan(tau_sim) else None,
    "tau_ref_s": float(tau_ref) if not np.isnan(tau_ref) else None,
    "tau_error_pct": float(tau_error_pct) if not np.isnan(tau_error_pct) else None,
    "validation_passed": validation_passed,
    "parameters": {"Rc": Rc, "Rp": Rp, "C": C, "L": L}
}

with open(output_dir / "validation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"📊 Metrics saved to: {output_dir / 'validation_metrics.json'}")
print(f"📈 Plot saved to: {output_dir / 'validation_waveform.png'}")

# Optional: exit with error code if validation failed (useful for CI/CD)
if not validation_passed:
    sys.exit(1)