#!/usr/bin/env python3
"""Post-traitement CHT pour le cas Heated Duct (CHT tutorial).

Charge les fichiers VTK générés par foamToVTK et calcule :
- Profil de température dans le solide et le fluide
- Flux de chaleur à l'interface
- Nombre de Nusselt
- Coefficient de transfert de chaleur
- Balance énergétique globale
- Visualisations (images PNG)
"""

import sys
from pathlib import Path
import numpy as np
import pyvista as pv
import pandas as pd

# Ensure foampilot is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from foampilot.cht import (
    calc_nusselt_number,
    calc_heat_transfer_coefficient,
    calc_total_heat_balance,
    calc_temperature_contour,
    calc_thermal_resistance,
)

case_path = Path(__file__).resolve().parent
vtk_dir = case_path / "VTK"
output_dir = case_path / "postProcessing"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load VTK data for both regions
# ---------------------------------------------------------------------------
print("=== Loading VTK data ===")

def latest_internal_vtk(region: str) -> Path:
    candidates = [p for p in (vtk_dir / region).glob("*.vtk") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No internal VTK found for region {region}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

# Load the generated multi-region internal meshes.
fluid_mesh = pv.read(str(latest_internal_vtk("fluid")))
metal_mesh = pv.read(str(latest_internal_vtk("metal")))
heater_mesh = pv.read(str(latest_internal_vtk("heater")))
solid_mesh = pv.merge([metal_mesh, heater_mesh])

# Load the reference fluid-metal interface patches.
fluid_interface = pv.read(str(max((vtk_dir / "fluid" / "fluid_to_metal").glob("*.vtk"), key=lambda p: p.stat().st_mtime)))
solid_interface = pv.read(str(max((vtk_dir / "metal" / "metal_to_fluid").glob("*.vtk"), key=lambda p: p.stat().st_mtime)))

print(f"Fluid mesh: {fluid_mesh.n_cells} cells, {fluid_mesh.n_points} points")
print(f"Solid mesh: {solid_mesh.n_cells} cells, {solid_mesh.n_points} points")

# ---------------------------------------------------------------------------
# 2. Temperature statistics
# ---------------------------------------------------------------------------
print("\n=== Temperature statistics ===")

T_fluid = fluid_mesh.point_data["T"]
T_solid = solid_mesh.point_data["T"]

stats = {
    "fluid": {"min": float(np.min(T_fluid)), "max": float(np.max(T_fluid)), "mean": float(np.mean(T_fluid))},
    "solid": {"min": float(np.min(T_solid)), "max": float(np.max(T_solid)), "mean": float(np.mean(T_solid))},
}

print(f"Fluid T: min={stats['fluid']['min']:.2f} K, max={stats['fluid']['max']:.2f} K, mean={stats['fluid']['mean']:.2f} K")
print(f"Solid T: min={stats['solid']['min']:.2f} K, max={stats['solid']['max']:.2f} K, mean={stats['solid']['mean']:.2f} K")

# ---------------------------------------------------------------------------
# 3. Interface heat transfer
# ---------------------------------------------------------------------------
print("\n=== Interface heat transfer ===")

T_interface_fluid = float(np.mean(fluid_interface.point_data["T"]))
T_interface_solid = float(np.mean(solid_interface.point_data["T"]))
T_interface = (T_interface_fluid + T_interface_solid) / 2.0

print(f"Interface T (fluid side): {T_interface_fluid:.2f} K")
print(f"Interface T (solid side): {T_interface_solid:.2f} K")
print(f"Interface T (average):    {T_interface:.2f} K")

# Calculate heat transfer coefficient
# h = q / (T_wall - T_fluid_bulk)
T_bulk = float(np.mean(T_fluid))
T_wall = T_interface_fluid
delta_T = T_wall - T_bulk
if abs(delta_T) > 1e-10:
    # Estimate wall heat flux from temperature gradient
    grad_T = np.gradient(T_solid, 0.002 / len(T_solid))
    q_wall = 380.0 * float(np.mean(grad_T))  # copper conductivity
    h = calc_heat_transfer_coefficient(q_wall, T_wall, T_bulk)
else:
    q_wall = 0.0
    h = 0.0

print(f"Bulk fluid T: {T_bulk:.2f} K")
print(f"Heat flux at interface: {q_wall:.2f} W/m²")
print(f"Heat transfer coefficient: {h:.2f} W/(m²·K)")

# ---------------------------------------------------------------------------
# 4. Nusselt number
# ---------------------------------------------------------------------------
print("\n=== Nusselt number ===")

L = 0.002  # solid thickness (m)
k_fluid = 0.026  # air thermal conductivity (W/(m·K))
q_wall_abs = abs(q_wall)

nu = calc_nusselt_number(
    q_wall=q_wall_abs,
    L=L,
    k_fluid=k_fluid,
    T_wall=T_wall,
    T_bulk=T_bulk,
)
print(f"Nusselt number (based on solid thickness): {nu:.4f}")

# ---------------------------------------------------------------------------
# 5. Thermal resistance
# ---------------------------------------------------------------------------
print("\n=== Thermal resistance ===")

R_thermal = calc_thermal_resistance(T_hot=350.0, T_cold=T_bulk, Q_total=q_wall)
print(f"Thermal resistance: {R_thermal:.4f} K/W")

# ---------------------------------------------------------------------------
# 6. Temperature contours
# ---------------------------------------------------------------------------
print("\n=== Temperature contours ===")

contour_info = calc_temperature_contour(T_fluid, levels=10)
print(f"Fluid temperature contours: {len(contour_info['levels'])} levels")
print(f"  T_range: {contour_info['T_min']:.2f} - {contour_info['T_max']:.2f} K")

contour_info_solid = calc_temperature_contour(T_solid, levels=10)
print(f"Solid temperature contours: {len(contour_info_solid['levels'])} levels")
print(f"  T_range: {contour_info_solid['T_min']:.2f} - {contour_info_solid['T_max']:.2f} K")

# ---------------------------------------------------------------------------
# 7. Energy balance
# ---------------------------------------------------------------------------
print("\n=== Energy balance ===")

Q_in = 350.0 * 7800.0 * 460.0 * 0.002 * 0.1 * 0.0005  # rough estimate
balance = calc_total_heat_balance(
    Q_in=Q_in,
    Q_out=q_wall * 0.1 * 0.0005,  # q_wall * area
    Q_stored=0.0,
    tolerance=0.1,
)
print(f"Heat balance: {balance['balance']:.2f} W")
print(f"Balance error: {balance['balance_error']:.4f}")
print(f"Conserved: {balance['is_conserved']}")

# ---------------------------------------------------------------------------
# 8. Export statistics to CSV
# ---------------------------------------------------------------------------
print("\n=== Exporting results ===")

# Export temperature profile across the domain
y_coords = mesh_points = fluid_mesh.points[:, 1]
T_profile = T_fluid

df_profile = pd.DataFrame({
    "y_coordinate": y_coords,
    "fluid_temperature": T_profile,
})
df_profile.to_csv(output_dir / "temperature_profile.csv", index=False)

# Combine fluid and solid temperature profiles
all_y = np.concatenate([solid_mesh.points[:, 1], fluid_mesh.points[:, 1]])
all_T = np.concatenate([T_solid, T_fluid])
df_combined = pd.DataFrame({
    "y_coordinate_m": all_y,
    "temperature_K": all_T,
})
df_combined.to_csv(output_dir / "temperature_profile_combined.csv", index=False)

# Export summary statistics
df_stats = pd.DataFrame([
    {"region": "fluid", "T_min": stats["fluid"]["min"], "T_max": stats["fluid"]["max"], "T_mean": stats["fluid"]["mean"]},
    {"region": "solid", "T_min": stats["solid"]["min"], "T_max": stats["solid"]["max"], "T_mean": stats["solid"]["mean"]},
])
df_stats.to_csv(output_dir / "temperature_statistics.csv", index=False)

print(f"Statistics exported to {output_dir}")

# ---------------------------------------------------------------------------
# 9. Generate plots
# ---------------------------------------------------------------------------
print("\n=== Generating plots ===")

# Plot 1: Temperature contour (fluid)
pl = pv.Plotter(off_screen=True)
pl.add_mesh(
    fluid_mesh.slice("z"),
    scalars="T",
    lighting=False,
    scalar_bar_args={"title": "Temperature (K)"},
    cmap="coolwarm",
)
pl.screenshot(str(output_dir / "fluid_temperature_contour.png"), window_size=(1200, 400))
pl.clear()

# Plot 2: Temperature contour (solid)
pl = pv.Plotter(off_screen=True)
pl.add_mesh(
    solid_mesh.slice("z"),
    scalars="T",
    lighting=False,
    scalar_bar_args={"title": "Temperature (K)"},
    cmap="coolwarm",
)
pl.screenshot(str(output_dir / "solid_temperature_contour.png"), window_size=(1200, 300))
pl.clear()

# Plot 3: Combined temperature (both regions overlaid)
pl = pv.Plotter(off_screen=True)
pl.add_mesh(
    fluid_mesh.slice("z"),
    scalars="T",
    lighting=False,
    scalar_bar_args={"title": "Temperature (K)"},
    cmap="coolwarm",
    opacity=0.7,
)
pl.add_mesh(
    solid_mesh.slice("z"),
    scalars="T",
    lighting=False,
    cmap="coolwarm",
    opacity=0.9,
)
pl.screenshot(str(output_dir / "cht_temperature_contour.png"), window_size=(1200, 400))
pl.clear()

print(f"Plots saved to {output_dir}")

# ---------------------------------------------------------------------------
# 10. Summary report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CHT POST-PROCESSING SUMMARY")
print("=" * 60)
print(f"Case: {case_path.name}")
print(f"Solver: chtMultiRegionFoam (OpenFOAM 13)")
print(f"Time: 20 s (end time)")
print(f"")
print(f"Fluid region: {fluid_mesh.n_cells} cells")
print(f"  Temperature: {stats['fluid']['min']:.2f} - {stats['fluid']['max']:.2f} K (mean: {stats['fluid']['mean']:.2f} K)")
print(f"")
print(f"Solid region: {solid_mesh.n_cells} cells")
print(f"  Temperature: {stats['solid']['min']:.2f} - {stats['solid']['max']:.2f} K (mean: {stats['solid']['mean']:.2f} K)")
print(f"")
print(f"Interface temperature: {T_interface:.2f} K")
print(f"Heat transfer coefficient: {h:.2f} W/(m²·K)")
print(f"Nusselt number: {nu:.4f}")
print(f"Thermal resistance: {R_thermal:.4f} K/W")
print("=" * 60)
print(f"\nAll results saved to: {output_dir}/")
