# CFD and Physics: Calculating Reynolds Number, Viscosity, and Boundary Layers with foampilot

*Before running any simulation, you need to understand the numbers that govern it. Here's how foampilot integrates fluid physics directly into the Python workflow.*

---

## Introduction: Physics First

In CFD, code is secondary. What matters is **physics**. A perfect mesh, a powerful solver, sophisticated numerical schemes — if the physical conditions are wrong, the results are wrong.

Before launching `simpleFoam` or `pimpleFoam`, engineers must answer three questions:

1. **Which fluid?** (water, air, oil, blood, carbon dioxide…)
2. **What properties?** (viscosity, density, thermal conductivity…)
3. **What regime?** (laminar, transient, turbulent) and at what **Reynolds number**?

Traditionally, these calculations are done in a spreadsheet or separate code, then values are manually copied into OpenFOAM dictionaries. foampilot changes this by integrating these calculations directly into the Python workflow.

In this article, I'll show you how to:
- Calculate properties of real fluids (air, water, oils) with **pyfluids**
- Determine flow regime via the **Reynolds number**
- Validate a simulation against the analytical **plane Poiseuille** solution
- Extract derived physical quantities: **y+**, **wall shear stress**, **strain rate**

---

## Part 1: Fluid Properties — From Air to Water

### 1.1 Why viscosity is the most critical parameter

Viscosity determines everything in CFD:

- The **Reynolds number**: `Re = ρvL/μ = vL/ν`
- The **boundary layer thickness**: `δ ≈ L / √Re`
- The flow type: laminar, transient, or turbulent
- The size of the first mesh cell (`y+`)

An error in viscosity invalidates all calculations. If you use `ν = 1e-5 m²/s` for water at 20°C instead of `ν ≈ 1e-6 m²/s`, your simulation will appear 10 times more viscous than it actually is.

### 1.2 Properties of common fluids

Reference values at 20°C (293.15 K):

| Fluid | Density ρ (kg/m³) | Dynamic viscosity μ (Pa·s) | Kinematic viscosity ν (m²/s) |
|-------|-------------------|----------------------------|------------------------------|
| Air | 1.204 | 1.81×10⁻⁵ | 1.52×10⁻⁵ |
| Water | 998.2 | 1.002×10⁻³ | 1.004×10⁻⁶ |
| Engine oil | 888 | 0.290 | 3.27×10⁻⁴ |
| Blood (35°C) | 1060 | 3.5×10⁻³ | 3.3×10⁻⁶ |
| CO₂ | 1.842 | 1.48×10⁻⁵ | 8.03×10⁻⁶ |

### 1.3 Automatic calculation with pyfluids

foampilot integrates **pyfluids** to calculate these properties automatically:

```python
from foampilot import FluidMechanics, ValueWithUnit
from pyfluids import FluidsList

# Define fluid and conditions
fluid = FluidMechanics(
    fluid_name=FluidsList.Air,
    temperature=ValueWithUnit(293.15, "K"),   # 20°C
    pressure=ValueWithUnit(101325, "Pa")      # Atmospheric pressure
)

# Retrieve all available properties
props = fluid.get_fluid_properties()

print(f"Kinematic viscosity: {props['kinematic_viscosity']}")
print(f"Dynamic viscosity: {props['dynamic_viscosity']}")
print(f"Density: {props['density']}")
print(f"Thermal conductivity: {props['thermal_conductivity']}")
```

**Output:**
```
Kinematic viscosity: 1.520e-05 m²/s
Dynamic viscosity: 1.813e-05 Pa·s
Density: 1.204 kg/m³
Thermal conductivity: 0.0257 W/(m·K)
```

---

## Part 2: The Reynolds Number — The Most Important Number in CFD

### 2.1 Physical definition

The Reynolds number is the ratio between inertial forces and viscous forces:

```
Re = (ρ × v × L) / μ = (v × L) / ν
```

Where:
- `ρ` = density (kg/m³)
- `v` = characteristic velocity (m/s)
- `L` = characteristic length (m)
- `μ` = dynamic viscosity (Pa·s)
- `ν` = kinematic viscosity (m²/s)

### 2.2 Regime interpretation

| Reynolds Number | Regime | Physical observation |
|-----------------|--------|----------------------|
| Re < 1 | **Stokes** | Highly viscous flow (honey, syrup) |
| 1 < Re < 2300 | **Laminar** | Smooth flow, no turbulence |
| 2300 < Re < 4000 | **Transient** | Intermittent turbulence appearance |
| Re > 4000 | **Turbulent** | Chaotic flow, intense mixing |

### 2.3 Real-world examples and reference values

| Configuration | Re | Regime |
|---------------|-----|--------|
| Microorganism in water | 1×10⁻⁶ | Stokes |
| Flow in microchannel (water) | 100 | Laminar |
| Flow in pipe (water, 1 m/s, D=0.1m) | 100 000 | Turbulent |
| **Car at 120 km/h** | **7×10⁶** | **Turbulent** |
| **Commercial airliner (cruise)** | **10⁸–10⁹** | **Turbulent** |
| Plane Poiseuille flow (test case) | 100–1000 | Laminar |

### 2.4 Calculation with foampilot

```python
from foampilot import FluidMechanics, ValueWithUnit
from pyfluids import FluidsList

# Case: flow around a car
fluid = FluidMechanics(
    fluid_name=FluidsList.Air,
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa"),
    velocity=ValueWithUnit(30, "m/s"),           # 108 km/h
    characteristic_length=ValueWithUnit(4.5, "m") # Car length
)

Re = fluid.calculate_reynolds()
print(f"Reynolds = {Re:.2e}")
# Output: Reynolds = 9.03e+06
```

**Interpretation**: `Re ≈ 9×10⁶` means **deeply turbulent flow**. A `kEpsilon` or `kOmegaSST` turbulence model is required. A laminar model would produce absurd results.

---

## Part 3: The Poiseuille Test Case — Physical Validation

Plane Poiseuille flow is the **fundamental validation test** in CFD. It's a laminar flow between two parallel plates, with an imposed pressure generating a parabolic velocity profile.

### 3.1 Analytical solution

For incompressible plane Poiseuille flow, the analytical velocity solution is:

```
u(y) = (1 / 2ν) × (dp/dx) × (h² - y²)
```

Where:
- `dp/dx` = imposed pressure gradient (Pa/m)
- `h` = half-channel height (m)
- `ν` = kinematic viscosity (m²/s)
- `y` = distance from centerline (m)

### 3.2 Analytical wall shear stress

The wall shear stress `τ_w` for plane Poiseuille flow is:

```
τ_w = (h/2) × |dp/dx|
```

Or equivalently:
```
τ_w = ρ × ν × (du/dy)|_wall
```

### 3.3 Implementation in foampilot

```python
# Poiseuille test case parameters (see test_cfd_methods.py)
nu = 0.1          # Kinematic viscosity (m²/s)
rho = 1.0         # Density (kg/m³)
G = 5.0           # Pressure gradient (Pa/m)
H = 1.0           # Half-channel height (m)

# Centerline velocity — analytical solution: u_max = G·h² / 2ν
U_max = G * H**2 / (2 * nu)
print(f"U_max = {U_max:.1f} m/s")  # Output: U_max = 25.0 m/s

# Mean velocity (U_avg = U_max / 2 for Poiseuille)
U_avg = U_max / 2

# Reynolds number
Re = U_avg * (2 * H) / nu
print(f"Re = {Re:.1f}")  # Output: Re = 250.0

# Regime check
if Re < 2300:
    print("Laminar regime — analytical solution valid")
else:
    print("Turbulent regime — need turbulence model")
```

### 3.4 Quantitative validation

In `test_cfd_methods.py`, foampilot validates its post-processing against the analytical solution. The `get_structure` method reads VTK files generated by `foamToVTK` — a `foamToVTK -case ./planarPoiseuille` run is required before reading results:

```python
import numpy as np
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./planarPoiseuille")
mesh = fp.get_structure(time_step=25)["cell"]

# Analytical wall shear stress
G = 5.0      # Pa/m (pressure gradient)
H = 1.0      # m (half-height)
rho = 1.0    # kg/m³
tau_w_analytical = 0.5 * rho * G * H
print(f"Analytical τ_w = {tau_w_analytical:.4f} Pa")

# Wall shear stress calculated by foampilot
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=0.1,  # viscosity used in test case
    wall_normal=[0, 1, 0]
)

wss_max = np.max(mesh_wss.point_data["wall_shear_stress"])
error = abs(wss_max - tau_w_analytical) / tau_w_analytical
print(f"Relative error: {error:.2%}")
# Expected output: Relative error < 5%
```

**Expected result**: error below 5% for a sufficiently refined mesh (≥ 50 cells in the normal direction).

---

## Part 4: Boundary Layers and y+

### 4.1 What is the boundary layer?

When a fluid flows along a wall, velocity goes from 0 to the free-stream velocity over a short distance: the **viscous boundary layer**. This is where vortices form and turbulence is born.

### 4.2 The y+ parameter

In turbulence, the first mesh cell must be placed within the boundary layer. The key parameter is **y+**:

```
y+ = (u_τ × y) / ν
```

Where `u_τ = √(τ_w / ρ)` is the friction velocity.

### 4.3 Meshing rules by turbulence model

| Turbulence model | Target y+ | First cell position |
|------------------|-----------|---------------------|
| Standard k-epsilon | 30–300 | Integrated wall law |
| k-omega SST | y+ < 1 | Resolved in boundary layer |
| Spalart-Allmaras | 1–20 | Partially resolved |

### 4.4 Automatic calculation with foampilot

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./planarPoiseuille")
mesh = fp.get_structure(time_step=25)["cell"]

# Calculate y+ on "walls" patch
mesh_yp = fp.calc_y_plus(
    mesh, 
    wall_patch_name="walls", 
    velocity_field="U", 
    viscosity=0.1  # test case viscosity
)

print(f"y+ min = {mesh_yp.point_data['y_plus'].min():.2f}")
print(f"y+ max = {mesh_yp.point_data['y_plus'].max():.2f}")
print(f"y+ mean = {mesh_yp.point_data['y_plus'].mean():.2f}")
```

---

## Part 5: Derived Physical Quantities

### 5.1 Wall shear stress

Wall shear stress is the tangential force exerted by the fluid on the wall:

```
τ_w = μ × (du/dy)|_wall
```

```python
# Calculate wall shear stress
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=0.1, 
    wall_normal=[0, 1, 0]
)

tau_w = mesh_wss.point_data["wall_shear_stress"]
print(f"τ_w max = {tau_w.max():.4f} Pa")
print(f"τ_w mean = {tau_w.mean():.4f} Pa")

# Analytical validation for plane Poiseuille
tau_w_analytical = 0.5 * rho * G * H
print(f"Analytical τ_w = {tau_w_analytical:.4f} Pa")
```

### 5.2 Strain rate

Strain rate measures how the fluid deforms locally:

```
γ̇ = √(2 × S : S)
```

Where `S` is the strain rate tensor: `S = 0.5 × (∇u + ∇uᵀ)`

```python
# Calculate strain rate
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
strain_rate = mesh_sr.point_data["strain_rate"]
print(f"Max strain rate = {strain_rate.max():.4f} 1/s")
print(f"Mean strain rate = {strain_rate.mean():.4f} 1/s")
```

### 5.3 Analytical vs numerical velocity profile

```python
import numpy as np

# Test case parameters (identical to Part 3)
nu = 0.1      # Kinematic viscosity (m²/s)
G = 5.0       # Pressure gradient (Pa/m)
H = 1.0       # Half-channel height (m)

# Extract velocity profile on centerline
centerline = mesh.slice(normal="y", origin=(0, 0, 0))
y_points = centerline.points[:, 2]  # Z coordinate

# Analytical Poiseuille solution
U_analytical = (1 / (2 * nu)) * (-G) * (H**2 - y_points**2)

# Comparison
U_numerical = centerline.point_data["U"][:, 0]
error = np.mean(np.abs(U_numerical - U_analytical))
print(f"Mean profile error: {error:.4f} m/s")
```

---

## Part 6: From Fluid to Simulation — Complete Example

Here's how to integrate these physical calculations into a complete foampilot workflow:

```python
from foampilot import Solver, Meshing, FluidMechanics, ValueWithUnit
from pyfluids import FluidsList

# 1. Choose fluid and calculate properties
fluid = FluidMechanics(
    fluid_name=FluidsList.Water,
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa"),
    velocity=ValueWithUnit(2.0, "m/s"),           # Inlet velocity
    characteristic_length=ValueWithUnit(0.05, "m")  # Pipe diameter
)
props = fluid.get_fluid_properties()

# 2. Reynolds number
Re = fluid.calculate_reynolds()
print(f"Re = {Re:.0f}")  # Example: Re = 99,800 (turbulent)

# 3. Solver configuration
solver = Solver(case_path="./pipe_flow")
solver.transient = False
solver.turbulence_model = "kOmegaSST"  # Chosen based on Re

# 4. Inject physical properties
solver.constant.transportProperties.nu = props['kinematic_viscosity']

# 5. Mesh and boundary conditions
meshing = Meshing(case_path="./pipe_flow", mesher="blockMesh")
meshing.write()

solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    "inlet", "velocityInlet",
    velocity=(ValueWithUnit(2.0, "m/s"),
              ValueWithUnit(0, "m/s"),
              ValueWithUnit(0, "m/s"))
)
solver.boundary.write_boundary_conditions()

# 6. Write and run
solver.write_case()
solver.run_simulation(nb_proc=2)
```

---

## Conclusion: Physics as Foundation

In CFD, code is a tool. Physics is the foundation. foampilot doesn't just generate files — it integrates **physical calculations** (fluid properties, Reynolds number, y+, wall shear stress) directly into the Python workflow.

In the next article, I'll show you how to automate the complete creation of an OpenFOAM case with these physical concepts integrated.

**Resources:**
- Poiseuille test case: [github.com/stevendaix/foampilot/tree/main/planarPoiseuille](https://github.com/stevendaix/foampilot/tree/main/planarPoiseuille)
- FluidMechanics documentation: [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)
- pyfluids: [pypi.org/project/pyfluids](https://pypi.org/project/pyfluids)

---

*Article under improvement — version 1.1*
