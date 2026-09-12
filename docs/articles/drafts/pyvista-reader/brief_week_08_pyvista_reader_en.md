# Direct OpenFOAM Reader for PyVista: Architecture and Physical Quantity Extraction

*How foampilot reads OpenFOAM results directly into PyVista, without conversion, and extracts physical quantities (y+, wall shear stress, strain rate).*

---

## Proposed Structure

### Introduction — The foamToVTK bottleneck

Storytelling: You have a running simulation. You want to visualize results. You run `foamToVTK`, wait 10 minutes, open Paraview. But what if you could visualize directly from Python, without conversion?

**Promise**: With `OpenFOAMDirectReader` and PyVista, it's possible. And you can calculate y+, wall shear stress, strain rate — directly in the script.

### Section 1 — Classic workflow (and its limits)

1. Run OpenFOAM simulation
2. Run `foamToVTK` to convert to VTK
3. Open Paraview
4. Load VTK file
5. Configure visualization

**Problems**:
- Conversion time (sometimes long for large cases)
- Disk space doubled (OpenFOAM + VTK)
- Non-automatable workflow
- No scriptability for physical quantities

### Section 2 — The solution: direct reading

Present the direct reader:

```python
import pyvista as pv
from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader

reader = OpenFOAMDirectReader(case_path="./poiseuille")
mesh = reader.read(time_step=100)

# Direct visualization
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="U", cmap="viridis")
plotter.show()
```

**Advantages**:
- No conversion
- Direct field access
- Scriptable
- Integrable in reports

### Section 3 — Under the hood: reader architecture

#### 3.1 Reading OpenFOAM field files

```python
# Parsing volVectorField / volScalarField
# Header extraction
# Boundary field parsing
# Internal field reconstruction
```

#### 3.2 Mesh reconstruction from polyMesh

```python
# Reading points, faces, owner, neighbour
# Building cell connectivity
# Patch extraction
```

#### 3.3 Time step management

```python
# Listing time directories
# Selecting specific time step
# Lazy loading for memory efficiency
```

### Section 4 — Extracting physical quantities

#### 4.1 y+ calculation

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./poiseuille")
mesh = fp.read_direct(time_step=100)

mesh_yp = fp.calc_y_plus(
    mesh, 
    wall_patch_name="walls", 
    velocity_field="U", 
    viscosity=1.52e-5
)
```

#### 4.2 Wall shear stress

```python
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=1.52e-5, 
    wall_normal=[0, 1, 0]
)
```

#### 4.3 Strain rate

```python
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
```

### Section 5 — CHT multi-region reading

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)
```

### Section 6 — Performance

- Read time for 1M cells: ~1.5s
- Memory: ~500 MB for 1M cell case
- Comparison with foamToVTK + VTK reader
- Lazy loading strategy

### Conclusion — From mesh to results in one pipeline

CTA: *"Now that you can read and visualize your results, I'll show you how to generate professional reports with LaTeX and Typst."*

---

## Code to prepare

- [ ] Simple direct reader script
- [ ] y+ extraction with validation
- [ ] Wall shear stress extraction with analytical validation
- [ ] CHTDirectReader script
- [ ] Performance benchmark

## Images to prepare

- [ ] Reader architecture diagram
- [ ] Performance comparison chart
- [ ] PyVista visualization screenshot
- [ ] y+ heatmap on walls
- [ ] Memory usage graph
