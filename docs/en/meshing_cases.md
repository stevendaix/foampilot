# Meshing cases and mesh strategy

Meshing is not a pre-processing detail that can be separated from the physics. It determines which gradients can be resolved, which wall treatment is valid, how accurately forces and heat fluxes are integrated, and how much numerical diffusion enters the solution. FoamPilot orchestrates several meshing strategies, but the user remains responsible for choosing the geometry representation, refinement targets, patch topology, and quality criteria.

## Mesh strategy selection

| Geometry or objective | Recommended route | Why |
| --- | --- | --- |
| Rectangular cavity, channel, duct, or 2-D benchmark | `blockMesh` | Explicit topology, predictable cells, excellent for verification. |
| Structured multi-block geometry | `classy_blocks` / `blockMesh` | Strong control of grading, blocks, arcs, and named patches. |
| CAD solid or STEP geometry | Gmsh | Flexible unstructured surface/volume meshing and CAD import. |
| STL, OBJ, building, vehicle, or biological surface | Background `blockMesh` + `snappyHexMesh` | Local refinement and snapping around complex triangulated surfaces. |
| Existing OpenFOAM mesh | Direct mesh reader/exporter | Avoids remeshing and enables a Python-controlled post-processing path. |
| Large urban data set | Urban readers + simplification + Gmsh/surface builder | Controls geometry complexity, metric coordinates, and cell budget. |
| CHT fluid/solid case | `blockMesh` or Gmsh + cell zones + region splitting | The mesh must represent both regions and their coupled interface. |

## 1. Structured `blockMesh` cases

`blockMesh` is the preferred route for verification cases because the topology is explicit. The user controls vertices, blocks, edges, grading, and boundary patches. This makes it appropriate for the cavity, scalar channel, buoyancy room, backward-facing step, and heated-duct background mesh.

A structured case should define:

1. the coordinate system and dimensions;
2. the block connectivity and cell count;
3. the grading ratio in each direction;
4. the patch names and OpenFOAM patch types;
5. the dimensional scale;
6. the intended wall resolution and symmetry assumptions.

The main risks are incorrect vertex ordering, inconsistent face normals, excessive grading, and patches that do not match the boundary-condition code. Run `blockMesh` and `checkMesh` before writing the rest of the case.

## 2. `classy_blocks` and multi-block geometry

`classy_blocks` is useful when a geometry is naturally assembled from cylinders, extrusions, rings, elbows, or chained blocks. The FoamPilot user guide demonstrates shape construction, chaining, expansion, filling, directional chopping, and patch assignment.

The advantage is geometric control. The disadvantage is that the user must understand how the blocks meet and how cell grading propagates across block interfaces. Use it for a geometry whose topology is known; do not use it to hide a poorly understood CAD surface.

## 3. Gmsh cases

Gmsh is appropriate for STEP/IGES/CAD-like geometry and for domains where unstructured tetrahedral or hybrid meshing is preferable. A Gmsh case must document:

| Input | Required decision |
| --- | --- |
| CAD units | Confirm whether the source is in metres, millimetres, or another unit system. |
| Physical groups | Define inlet, outlet, walls, symmetry, interfaces, and solid regions explicitly. |
| Element order | Choose linear or higher-order elements consistently with the solver pipeline. |
| Surface quality | Remove duplicate, self-intersecting, or badly oriented faces. |
| Volume closure | Confirm that each fluid or solid volume is watertight. |
| Conversion | Check how the generated mesh is converted to OpenFOAM and how patch names survive. |

Gmsh refinement should be driven by the physics: narrow gaps, high-curvature surfaces, separation edges, thermal interfaces, and boundary layers need more cells than uniform regions.

## 4. `snappyHexMesh` cases

The standard complex-geometry sequence is:

```text
background blockMesh
→ surfaceFeatureExtract
→ castellatedMesh
→ snap
→ addLayers (optional)
→ checkMesh
```

The background mesh defines the outer domain. Surface geometry is placed under `constant/triSurface` or the configured geometry directory. `snappyHexMesh` removes or refines cells according to geometry intersections, snaps points to the surface, and can add prism layers.

### Refinement regions

Use local refinement around:

- leading and trailing edges;
- building corners and roof lines;
- vehicle wheels, fairings, and underbody gaps;
- wake regions behind bluff bodies;
- thermal interfaces and narrow fluid passages;
- medical stenoses, aneurysm necks, branches, and inlet/outlet extensions.

The refinement level must be balanced against the turbulence model and wall treatment. A fine surface mesh with an under-resolved boundary layer is not automatically a good CFD mesh.

### Surface and feature checks

Before running a complex case, inspect the surface in a viewer and check:

| Check | Typical consequence if it fails |
| --- | --- |
| Closed and orientable surface | Leaks, missing cells, incorrect inside/outside classification. |
| Consistent scale | Geometry is too large or too small relative to velocity and viscosity. |
| Feature extraction | Sharp edges are rounded or patches are merged unexpectedly. |
| Patch names | Boundary conditions are applied to the wrong surface. |
| Surface normals | Wall orientation or flux signs are incorrect. |
| Layer feasibility | Prism layers collapse or create non-orthogonal cells. |

## 5. Urban and atmospheric meshes

Urban CFD requires a geospatial stage before OpenFOAM meshing. Convert the data to a metric coordinate system, define the wind frame, remove irrelevant objects, simplify building footprints, assign heights, and establish terrain and domain margins. The urban package contains models for buildings, roads, terrain, CFD domains, geometry simplification, cleanup, mesh sizing, wake refinement, boundary layers, patch assignment, and validation.

The mesh domain should be justified by the incoming atmospheric boundary layer and the downstream wake. A domain that is too short recycles pressure and turbulence disturbances into the region of interest. A domain that is too small laterally constrains the wind and exaggerates blockage.

## 6. Biomedical surface and volume meshes

Biomedical meshes require extra care because the geometry is patient-specific and the quantities of interest often depend on derivatives: wall shear stress, pressure drop, residence time, or heat transfer. The workflow typically includes image segmentation or surface import, cleaning, hole closure, smoothing with a controlled tolerance, inlet/outlet extension, surface remeshing, volume meshing, and boundary-layer refinement when appropriate.

A geometry-processing operation should never be described only as “cleaning”. Record the algorithm, tolerance, target edge length, number of triangles, smoothing iterations, and whether the operation changes lumen volume or branch diameters. Validate the final mesh against the original imaging-derived surface.

For blood flow, refine regions with high curvature, stenosis, bifurcation, recirculation, and expected high wall-shear gradients. Extend outlets sufficiently to reduce the influence of artificial boundary conditions on the region of interest.

## 7. CHT meshes and region interfaces

A CHT mesh must distinguish fluid cells from solid cells and must preserve a conformal or otherwise correctly coupled interface. The tutorial uses a background mesh and cell-zone definitions before splitting the case into `fluid` and `solid` regions.

The interface requires:

- matching or correctly mapped faces;
- region-specific temperature fields;
- thermophysical properties in each region;
- coupled temperature and heat-flux boundary conditions;
- a consistent normal direction and interface naming convention;
- sufficient resolution across the thermal boundary layer and solid conduction path.

The smallest cell size should be justified by both momentum and thermal gradients. A mesh can resolve velocity while under-resolving temperature, or the reverse. Use thermal boundary-layer estimates and the local Prandtl number to guide the first mesh, then perform a refinement study.

## 8. Mesh-quality indicators

`checkMesh` is necessary but not sufficient. Report at least the following indicators:

| Indicator | Interpretation |
| --- | --- |
| Non-orthogonality | Large values increase discretisation error and may require correction or a different mesh. |
| Skewness | High skewness degrades gradient and flux reconstruction. |
| Aspect ratio | High ratios can be valid in boundary layers but harmful in poorly aligned regions. |
| Volume ratio | Abrupt cell-size changes can produce numerical stiffness. |
| Negative or zero volume | Invalid mesh; stop before solving. |
| Boundary-layer count | Determines whether the wall model or low-Re treatment is appropriate. |
| $y^+$ distribution | Must be compatible with the selected wall treatment. |
| Cell count by region | Important for CHT balances and parallel decomposition. |

## 9. Wall resolution and $y^+$

The target $y^+$ depends on the wall treatment. Low-Re approaches aim to resolve the viscous sublayer, typically with $y^+$ near unity. Wall-function approaches place the first cell in a logarithmic region and require a target range consistent with the particular wall function and turbulence model. The exact target is not universal.

Use:

$$
 y^+ = \frac{u_\tau y}{\nu},
$$

where $u_\tau=\sqrt{\tau_w/\rho}$ is the friction velocity and $y$ is the distance from the wall to the cell centre. Since $u_\tau$ is initially unknown, estimate it from a flat-plate or pipe-flow correlation, create a preliminary mesh, run the case, and then inspect the actual $y^+$ field.

## 10. Mesh convergence protocol

A defensible mesh study changes one resolution parameter at a time and compares the engineering outputs that matter: pressure drop, drag coefficient, reattachment length, heat-transfer coefficient, Nusselt number, WSS, or scalar mixing index. Compare both global quantities and local profiles. A small residual does not prove mesh independence.

For transient cases, perform a time-step study separately. For multiphase cases, monitor interface resolution and volume conservation. For CHT, include total heat balance and interface temperature continuity in the convergence criteria.
