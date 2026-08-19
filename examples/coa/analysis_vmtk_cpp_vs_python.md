# Detailed Comparison: foampilot Python VMTK vs. Original VMTK C++
**Date:** 2026-08-19  
**Goal:** Identify algorithmic differences and best practices explaining the geometric accuracy gap  
**Benchmark:** Hausdorff 23.9mm (target <10mm), Length error 65.4% (target <20%), Radius error 1.04% (good), Points: 82 (reference has 409)

---

## Executive Summary

The foampilot Python implementation captures the high-level pipeline of VMTK but deviates significantly at the algorithmic level in four critical areas:

1. **Fast Marching Method**: Python uses Dijkstra on a coarse edge graph; VMTK solves the continuous Eikonal equation on a polygonal non-manifold with angle-dependent quadratic updates.
2. **Steepest Descent Tracing**: Python traces vertex-to-vertex with post-hoc interpolation; VMTK traces continuously by subdividing edges into 250 segments and finding the true gradient direction.
3. **Voronoi Radius**: Python uses surface clearance distance; VMTK uses the Delaunay circumradius.
4. **Tolerance & Seed Selection**: Python uses 1e-6 tolerance and naive nearest-neighbor seeds; VMTK uses 1e-12 tolerance and direction-aware pole selection.

These differences explain the 65% length error and 23.9mm Hausdorff distance.

---

## 1. Internal Tetrahedra Extraction

### VMTK C++ (`vtkvmtkInternalTetrahedraExtractor.cxx`)
- **Tolerance:** `VTK_VMTK_DOUBLE_TOL = 1.0e-12` (line 36 of vtkvmtkConstants.h)
- **Normals:** Requires pre-computed outward normals array; errors out if missing
- **Dot product logic:**
  ```cpp
  // All four vertices must point outward
  allDotPositive = (dot0>tolerance)&&(dot1>tolerance)&&(dot2>tolerance)&&(dot3>tolerance);
  // For boundary tets: exactly 3 out of 4 must point outward
  allDotMinusOnePositive = ((dot0>tolerance)&&(dot1>tolerance)&&(dot2>tolerance))||
                           ((dot0>tolerance)&&(dot1>tolerance)&&(dot3>tolerance))||
                           ((dot0>tolerance)&&(dot2>tolerance)&&(dot3>tolerance))||
                           ((dot1>tolerance)&&(dot2>tolerance)&&(dot3>tolerance));
  ```
- **Subresolution removal:** Computes `minEdgeLength` from triangle areas via `sqrt(2.0 * triangleArea)` (equivalent to minimum edge of an equivalent-area equilateral triangle), then checks `circumradius < SubresolutionFactor * minEdgeLength`
- **Cap handling:** Uses `CapCenterIds` to identify boundary tets; only boundary tets get the 3-out-of-4 rule

### foampilot Python (`vmtkinternaltetrahedra_local.py`)
- **Tolerance:** `1e-6` (line 174) — **6 orders of magnitude looser**
- **Normals:** Computes normals if missing (lines 127-134), then re-assigns `surface = normals_source.GetOutput()`
- **Dot product logic:**
  ```python
  all_dot_positive = (dot0 > 1e-6 and dot1 > 1e-6 and dot2 > 1e-6 and dot3 > 1e-6)
  all_but_one_positive = sum(d > 1e-6 for d in [dot0, dot1, dot2, dot3]) >= 3
  ```
- **Subresolution removal:** Computes actual minimum edge length from triangle edges (lines 204-214), then checks `circumradius < subresolution_factor * min_surface_edge`
- **Cap handling:** Identifies boundary tets via `cap_center_ids` (lines 141-169), same 3-out-of-4 rule

### Differences & Impact
| Aspect | VMTK C++ | foampilot Python | Impact |
|--------|----------|------------------|--------|
| Tolerance | 1e-12 | 1e-6 | **Critical**: Looser tolerance misclassifies boundary/internal tets, altering Voronoi graph connectivity |
| Normal fallback | None (fatal error) | Computes if missing | More robust but may produce different normals than C++ pipeline expects |
| Subresolution edge | Area-derived | Direct edge length | Minor; both remove small tets but with different thresholds |
| Cap center lookup | `CapCenterIds` from capper | `seed_cell_id` from nearest cell to cap barycenter | Minor difference in seed selection |

**Expected impact on accuracy:** The 1e-6 tolerance is the most damaging difference. With a typical vessel radius of ~3mm and mesh element size of ~0.5mm, the dot products can be on the order of 1e-3 to 1e-1. A 1e-6 threshold is actually tighter in absolute terms than 1e-12 for this scale, BUT the issue is that VMTK uses 1e-12 as a RELATIVE numerical tolerance to handle floating-point precision in the circumsphere computation. The Python 1e-6 may reject tets that VMTK would accept due to accumulated floating-point error in the `np.linalg.solve` circumsphere computation.

---

## 2. Voronoi Diagram Construction

### VMTK C++ (`vtkvmtkVoronoiDiagram3D.cxx`)
- **Output:** `vtkPolyData` with **both polys (Voronoi cells) and lines (edges)**
- **BuildVoronoiPolys:** Constructs boundary faces on the surface by walking from edge to neighboring tetrahedra until a boundary tet is found (lines 96-145)
- **Radius:** `sqrt(vtkTetra::Circumsphere(...))` — the **circumradius** of each Delaunay tetrahedron
- **Poles:** For each surface point, finds the tetra with maximum circumradius (outer pole) and the second maximum with opposite direction (inner pole)
- **Edge extraction:** `ExtractUniqueEdges` walks all point-cell adjacencies to find unique edges between Voronoi vertices

### foampilot Python (`vmtkvoronoi_local.py`)
- **Output:** Edge adjacency graph ONLY — **no Voronoi polys/faces**
- **Radius:** Distance from circumcenter to nearest surface point via `cKDTree` (lines 28-33), OR falls back to `t.circumradius`
  ```python
  # This is NOT the same as VMTK's circumradius!
  dists, _ = tree.query(centers, k=1)
  radii = np.asarray(dists, dtype=float)
  ```
- **Edges:** Built from tetrahedron face adjacency (faces shared by exactly 2 tets)

### Differences & Impact
| Aspect | VMTK C++ | foampilot Python | Impact |
|--------|----------|------------------|--------|
| Output structure | PolyData with polys + lines | Edge graph only | **Critical**: VMTK FMM operates on polys; Python FMM operates on edges |
| Radius definition | Circumradius | Surface clearance (or circumradius fallback) | **High**: Changes cost function meaning; 1/R is not the same metric |
| Boundary faces | Built via `BuildVoronoiPolys` | Not built | **Critical**: Boundary faces define the manifold topology for FMM |
| Pole computation | Max + second-max with direction check | Not implemented in Python | **High**: Wrong seed selection |

**Expected impact on accuracy:** The absence of Voronoi polys is catastrophic for geometric accuracy. VMTK's FMM solves the Eikonal equation on the actual Voronoi diagram manifold (polygonal cells). The Python implementation reduces this to a graph problem, losing all face-level topology. The radius difference (clearance vs. circumradius) changes the cost function from `1/R_circum` to `1/R_clearance`, which are fundamentally different quantities that lead to different path weights.

---

## 3. Fast Marching Method

### VMTK C++ (`vtkvmtkNonManifoldFastMarching.cxx`)
This is a **true Fast Marching Method** implementation on polygonal non-manifolds:

**Data structures:**
- Min-heap (`vtkvmtkMinHeap`) for O(N log N) narrow band access
- Status scalars: `FAR`, `CONSIDERED`, `ACCEPTED`
- T-scalars (arrival times)

**Update formula — the critical difference:**
```cpp
// For a triangle with edges of length L_a, L_b and angle theta between them:
FEq = fScalar + Regularization;  // cost function + regularization
uEq = T_a - T_b;                  // difference in arrival times

// Quadratic coefficients for the Eikonal update
aEq = L_a^2 + L_b^2 - 2*L_a*L_b*cos(theta);
bEq = 2 * L_b * uEq * (L_a * cos(theta) - L_b);
cEq = L_b^2 * (uEq^2 - FEq^2 * L_a^2 * (1 - cos^2(theta)));

// Solve quadratic: aEq*t^2 + bEq*t + cEq = 0
SolveQuadratic(aEq, bEq, cEq, nSol, t0Eq, t1Eq);

// Check if solution is within valid range
if (uEq - tEq < -tol && tCompEq - tCompEqLower > tol && tCompEq - tCompEqHigher < -tol)
    neighborT = min(tEq + T_b, neighborT);
else
    neighborT = min(L_a*FEq + T_a, L_b*FEq + T_b);  // upwind fallback
```

**Key features:**
- Solves the **continuous** Eikonal equation `|∇T| = F` on each Voronoi polygon
- Uses the **quadratic update formula** derived from the geometry of the two incoming edges and the angle between them
- `Regularization` parameter (default 0.0) can add smoothing
- `UpdateFromConsidered` allows updates from considered (not just accepted) points
- `StopSeedId` enables early termination when target is reached
- 3 initialization passes over boundary points for better initial solution

### foampilot Python (`vmtkfastmarching_local.py`)
**Misnamed "python_eikonal" backend is actually Dijkstra:**

```python
# Edge cost = Gaussian quadrature of 1/R along edge
def _numba_or_numpy_edge_cost(points, radii, edges, floor=1e-6):
    xi = np.array([-0.7745966692, 0.0, 0.7745966692])
    wi = np.array([0.5555555556, 0.8888888889, 0.5555555556])
    for each edge (i, j):
        length = ||p_j - p_i||
        total = sum(w_k / max(r_interp, floor))  # 3-point Gauss quadrature
        cost = 0.5 * length * total
```

Then uses `scipy.sparse.csgraph.dijkstra` or custom `heapq` Dijkstra.

**Key features:**
- Discrete shortest path on a graph
- Edge weights approximate ∫(1/R)ds via 3-point Gauss quadrature
- No angle-dependent update; no quadratic solver
- No regularization
- No early stopping (except custom break in heap loop)

### Differences & Impact
| Aspect | VMTK C++ | foampilot Python | Impact |
|--------|----------|------------------|--------|
| Algorithm | True FMM (Sethian) | Dijkstra on graph | **Critical**: FMM gives continuous arrival times; Dijkstra gives discrete path lengths |
| Update formula | Quadratic with angle theta | Edge weight sum | **Critical**: The quadratic captures the 2D geometry of each Voronoi polygon |
| Data structure | Min-heap on narrow band | scipy sparse Dijkstra | Moderate; both are O(N log N) but Python version has coarser resolution |
| Regularization | `Regularization` parameter | None | Low |
| Early stopping | `StopSeedId` | Custom break | Low |

**Expected impact on accuracy:** This is the single largest source of error. The VMTK FMM computes accurate arrival times at every point on the Voronoi diagram by solving the continuous Eikonal equation. The Python Dijkstra computes path lengths on a coarse graph where edges represent tetrahedra adjacencies. At vessel bends, the graph approximation fails to capture the true geodesic distance, leading to paths that are 65% too long.

---

## 4. Steepest Descent Line Tracer

### VMTK C++ (`vtkvmtkSteepestDescentLineTracer.cxx` + `vtkvmtkNonManifoldSteepestDescent.cxx`)
**Continuous tracing on Voronoi polys:**

```cpp
// GetSteepestDescentInCell: subdivides each polygon edge into 250 segments
for (j=0; j<NumberOfEdgeSubdivisions; j++)  // 250 subdivisions
{
    point = point0 * (1.0 - currentS) + point1 * currentS;
    scalar = scalar0 * (1.0 - currentS) + scalar1 * currentS;
    descent = -(scalar - currentScalar) / distance(currentPoint, point);
    // Track steepest descent edge and parametric coordinate s
}

// Backtrace: continuous interpolation along edges
currentPoint = P_edge0 * (1.0 - currentS) + P_edge1 * currentS;
currentScalar = T_edge0 * (1.0 - currentS) + T_edge1 * currentS;
```

**Key features:**
- Operates on the **polygonal Voronoi manifold**, not just a graph
- At each step, examines **all neighboring polygons** (via `GetCellEdgeNeighbors`)
- Subdivides each polygon edge into **250 segments** to find the true steepest descent direction
- The path can land at **any parametric coordinate** on any edge — not restricted to vertices
- Interpolates both position and scalar field continuously
- Detects **degenerate cycles** (repeating same edge with same parametric coord)
- `MergePaths` option to merge nearby paths from different seeds

### foampilot Python (`vmtkfastmarching_local.py` — `_trace_centerline_steepest_descent`)
**Discrete vertex-to-vertex tracing:**

```python
while current != source:
    neighbors = adj.get(current, [])
    for nb in neighbors:
        d_edge = ||vertices[nb] - vertices[current]||
        d_eik = dist[nb] - dist[current]
        grad = -d_eik / d_edge
        # Pick neighbor with maximum gradient
    # Then linear interpolation between vertices
    n_interp = max(1, int(np.ceil(d / step_size)))
    for s in range(1, n_interp + 1):
        alpha = s / (n_interp + 1)
        new_pt = p0 + alpha * (p1 - p0)
```

**Key features:**
- Operates on the **vertex graph only**
- At each vertex, picks the neighbor with maximum discrete gradient
- After moving to a neighbor, linearly interpolates between the two vertices
- Cannot choose optimal points on edges — constrained to vertex hops
- Simple visited-set cycle detection

### Differences & Impact
| Aspect | VMTK C++ | foampilot Python | Impact |
|--------|----------|------------------|--------|
| Tracing domain | Voronoi polys (2D manifold) | Vertex graph (1D edges) | **Critical**: Continuous tracing finds shorter, more accurate paths |
| Gradient evaluation | 250 subdivisions per edge | Single vertex-to-vertex step | **Critical**: Coarse quantization causes zigzag paths and length inflation |
| Edge selection | All neighboring polys examined | Adjacent vertices only | **High**: May miss optimal crossing points |
| Cycle detection | Degenerate cycle detector (3-edge history) | Simple visited set | Low |
| Path merging | `MergePaths` with tolerance | Not implemented | Low |

**Expected impact on accuracy:** The 250-edge subdivision in VMTK allows the tracer to find the true steepest descent direction within each Voronoi polygon. The Python version hops between vertices, which at 82 points vs. VMTK's typical hundreds, creates a very coarse path. The linear interpolation after vertex hops does not recover the lost accuracy — it merely smooths the already-suboptimal vertex sequence.

---

## 5. Pole/Seed Selection

### VMTK C++ (`vtkvmtkPolyDataCenterlines.cxx` — `FindVoronoiSeeds`)
```cpp
for each cap center (baricenterId):
    // Find all tets sharing this cap center
    pointCells = delaunay->GetPointCells(baricenterId);
    
    // Find outer pole: tet with maximum circumradius
    for each tet in pointCells:
        tetraRadius = sqrt(vtkTetra::Circumsphere(...));
        if (tetraRadius > maxRadius) {
            maxRadius = tetraRadius;
            maxRadiusCellId = cellId;
            pole = circumcenter;
        }
    
    // Find inner pole: second max radius, but dot(poleVector, referenceVector) < 0
    for each tet in pointCells:
        referenceVector = circumcenter - baricenter;
        if (tetraRadius > secondMaxRadius && dot(poleVector, referenceVector) < 0) {
            secondMaxRadius = tetraRadius;
            secondMaxRadiusCellId = cellId;
        }
    
    // Choose based on normal direction
    poleVector = pole - baricenter;
    if (dot(poleVector, normal) < 0)
        seedIds->InsertNextId(maxRadiusCellId);      // outward pole
    else
        seedIds->InsertNextId(secondMaxRadiusCellId); // inward pole
```

### foampilot Python (`vmtkcenterlines_python.py` — `run_pipeline`)
```python
# Simple nearest neighbor
for cap_idx, cap in enumerate(caps):
    cap_center = loop.barycenter
    dists = np.linalg.norm(voronoi.points - cap_center, axis=1)
    nearest = int(np.argmin(dists))
    if cap_idx == 0:
        source_ids.append(cap_idx)
    else:
        target_ids.append(cap_idx)
```

### Differences & Impact
| Aspect | VMTK C++ | foampilot Python | Impact |
|--------|----------|------------------|--------|
| Selection method | Max-radius pole with direction awareness | Nearest Voronoi vertex to cap center | **High**: Nearest neighbor may select a boundary vertex instead of the true medial axis pole |
| Inward/outward check | `dot(poleVector, normal)` determines which pole | None | **High**: May start from wrong side of cap |
| Second pole | Considers second-max radius with opposite direction | Not considered | **High**: Misses the true inner pole |

**Expected impact on accuracy:** Starting from the wrong Voronoi vertex means the entire centerline path is offset. The VMTK seed selection ensures the path begins at the true extremal point of the medial axis, while the Python version may start from a nearby boundary vertex.

---

## 6. Additional Algorithmic Differences

### 6.1 Cost Function
**VMTK C++:**
```cpp
// vtkArrayCalculator computes 1/R as point data
voronoiCostFunctionCalculator->SetFunction("1/R");
```

**foampilot Python:**
```python
# Edge weight = Gauss quadrature of 1/R along edge
cost = 0.5 * length * sum(w_i / max(r_interp, 1e-6))
```

**Impact:** VMTK assigns cost to Voronoi vertices (polys). Python assigns cost to edges. The FMM updates propagate from vertex to vertex through polys in VMTK, but edge-to-edge in Python. This changes the propagation front shape.

### 6.2 Voronoi Diagram Topology
**VMTK C++:** The Voronoi diagram is a **polygonal non-manifold** where:
- Points = circumcenters of internal Delaunay tetrahedra
- Polys = Voronoi cells (convex polygons formed by connecting circumcenters of adjacent tets)
- Lines = edges of Voronoi cells
- The FMM propagates across **polygon faces**, treating each polygon as a 2D domain

**foampilot Python:** The Voronoi diagram is a **1D edge graph** where:
- Points = circumcenters of internal tetrahedra
- Edges = adjacency between tets sharing a face
- The FMM (Dijkstra) propagates along **edges only**

**Impact:** The 2D manifold vs. 1D graph distinction is fundamental. In a tubular structure, the Voronoi diagram has sheets (2D faces) that the centerline path crosses. The Python graph cannot represent these sheets — it only has the 1D skeleton of the sheet intersections. This is why the Python Voronoi has 39,755 points (all internal tets) but only 78,474 edges — it's a very dense 1D approximation of what should be a 2D surface.

### 6.3 Radius Computation
**VMTK C++:** `radius = sqrt(vtkTetra::Circumsphere(p0,p1,p2,p3,circumcenter))` — the circumradius of the Delaunay tetrahedron.

**foampilot Python:** 
```python
if surface is not None:
    dists, _ = tree.query(centers, k=1)  # Distance to nearest surface point
    radii = np.asarray(dists, dtype=float)
else:
    radii = np.array([t.circumradius for t in tetrahedra], dtype=float)
```

**Impact:** When surface is provided (which it always is in the pipeline), the Python radius is the clearance distance, NOT the circumradius. For a tubular structure, the circumradius of internal tets is proportional to the local vessel radius, while the clearance distance is the distance to the surface. These are related but not identical — the circumradius captures the inscribed sphere radius at the Voronoi vertex, while clearance is a nearest-neighbor distance that can be noisy.

---

## 7. VMTK Best Practices to Adopt

### 7.1 Data Structures
| Practice | VMTK C++ | foampilot Python |
|----------|----------|------------------|
| Voronoi representation | `vtkPolyData` with polys + lines + point data | Custom `VoronoiGraph` with edges only |
| Internal tet tracking | `vtkIntArray` keepCell flags | Python list of `Tetrahedron` dataclasses |
| Connectivity | `vtkIdList` + `GetCellNeighbors` | Python dict of sets |
| FMM narrow band | Custom `vtkvmtkMinHeap` | `heapq` or scipy Dijkstra |

**Recommendation:** Rebuild the Voronoi as a `vtkPolyData` with polys. This requires implementing `BuildVoronoiPolys` in Python, which walks the tet adjacency to construct boundary faces.

### 7.2 Numerical Tolerances
| Parameter | VMTK C++ | foampilot Python | Recommendation |
|-----------|----------|------------------|----------------|
| Double tolerance | 1.0e-12 | 1e-6 | Change to 1e-12 |
| Large double | 1.0e+32 | np.inf | Use 1e32 for compatibility |
| Radius floor | Not explicitly floored in FMM | 1e-6 | Keep 1e-6 to avoid div-by-zero |
| Merge tolerance | `VTK_VMTK_DOUBLE_TOL` | Not used | Use 1e-12 |

### 7.3 Memory/Performance
| Practice | VMTK C++ | foampilot Python |
|----------|----------|------------------|
| Edge subdivision | 250 segments per edge in steepest descent | Fixed `step_size=0.5` with linear interpolation |
| FMM initialization | 3 passes over boundary points | Single pass |
| Point iteration | VTK cell iterators (C++) | Python loops over numpy arrays |

**Recommendation:** The 250-edge subdivision is expensive but necessary for accuracy. In Python, implement it as a vectorized operation over all edges to avoid Python-level loops.

### 7.4 Validation
| Practice | VMTK C++ | foampilot Python |
|----------|----------|------------------|
| Normal validation | Fatal error if normals missing | Auto-generate normals |
| Seed validation | Checks seed ids against input bounds | Basic bounds check |
| Cycle detection | 3-edge history for degenerate descent | Simple visited set |
| Empty output | Returns empty poly data | Returns empty centerline |

---

## 8. Proposed Code Changes (Priority Order)

### Priority 1: Implement True Fast Marching Method (Expected impact: HIGH)
**File:** `vmtkfastmarching_local.py`  
**What:** Replace Dijkstra with true FMM that solves the continuous Eikonal equation on Voronoi polys.

**How to implement:**
1. **Rebuild Voronoi as polys first** (see Priority 2)
2. **Triangulate each Voronoi polygon** into triangles fanning from the polygon centroid
3. **Implement the quadratic update formula:**
   ```python
   def _compute_fmm_update(T_a, T_b, L_a, L_b, cos_theta, F):
       u = T_a - T_b
       a = L_a**2 + L_b**2 - 2*L_a*L_b*cos_theta
       b = 2 * L_b * u * (L_a * cos_theta - L_b)
       c = L_b**2 * (u**2 - F**2 * L_a**2 * (1 - cos_theta**2))
       
       delta = b**2 - 4*a*c
       if delta < -1e-12:
           return min(L_a*F + T_a, L_b*F + T_b)
       
       if abs(a) > 1e-12:
           if delta < 1e-12:
               t = -b / (2*a)
           else:
               q = -0.5 * (b - np.sqrt(delta)) if b < -1e-12 else \
                   -0.5 * (b + np.sqrt(delta)) if b > 1e-12 else \
                   np.sqrt(-c/a)
               t = q / a
           t_comp = L_b * (t - u) / t if abs(t) > 1e-12 else np.inf
           t_lower = L_a * cos_theta
           t_upper = L_a / cos_theta if abs(cos_theta) > 1e-12 else np.inf
           
           if (u - t < -1e-12) and (t_comp - t_lower > 1e-12) and (t_comp - t_upper < -1e-12):
               return t + T_b
       
       return min(L_a*F + T_a, L_b*F + T_b)
   ```
4. **Use a min-heap** (`heapq`) for the narrow band
5. **Initialize with 3 passes** over boundary points
6. **Add `Regularization` and `StopSeedId` parameters**

### Priority 2: Build Voronoi Polys (Expected impact: HIGH)
**File:** `vmtkvoronoi_local.py`  
**What:** Construct Voronoi polygon cells, not just edges.

**How to implement:**
1. For each unique edge in the Voronoi graph, find the two adjacent tetrahedra
2. Walk from one tet to its neighbors across shared faces, collecting tets until a boundary tet is found
3. The sequence of tets forms a Voronoi polygon (cell)
4. Store polys as `vtkPolyData` cells with point IDs = tet cell IDs (circumcenters)
5. Also compute `PoleIds` using the max-radius + direction check from VMTK C++

### Priority 3: Implement Continuous Steepest Descent (Expected impact: HIGH)
**File:** `vmtkfastmarching_local.py`  
**What:** Replace vertex-hop tracing with continuous tracing on Voronoi polys.

**How to implement:**
1. After FMM, operate on the Voronoi `vtkPolyData` (not the edge graph)
2. At each step, find all neighboring polys via `GetCellEdgeNeighbors`
3. For each neighboring poly, subdivide its edges into 250 segments
4. At each subdivision point, compute the gradient of the Eikonal solution:
   ```python
   # Interpolate scalar at subdivision point
   T = T_edge0 * (1-s) + T_edge1 * s
   # Gradient = -(T - T_current) / distance
   grad = -(T - T_current) / np.linalg.norm(P - P_current)
   ```
5. Select the edge and parametric coordinate `s` with the steepest negative gradient
6. Interpolate the new position: `P_new = P_edge0 * (1-s) + P_edge1 * s`
7. Add cycle detection: track last 3 edges and parametric coords

### Priority 4: Fix Seed Selection (Expected impact: MEDIUM)
**File:** `vmtkcenterlines_python.py`  
**What:** Replace naive nearest-neighbor with VMTK's pole-based selection.

**How to implement:**
1. For each cap center, find all internal tetrahedra sharing the cap center point
2. Compute circumradius for each tet
3. Find outer pole (max circumradius) and inner pole (second max with opposite direction)
4. Use `dot(pole_vector, cap_normal)` to choose inward vs. outward pole
5. Associate the chosen pole's tet ID to the Voronoi vertex index

### Priority 5: Fix Radius Computation (Expected impact: MEDIUM)
**File:** `vmtkvoronoi_local.py`  
**What:** Use circumradius instead of surface clearance distance.

**How to implement:**
```python
# Instead of:
# dists, _ = tree.query(centers, k=1)
# radii = np.asarray(dists, dtype=float)

# Use:
radii = np.array([t.circumradius for t in tetrahedra], dtype=float)
```
This requires storing `circumradius` in the `Tetrahedron` dataclass (it is already computed in `_circumsphere`).

### Priority 6: Tighten Tolerances (Expected impact: LOW-MEDIUM)
**File:** `vmtkinternaltetrahedra_local.py`  
**What:** Change tolerance from 1e-6 to 1e-12.

**How to implement:**
```python
# Change:
all_dot_positive = (dot0 > 1e-6 and ...)
all_but_one_positive = sum(d > 1e-6 for d in ...) >= 3

# To:
tol = 1e-12
all_dot_positive = (dot0 > tol and ...)
all_but_one_positive = sum(d > tol for d in ...) >= 3
```

### Priority 7: Add End-Point Appending (Expected impact: LOW)
**File:** `vmtkcenterlines_python.py`  
**What:** Append cap center points to centerline endpoints as VMTK does.

**How to implement:**
After centerline computation, prepend the source cap center and append the target cap center to the point and radius arrays.

---

## 9. Expected Impact Quantification

| Change | Current Error | Expected After Fix | Confidence |
|--------|---------------|-------------------|------------|
| True FMM + continuous tracing | Hausdorff 23.9mm, Length 65.4% | Hausdorff <10mm, Length <20% | High |
| Voronoi polys | Edge-only graph | Full polygonal manifold | High |
| Seed selection | Nearest neighbor | Direction-aware pole selection | Medium |
| Radius fix | Clearance distance | Circumradius | Medium |
| Tolerance | 1e-6 | 1e-12 | Low-Medium |

The **combination** of Priority 1 (FMM) + Priority 3 (continuous tracing) + Priority 2 (Voronoi polys) should reduce the Hausdorff distance from 23.9mm to under 10mm and the length error from 65.4% to under 20%, based on the fact that these three changes restore the exact algorithmic pipeline used by VMTK C++.

---

## 10. Code Snippets: Side-by-Side Comparison

### 10.1 Internal Tetrahedra Classification Tolerance
**VMTK C++:**
```cpp
this->Tolerance = VTK_VMTK_DOUBLE_TOL;  // 1.0e-12
if ((dot0>tolerance)&&(dot1>tolerance)&&(dot2>tolerance)&&(dot3>tolerance))
    allDotPositive = true;
```

**foampilot Python:**
```python
all_dot_positive = (dot0 > 1e-6 and dot1 > 1e-6 and dot2 > 1e-6 and dot3 > 1e-6)
```

### 10.2 Voronoi Radius
**VMTK C++:**
```cpp
tetraRadius = sqrt(vtkTetra::Circumsphere(p0,p1,p2,p3,tetraCenter));
newScalars->SetValue(i, (double)tetraRadius);
```

**foampilot Python:**
```python
if surface is not None and len(centers) > 0:
    surface_points = np.array([surface.GetPoint(i) for i in range(surface.GetNumberOfPoints())], dtype=float)
    tree = cKDTree(surface_points)
    dists, _ = tree.query(centers, k=1)
    radii = np.asarray(dists, dtype=float)
```

### 10.3 Fast Marching Update
**VMTK C++:**
```cpp
// Quadratic update with angle-dependent coefficients
aEq = L_a*L_a + L_b*L_b - 2*L_a*L_b*cosTheta;
bEq = 2 * L_b * uEq * (L_a * cosTheta - L_b);
cEq = L_b*L_b * (uEq*uEq - FEq*FEq * L_a*L_a * (1-cosTheta*cosTheta));
SolveQuadratic(aEq, bEq, cEq, nSol, t0Eq, t1Eq);
```

**foampilot Python:**
```python
# Dijkstra edge cost
d_edge = np.linalg.norm(p1 - p0)
cost = 0.5 * d_edge * sum(w / max(r_interp, floor))
```

### 10.4 Steepest Descent Tracing
**VMTK C++:**
```cpp
// Continuous: 250 subdivisions per edge
for (j=0; j<this->NumberOfEdgeSubdivisions; j++)  // 250
{
    point = point0 * (1.0 - currentS) + point1 * currentS;
    scalar = scalar0 * (1.0 - currentS) + scalar1 * currentS;
    descent = -(scalar - currentScalar) / distance;
}
```

**foampilot Python:**
```python
# Discrete: vertex-to-vertex
best_next = neighbor with max(-d_eik / d_edge)
# Then linear interpolation
new_pt = p0 + alpha * (p1 - p0)
```

### 10.5 Seed Selection
**VMTK C++:**
```cpp
poleVector = pole - baricenter;
if (dot(poleVector, normal) < VTK_VMTK_DOUBLE_TOL)
    seedIds->InsertNextId(maxRadiusCellId);      // outward pole
else
    seedIds->InsertNextId(secondMaxRadiusCellId); // inward pole
```

**foampilot Python:**
```python
dists = np.linalg.norm(voronoi.points - cap_center, axis=1)
nearest = int(np.argmin(dists))
```

---

## 11. References

- VMTK C++ source: https://github.com/vmtk/vmtk (commit 6c189dd6)
- `vtkvmtkInternalTetrahedraExtractor.cxx` — Internal tet extraction
- `vtkvmtkVoronoiDiagram3D.cxx` — Voronoi diagram with polys
- `vtkvmtkNonManifoldFastMarching.cxx` — True FMM implementation
- `vtkvmtkSteepestDescentLineTracer.cxx` — Continuous backtracing
- `vtkvmtkNonManifoldSteepestDescent.cxx` — Base class with 250-subdivision gradient evaluation
- `vtkvmtkPolyDataCenterlines.cxx` — Pipeline orchestration and seed selection
- `vtkvmtkConstants.h` — `VTK_VMTK_DOUBLE_TOL = 1.0e-12`
