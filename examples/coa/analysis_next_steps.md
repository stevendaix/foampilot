# VMTK Centerline Implementation Review & Next Steps

**Date:** 2026-08-19  
**Author:** Code Review  
**Benchmark baseline (python_eikonal backend):** Hausdorff 23.8 mm, Length error 65.7 %, Radius error 0.8 %, Points 81 vs 409 ref, Total time 17.5 s

---

## 1. Code Review Findings

### 1.1 Current Implementation Status

The `foampilot/src/foampilot/geometry/topology/vmtk/` package contains ~4,800 lines of Python across 23 modules. The high-level pipeline in `vmtkcenterlines_python.py` correctly follows the VMTK sequence: preprocess → cap → Delaunay3D → internal tets → Voronoi dual → fast marching / pathfinding → resample → sections → network. This is a significant architectural improvement over the legacy `vmtkcenterlines.py`, which incorrectly used `scipy.spatial.Voronoi` on surface points (a fundamentally wrong algorithm).

### 1.2 Bugs and Inefficiencies

#### Critical: Misnamed and Dead Code in Fast Marching Module

**File:** `vmtkfastmarching_local.py`

| Issue | Severity | Details |
|-------|----------|---------|
| `"python_eikonal"` backend is **not** an eikonal solver | **High** | `run_pipeline` passes `backend="python_eikonal"` by default. In `Execute()`, this branch calls `scipy.sparse.csgraph.dijkstra` on a static weighted graph. There is no eikonal equation solving. |
| `_python_eikonal_backend` is **dead code** | **High** | This function (lines 237–285) implements a heap-based Dijkstra variant and is never called by `Execute()`. The default branch calls `dijkstra()` directly from scipy. |
| `_true_fmm_backend` is mathematically incomplete | **High** | This function (lines 288–397) attempts a quadratic FMM update but operates on a **1-D edge graph**, not on 2-D Voronoi polygons. On a 1-D graph, each neighbor has at most one ALIVE predecessor, making the quadratic update degenerate in most cases and falling back to linear edge relaxation. It is not a true FMM. |

#### High: Voronoi Polys Built but Discarded

**File:** `vmtkvoronoi_local.py`

`_build_voronoi_polys()` (lines 80–135) correctly constructs Voronoi polygon cells and stores them as `result.polys` and `result.polys_edges`. However:
- `VoronoiGraph` dataclass does not declare `polys` or `polys_edges` fields.
- `vmtkfastmarching_local.py` never accesses `polys` or `polys_edges`.
- `_voronoi_to_vtp()` in `vmtkcenterlines_python.py` does not write polys.

This means the 2-D manifold topology required by VMTK's FMM is computed but immediately discarded.

#### High: EDT-Based Pole Selection is Dead Code

**File:** `vmtkfastmarching_local.py`, lines 571–631

`_compute_edt_poles()` implements an elaborate EDT-based pole selection with inward-direction checking, clearance scoring, and cap-normal alignment. The computed `Pole` objects are stored in `self.Poles` but **never consumed** by the pathfinder. Source/target selection in `Execute()` (lines 530–534) simply finds the nearest Voronoi vertex to each cap center via `cKDTree.query()`. This is a pure performance waste and a maintenance hazard.

#### Medium: Tolerance Mismatch with VMTK C++

**File:** `vmtkinternaltetrahedra_local.py`, line 174

```python
all_dot_positive = (dot0 > 1e-6 and dot1 > 1e-6 and dot2 > 1e-6 and dot3 > 1e-6)
```

VMTK C++ uses `VTK_VMTK_DOUBLE_TOL = 1.0e-12`. The Python `1e-6` threshold is 6 orders of magnitude looser. While the current code correctly uses the circumcenter + dot-product method (contradicting the older analysis document), the loose tolerance can misclassify boundary tetrahedra that VMTK would accept, altering Voronoi connectivity.

#### Medium: Radius Definition Mismatch

**File:** `vmtkvoronoi_local.py`, lines 28–32

When a surface is provided (always true in the pipeline), radii are replaced with **Euclidean distance to the nearest surface point** via `cKDTree`. VMTK uses the **tetrahedron circumradius**. These are different quantities:
- Circumradius ≈ local vessel radius (smooth, Delaunay-derived).
- Distance-to-wall ≈ surface clearance (noisy, nearest-neighbor).

This changes the cost function from `1/R_circum` to `1/R_clearance`, which is systematically smaller and introduces surface-sampling noise.

#### Medium: Seed Selection is Naive Nearest-Neighbor

**File:** `vmtkcenterlines_python.py`, lines 289–298

```python
dists = np.linalg.norm(voronoi.points - cap_center, axis=1)
nearest = int(np.argmin(dists))
```

VMTK selects the **pole** (tetrahedron with maximum circumradius sharing the cap center), then checks that the pole vector is anti-aligned with the cap normal. Our nearest-vertex approach can select a boundary spike or wrong-side vertex, offsetting the initial centerline segment.

#### Medium: Subresolution Removal Threshold Bug

**File:** `vmtkinternaltetrahedra_local.py`, lines 194–217

The subresolution removal loop checks `if t.circumradius < 1e-6: continue` (line 198). This hardcoded floor means tetrahedra with circumradius between `1e-6` and `subresolution_factor * min_surface_edge` are **not** skipped, but the comparison `t.circumradius < subresolution_factor * min_surface_edge` still works. However, the `1e-6` floor is arbitrary and not documented. More importantly, VMTK computes `minEdgeLength` from triangle area (`sqrt(2.0 * triangleArea)`), while Python computes the actual minimum edge length. The two thresholds are not equivalent.

#### Low: Missing Error Handling in Pipeline

**File:** `vmtkcenterlines_python.py`

- Line 296: `cap_center = np.array([0.0, 0.0, 0.0])` fallback when no matching loop is found. This silently produces a bogus seed at the origin.
- No validation that `source_ids` and `target_ids` are disjoint or within Voronoi bounds before pathfinding.
- `resample_centerline` does not validate that `abscissas` is monotonically increasing.

#### Low: Performance Inefficiency

**File:** `vmtkcenterlineresampling_local.py`, lines 48–67

`_taubin_smooth` iterates over points in a Python `for` loop (O(n × neighbors)). For centerlines with thousands of points, this is slow. Vectorized Laplacian smoothing would be faster.

### 1.3 Structural Assessment

**Strengths:**
- Clean separation of concerns across phases.
- Dataclass-based data flow (`Tetrahedron`, `VoronoiGraph`, `Centerline`, `PipelineReport`) is type-safe and maintainable.
- The Delaunay-dual Voronoi construction is mathematically correct.
- Subresolution tetrahedra removal exists and is parameterized.
- Seed-component extraction from the connectivity graph is correct.

**Weaknesses:**
- The fast marching module conflates three different algorithms (Dijkstra, graph relaxation, FMM) under misleading names.
- Voronoi polys are computed but orphaned—they exist in `_build_voronoi_polys` but are never integrated into the data model or downstream consumers.
- The EDT pole computation is a "black hole"—expensive computation with no consumer.
- No unit tests exist for the VMTK pipeline (`test_topology_with_centerline.py` tests only the axis-extraction and profile-classification utilities, not the full centerline reconstruction).

---

## 2. Impact Assessment of Missing Features

| Missing Feature | Expected Hausdorff Improvement | Expected Length Error Improvement | Implementation Complexity | Performance Impact |
|-----------------|-------------------------------|----------------------------------|---------------------------|-------------------|
| **True FMM on Voronoi polys** | High (−10 to −15 mm) | High (−25 to −35 %) | **High** (3–5 days) | Moderate (+2–5 s) |
| **Continuous steepest descent tracing** | High (−5 to −10 mm) | High (−15 to −25 %) | **High** (2–4 days) | Low (+0.5–1 s) |
| **Voronoi polys construction & integration** | Medium (−3 to −5 mm) | Medium (−5 to −10 %) | **Medium** (1–2 days) | Low |
| **Fix internal tet tolerance (1e-6 → 1e-12)** | Low-Medium (−1 to −3 mm) | Low (−2 to −5 %) | **Low** (0.5 day) | None |
| **Switch radii to circumradius** | Low (−1 to −2 mm) | Low-Medium (−3 to −8 %) | **Low** (0.5 day) | None |
| **Fix pole/seed selection** | Medium (−3 to −5 mm) | Low (−2 to −5 %) | **Medium** (1–2 days) | Low |
| **Add SimplifyVoronoi** | Low (−1 to −2 mm) | Low (−1 to −3 %) | **Medium** (1–2 days) | Low (+0.5 s) |
| **StopFastMarchingOnReachingTarget** | None | None | **Low** (0.5 day) | Moderate (−5 to −30 % wall time) |

**Rationale for estimates:**
- The dominant errors (Hausdorff 23.8 mm, length error 65.7 %) come from the pathfinder operating on a coarse 1-D graph with Dijkstra, not following the true eikonal gradient, and starting from wrong seeds. True FMM + continuous tracing address the root cause directly.
- Tolerance, radius, and seed fixes are "hygiene" improvements that clean up the input to the pathfinder but cannot overcome the fundamental limitation of graph-based shortest-path on a 1-D skeleton.
- SimplifyVoronoi and early stopping are secondary; they improve robustness and performance but do not fix the core geometric gap.

---

## 3. Prioritized Roadmap

### Step 1: Integrate Voronoi Polys into the Data Model and Fast Marching

**Priority:** 1 (foundational)  
**Files to modify:**
- `vmtkvoronoi_local.py` — add `polys` and `polys_edges` to `VoronoiGraph` dataclass.
- `vmtkfastmarching_local.py` — extend `VoronoiGraph` to carry polys; modify `_true_fmm_backend` to operate on polygon cells instead of 1-D edges.
- `vmtkcenterlines_python.py` — pass polys through to the fast marcher.

**Key functions to add/modify:**
- `VoronoiGraph.__post_init__` — validate polys/edges consistency.
- `vmtkFastMarchingLocal._build_polygon_adjacency()` — map each polygon to its edge neighbors (VMTK's `GetCellEdgeNeighbors` equivalent).
- `vmtkFastMarchingLocal._fmm_update_polygon()` — for each polygon, collect ALIVE neighbors, compute the quadratic update using edge lengths and `cosTheta`, solve the quadratic, and update the CONSIDERED heap.

**How to test:**
- Unit test: construct a synthetic tubular mesh (e.g., a 10 mm cylinder with radius 1 mm), compute Delaunay + Voronoi, and verify that each Voronoi polygon has ≥3 edges and that polygon adjacency is symmetric.
- Integration test: run the aorta benchmark and check that the FMM arrival times at Voronoi vertices are smoother than the Dijkstra distances (lower maximum gradient along any edge).

**Estimated effort:** 3–5 days.

---

### Step 2: Replace Dijkstra with True FMM on Polys

**Priority:** 2 (highest accuracy impact)  
**Files to modify:**
- `vmtkfastmarching_local.py` — rewrite `Execute()` to use a min-heap narrow-band FMM on polygon cells.

**Key functions to add/modify:**
- `vmtkFastMarchingLocal.Execute()` — replace the scipy Dijkstra call with a narrow-band FMM loop.
- `_solve_quadratic_update(T_a, T_b, L_a, L_b, cos_theta, F)` — vectorized quadratic solver matching VMTK's `SolveQuadratic`.
- `_fmm_initialize_boundary()` — 3-pass initialization from boundary points (VMTK's `InitPropagation`).
- `vmtkFastMarchingLocal._accept_and_update()` — pop min from heap, mark ALIVE, update CONSIDERED neighbors via polygon adjacency.

**How to test:**
- Unit test: on a 2-D triangular mesh with known analytic eikonal solution (e.g., `T = distance from origin / F`), verify that FMM arrival times converge to the analytic solution within 1 %.
- Integration test: aorta benchmark. Expect Hausdorff to drop from 23.8 mm toward <15 mm and length error from 65.7 % toward <40 %.

**Estimated effort:** 3–5 days.

---

### Step 3: Implement Continuous Steepest Descent Tracing on Polys

**Priority:** 3 (highest accuracy impact after FMM)  
**Files to modify:**
- `vmtkfastmarching_local.py` — add a steepest descent tracer that operates on Voronoi polys.

**Key functions to add/modify:**
- `vmtkFastMarchingLocal._trace_steepest_descent(source_idx, target_idx)` — continuous backtracing.
- `_find_steepest_descent_edge(current_pos, current_T, polygon_id)` — for each edge of the current polygon, subdivide into `N_SUBDIVISIONS=250` segments (VMTK default), interpolate `T` and `R`, compute gradient `-(T - current_T) / distance`, select the edge/parameter with steepest negative gradient.
- `_detect_degenerate_cycle(history)` — track last 3 edge+parameter tuples; abort if a cycle is detected.

**How to test:**
- Unit test: on a straight cylinder, the steepest descent path should be a straight line with constant radius.
- Unit test: on a curved tube (e.g., 90° bend), verify that the traced centerline follows the medial axis and that point spacing is roughly uniform (~0.5 mm).
- Integration test: aorta benchmark. Expect point count to increase from 81 toward 300+, Hausdorff to drop further, length error to drop below 30 %.

**Estimated effort:** 2–4 days.

---

### Step 4: Fix Pole/Seed Selection and Radius Definition

**Priority:** 4 (medium impact, low effort)  
**Files to modify:**
- `vmtkcenterlines_python.py` — replace nearest-vertex seed selection with pole-based selection.
- `vmtkvoronoi_local.py` — switch radius from distance-to-wall to circumradius.

**Key functions to add/modify:**
- `vmtkCenterlinesPython._select_voronoi_seeds(caps, loops, voronoi, internal_tets)` — for each cap center, find the internal tet sharing the cap center point with maximum circumradius; check `dot(pole - cap_center, cap_normal) < 0`; fall back to second-max if needed.
- `build_voronoi_from_tetrahedra()` — remove the `cKDTree` distance-to-wall branch; always use `t.circumradius`.

**How to test:**
- Unit test: on a cylinder, verify that the selected seed is the Voronoi vertex at the cylinder axis, not a boundary vertex.
- Integration test: aorta benchmark. Expect initial-segment offset to decrease, contributing ~2–5 mm Hausdorff reduction.

**Estimated effort:** 1–2 days.

---

### Step 5: Tighten Tolerances, Add SimplifyVoronoi, and Early Stopping

**Priority:** 5 (polish)  
**Files to modify:**
- `vmtkinternaltetrahedra_local.py` — change tolerance from `1e-6` to `1e-12`.
- `vmtkvoronoi_local.py` — add `simplify_voronoi()` function to remove `ncells == 1` boundary spikes.
- `vmtkfastmarching_local.py` — add `StopSeedId` early termination to the FMM loop.

**How to test:**
- Unit test: verify that tightening tolerance to `1e-12` does not break classification on a known-good tetrahedral mesh.
- Unit test: construct a Voronoi diagram with a known boundary spike (a vertex used by only one cell); verify that `simplify_voronoi` removes it.
- Integration test: aorta benchmark with `StopFastMarchingOnReachingTarget=True`. Verify that path quality is unchanged and wall time decreases.

**Estimated effort:** 1–2 days.

---

## 4. Alternative Approaches (80/20 Rule)

If implementing full VMTK parity (Steps 1–3) is too costly, the following alternatives can capture most of the benefit with less effort:

### Alternative A: Improved Dijkstra with Sub-Edge Tracing (Estimated 1–2 days)

**Concept:** Keep the current `scipy.sparse.csgraph.dijkstra` for pathfinding (fast, robust), but replace the vertex-hop tracer with a **continuous steepest descent tracer on the edge graph**.

**Changes:**
1. After Dijkstra computes `dist[]` and `predecessor[]`, backtrack from target to source to get the vertex path.
2. For each edge `(u, v)` in the path, subdivide into 250 segments.
3. At each subdivision point `s ∈ (0, 1)`, interpolate `T(s) = (1-s) * T[u] + s * T[v]` and compute the local gradient.
4. Select the subdivision point with the steepest descent direction across all edges, not just the next vertex.

**Impact:** This captures ~60 % of the continuous tracing benefit without rebuilding the entire FMM. Point count would increase from 81 to ~200–300. Expected Hausdorff reduction: 5–10 mm. Expected length error reduction: 15–20 %.

**Effort:** 1–2 days (pure Python, no new data structures).

### Alternative B: Radius Switch + Seed Fix + Tolerance Tighten (Estimated 0.5–1 day)

**Concept:** Make three small changes that together address the most damaging input errors:
1. Switch Voronoi radii to circumradius (`vmtkvoronoi_local.py`).
2. Fix pole selection to use max-radius + normal alignment (`vmtkcenterlines_python.py`).
3. Tighten internal tet tolerance to `1e-12` (`vmtkinternaltetrahedra_local.py`).

**Impact:** These are "input hygiene" fixes. They do not change the pathfinding algorithm, but they remove systematic biases that push the pathfinder away from the true medial axis. Expected Hausdorff reduction: 3–6 mm. Expected length error reduction: 5–10 %.

**Effort:** 0.5–1 day.

### Alternative C: Graph-Based FMM on Edge Graph (Estimated 2–3 days)

**Concept:** Implement a **1-D fast marching method** on the existing edge graph that is mathematically closer to the eikonal equation than Dijkstra.

**Changes:**
1. Replace the static Dijkstra with a narrow-band FMM that updates `T[i]` using a quadratic solver on pairs of incoming edges (even on a 1-D graph, each node can have up to 2 settled predecessors).
2. Add `Regularization` parameter and `StopSeedId` early termination.
3. Use the arrival-time field `T[]` for steepest descent tracing instead of Dijkstra's distance.

**Impact:** This gives a proper eikonal solution on the current graph topology without requiring Voronoi polys. Expected Hausdorff reduction: 5–8 mm. Expected length error reduction: 10–15 %.

**Effort:** 2–3 days.

### Recommended 80/20 Path

Implement **Alternative B first** (0.5–1 day) for immediate baseline improvement, then **Alternative A** (1–2 days) for the bulk of the tracing accuracy gain. This two-step sequence costs ≤3 days total and should reduce Hausdorff from 23.8 mm to **<15 mm** and length error from 65.7 % to **<40 %**. If further improvement is needed, add **Alternative C** (2–3 days) to reach **<10 mm Hausdorff** and **<20 % length error**.

---

## 5. Summary of Findings

### What the Code Does Well
- Correct high-level pipeline architecture (Delaunay dual, internal tet classification with circumcenter test, seed-component extraction).
- Clean dataclass interfaces and phase-separated execution.
- The legacy `vmtkcenterlines.py` (scipy Voronoi on surface points) is correctly deprecated in favor of `vmtkcenterlines_python.py`.

### What Is Broken or Missing
1. **No eikonal solver.** The default `"python_eikonal"` backend is scipy Dijkstra on a static graph. The true FMM implementation exists but is incomplete and operates on the wrong topology (1-D edges instead of 2-D polys).
2. **No continuous tracing.** The tracer hops vertex-to-vertex and linearly interpolates afterward. This is not steepest descent.
3. **Voronoi polys are orphaned.** They are computed but never integrated into the data model or the pathfinder.
4. **Dead code.** `_python_eikonal_backend` and `_compute_edt_poles` are expensive computations with zero consumers.
5. **Radius and seed definitions mismatch VMTK.** Distance-to-wall and nearest-vertex selection introduce systematic biases.

### Recommended Next Actions (Ordered by ROI)

| Step | Action | Files | Effort | Expected Benefit |
|------|--------|-------|--------|------------------|
| 1 | Switch radii to circumradius + fix seed selection + tighten tolerance | `vmtkvoronoi_local.py`, `vmtkcenterlines_python.py`, `vmtkinternaltetrahedra_local.py` | 0.5–1 d | Hausdorff −3–6 mm |
| 2 | Implement continuous steepest descent tracer on existing edge graph | `vmtkfastmarching_local.py` | 1–2 d | Hausdorff −5–10 mm, points ×3–4 |
| 3 | Implement true FMM on Voronoi polys (requires Step 4 first) | `vmtkfastmarching_local.py`, `vmtkvoronoi_local.py` | 3–5 d | Hausdorff <10 mm, length error <20 % |
| 4 | Build Voronoi polys into data model and polygon adjacency | `vmtkvoronoi_local.py`, `VoronoiGraph` | 1–2 d | Foundation for Step 3 |
| 5 | Add SimplifyVoronoi + StopFastMarching + remove dead code | `vmtkvoronoi_local.py`, `vmtkfastmarching_local.py` | 1 d | Robustness + performance |

Steps 1 + 2 alone (2–3 days of work) should bring the benchmark within an order of magnitude of VMTK accuracy. Steps 3 + 4 (4–7 days) are required for full parity.
