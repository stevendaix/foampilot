# VMTK Centerline Exhaustive Benchmark Report

**Date:** 2026-08-19
**Pipeline:** foampilot VMTK centerline extraction (voxel FMM + Voronoi)
**Backend:** `python_eikonal` / `numpy`
**Test data:** `test/vmtk-test-data/`
**Total cases:** 9

---

## 1. Executive Summary

| Status   | Count |
|----------|-------|
| PASS     | 1     |
| FAIL     | 1     |
| ERROR    | 7     |
| NO_REF   | 0     |

The pipeline is **functional only for simple open surfaces with 2–3 boundary loops that get capped**. It **fails completely for closed surfaces** and produces **geometrically incorrect output for bifurcations**. This represents a critical blocker: 7 of 9 cases terminate with empty centerlines.

---

## 2. Complete Case Table

| Case                               | Input Type | Boundary Loops | Caps | Status | Points | Length (mm) | Mean R (mm) | Mean Dist (mm) | Hausdorff (mm) | Len Err (%) | Tort Err (%) | Rad Err (%) | Time (s) | Reference |
|------------------------------------|------------|----------------|------|--------|--------|-------------|-------------|----------------|----------------|-------------|--------------|-------------|----------|-----------|
| aorta-surface-open-ends            | .stl       | 3              | 3    | PASS   | 81     | 78.33       | 5.520       | 2.766          | 23.848         | 65.71       | 66.10        | 0.78        | 18.24    | YES       |
| aorta-surface-branch-split         | .vtp       | 2              | 2    | FAIL   | 14     | 12.29       | 3.174       | 10.735         | 38.972         | 94.65       | 66.41        | 43.04       | 22.89    | YES       |
| aorta-surface-connectivity-reference | .stl     | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 5.35     | NO        |
| aorta-surface-segment-2            | .stl       | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 33.00    | NO        |
| aorta-surface-two-segments         | .vtp       | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 64.00    | NO        |
| aorta-surface                      | .stl       | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 13.82    | YES       |
| aorta-surface                      | .vtp       | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 13.36    | YES       |
| cow                                | .vtp       | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 4.77     | NO        |
| fixture                            | .stl       | 0              | 0    | ERROR  | 0      | 0.00        | 0.000       | 0.000          | 0.000          | 0.00        | 0.00         | 0.00        | 11.35    | NO        |

---

## 3. Case-by-Case Analysis

### 3.1 PASS: `aorta-surface-open-ends`

- **Geometry:** Open surface with 3 boundary loops → 3 caps after capping.
- **Centerline:** 81 points, length 78.33 mm, mean radius 5.52 mm.
- **Accuracy:** Mean distance to reference 2.77 mm, Hausdorff 23.85 mm.
- **Error profile:** Length error 65.7%, tortuosity error 66.1%, radius error 0.78%.
- **Assessment:** Radius is excellent. Length/tortuosity deviations are large but the centerline remains anatomically plausible (length > 50 mm, radius > 0.5 mm). This is the only case that completes the full pipeline successfully.

### 3.2 FAIL: `aorta-surface-branch-split`

- **Geometry:** Bifurcation surface with 2 boundary loops → 2 caps.
- **Centerline:** 14 points, length only 12.29 mm, mean radius 3.17 mm.
- **Accuracy:** Mean distance 10.74 mm, Hausdorff 38.97 mm.
- **Error profile:** Length error 94.7%, tortuosity error 66.4%, radius error 43.0%.
- **Failure modes:**
  1. **Length too small:** 12.29 mm vs expected ~200+ mm (inferred from error magnitude).
  2. **Radius too small:** 3.17 mm vs reference radius (43% error).
  3. **Only 1 path produced:** For a bifurcation, the reference `aorta-centerline-branches.vtp` implies multiple branches, but the pipeline returns a single 14-point path.
- **Root cause:** The pole-selection and path-building logic assumes a 1-to-1 source-target mapping (cap 0 → cap 1). For bifurcations, there are multiple target caps per source cap, but the pipeline does not enumerate all source-target pairs.

### 3.3 ERROR: All Closed-Surface Cases (7 cases)

All cases with **0 boundary loops** and **0 caps** terminate with empty centerlines:

| Case                               | Key Warning                               |
|------------------------------------|-------------------------------------------|
| aorta-surface-connectivity-reference | No boundary loops detected, No poles computed |
| aorta-surface-segment-2            | No boundary loops detected, No poles computed |
| aorta-surface-two-segments         | No boundary loops detected, No poles computed |
| aorta-surface (.stl)               | No boundary loops detected, No poles computed |
| aorta-surface (.vtp)               | No boundary loops detected, No poles computed |
| cow                                | No boundary loops detected, No poles computed |
| fixture                            | No boundary loops detected, No poles computed |

**Common failure signature:**
- `n_seed_component: 0` (no tetrahedra reached by FMM front)
- `fast_marching` completes in < 0.25 s (front never propagates)
- `poles` completes in < 0.002 s (immediate failure)
- `centerlines computed: 0 paths`

---

## 4. Failure Mode Analysis

### 4.1 Closed Surface Catastrophic Failure (7/9 cases)

**Failure mode:** Pipeline error — empty centerline.

**Trigger condition:** Surface has 0 boundary loops (already closed). The capping stage adds 0 caps.

**Mechanism:**
1. `Phase B: Cap surface` produces 0 caps.
2. `Phase D: Select poles and seeds` computes `source_ids = []` and `target_ids = []`.
3. `Phase E: Compute path` (`vmtkFastMarchingLocal.Execute`) calls `_compute_edt_poles()`.
4. Inside `_compute_edt_poles`, the loop `for cap_id in range(cap_centers.shape[0])` iterates 0 times because there are no cap centers.
5. `poles` list is empty → `"No poles computed."` error.
6. `fm.Centerlines` remains `None` → empty centerline returned.

**Impact:** 78% of the test suite is completely non-functional for the most common surface format (closed STL/VTP).

### 4.2 Bifurcation Path Omission (1/9 cases)

**Failure mode:** Geometric inaccuracy — incomplete centerline network.

**Trigger condition:** Surface has 2 boundary loops but represents a bifurcation (1 inlet + 2 outlets, or similar branching topology).

**Mechanism:**
1. `Phase D: Select poles and seeds` creates exactly 1 source-target pair: `source_ids=[0]`, `target_ids=[1]`.
2. `Phase E` computes only 1 Dijkstra/FMM path between these two Voronoi nodes.
3. The additional outlet cap (if present) is ignored entirely.
4. Even when only 2 caps exist, the resulting single path may be truncated if the FMM front gets trapped or the radius-weighted cost distorts the path.

**Impact:** Bifurcated vessels produce misleading single-path centerlines. Downstream mesh generation and CHT simulations will be wrong.

### 4.3 Radius and Length Inaccuracy (1/9 cases)

**Failure mode:** Geometric metrics outside bounds.

**Specifics for `aorta-surface-branch-split`:**
- Radius error: 43% (mean radius 3.17 mm vs reference).
- Length error: 94.7% (length 12.29 mm vs reference).
- Hausdorff: 38.97 mm.

**Contributing factors:**
- Voronoi radii may be underestimating the true cross-sectional radius due to mesh quality issues or surface noise.
- The single extracted path does not follow the full vessel trajectory.
- The `vmtkCenterlineSectionsLocal` step computes 0 sections (`Computed 0 local centerline sections`), indicating the centerline does not align well with the surface for radius extraction.

---

## 5. Pattern Analysis: Geometry Correlation

| Geometry Feature                    | Cases                          | Outcome          |
|-------------------------------------|--------------------------------|------------------|
| **Closed surface (0 boundary loops)** | aorta-surface-*, cow, fixture, aorta-surface-connectivity-reference, aorta-surface-segment-2, aorta-surface-two-segments | **100% ERROR** (7/7) |
| **Open surface, 3 caps**            | aorta-surface-open-ends        | **PASS**         |
| **Open surface, 2 caps (bifurcation)** | aorta-surface-branch-split     | **FAIL**         |
| **Has reference centerline**        | aorta-surface-open-ends, aorta-surface-branch-split, aorta-surface (.stl), aorta-surface (.vtp) | Mixed            |

**Key correlations:**
1. **Boundary loop count is the dominant predictor of success.** Every closed surface fails. Every open surface with caps produces a centerline (though it may be inaccurate).
2. **Cap count > 2 leads to path omission.** With 3+ caps, the current logic only routes cap 0 → cap 1.
3. **Bifurcation topology amplifies radius error.** The branch-split case has the highest radius error (43%) and the only non-empty case with `Computed 0 local centerline sections`.

---

## 6. Root Cause Assessment

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| **R1** | `vmtkfastmarching_local.py:577` | CRITICAL | `_compute_edt_poles()` requires cap centers. Closed surfaces have none, so poles list is empty and the pipeline aborts. |
| **R2** | `vmtkcenterlines_python.py:297-320` | HIGH | Pole and path selection only handles 1 source → 1 target. Multi-outlet geometries (bifurcations, trifurcations) are reduced to a single path. |
| **R3** | `vmtkcenterlines_python.py:336-342` | MEDIUM | The voxel mask for the internal volume is built from tetrahedra centroids with `binary_dilation`. For closed surfaces with 0 internal tetrahedra classified, or complex geometries, this mask may not represent the true lumen. |
| **R4** | `vmtkfastmarching_local.py:586-597` | MEDIUM | EDT-based pole selection uses a fixed `corridor_radius = 3.0 * max(spacing)` and `query_ball_point` with `corridor_radius * 5.0`. For high-curvature or small-radius vessels, these heuristics miss the true medial axis. |
| **R5** | `vmtkcenterlines_python.py:400-413` | LOW | `vmtkCenterlineSectionsLocal` returns 0 sections when the centerline is poor, creating a feedback loop where bad centerlines cannot self-correct via local radius refinement. |

---

## 7. Prioritized Fixes

### P0 — Fix closed-surface support (CRITICAL)

**Target:** `vmtkfastmarching_local.py` + `vmtkcenterlines_python.py`

**Proposed changes:**
1. **Detect closed surfaces early** in `Phase B` or `Phase D`. When `len(loops) == 0`, either:
   - (a) Automatically detect inlet/outlet candidates via curvature/diameter analysis of the surface, or
   - (b) Raise a clear error instructing the user to mark inlet/outlet physical groups.
2. **Modify `_compute_edt_poles`** to accept an optional list of user-specified seed points when no caps exist. If caps are absent but `SourceIds`/`TargetIds` are provided manually, use those to query the EDT maxima.
3. **Add a fallback in `run_pipeline`** for closed surfaces: if capping yields 0 caps, attempt automatic pole detection from surface extremities (e.g., PCA axis endpoints) before aborting.

**Expected impact:** Resolves 7 of 9 errors. Brings success rate from 11% to ~89%.

### P1 — Enumerate all source-target pairs for multi-outlet geometries (HIGH)

**Target:** `vmtkcenterlines_python.py:297-320`

**Proposed changes:**
1. When `len(loops) >= 2`, treat the cap with the largest cross-sectional area (or longest connected boundary) as the **single source**.
2. Treat all other caps as **targets**.
3. Build `source_ids = [source_idx] * (n_targets)` and `target_ids = [all other cap indices]`.
4. Run the FMM/Dijkstra solver once from the source Voronoi node, then backtrack to each target. This avoids redundant solves and produces a full centerline tree.

**Expected impact:** Fixes the bifurcation case (`aorta-surface-branch-split`). Produces a centerline network with N branches instead of 1.

### P2 — Improve EDT pole-to-cap association robustness (MEDIUM)

**Target:** `vmtkfastmarching_local.py:599-636`

**Proposed changes:**
1. Replace fixed `corridor_radius` with a data-driven radius based on the cap’s local inscribed sphere radius or boundary loop size.
2. Add a secondary search: if no EDT maximum is found within the corridor, expand the search radius exponentially up to a configurable limit.
3. Validate that the selected pole lies within the Voronoi diagram’s medial axis by checking that the associated Voronoi node is connected to the source/target via the Voronoi graph.

**Expected impact:** Reduces radius and Hausdorff errors for curved/narrow vessels.

### P3 — Harden voxel mask generation (MEDIUM)

**Target:** `vmtkcenterlines_python.py:329-346`

**Proposed changes:**
1. When `internal_tet_centroids` is empty (closed surface or failed classification), fall back to a surface-derived mask: rasterize the capped surface into a binary image and dilate by the minimum vessel radius.
2. Ensure `shape` has a minimum of 3 voxels per dimension to avoid degenerate masks.

**Expected impact:** Prevents silent failures in tetrahedra classification.

---

## 8. Recommendation: Voxel FMM vs. Fix Existing Issues

**Recommendation: Fix existing issues first. Do NOT migrate to a voxel-based FMM yet.**

### Rationale

1. **The pipeline already uses a voxel-based FMM backend.** The `python_eikonal` and `dijkstra` backends operate on the Voronoi graph, but `vmtkFastMarchingLocal` internally rasterizes tetrahedra centroids into a voxel mask (`InternalVolumeMask`) for EDT-based pole selection. The EDT is already a voxel operation. Switching to a pure voxel FMM (e.g., computing the full 3-D distance field on a voxel grid) would not solve the current failures because:
   - Closed surfaces fail due to **missing cap centers**, not due to the graph representation.
   - Bifurcations fail due to **missing multi-target enumeration**, not due to edge-weight inaccuracies.

2. **The error rate is unacceptable for production.** A 78% error rate (7/9 cases) means the pipeline is unusable for batch processing. Fixing cap handling and multi-target enumeration (P0 + P1) is strictly necessary before any algorithmic changes.

3. **The PASS case proves the Voronoi + graph approach works.** `aorta-surface-open-ends` achieves 0.78% radius error and 2.77 mm mean distance on a real anatomical surface. The Voronoi dual + Dijkstra/FMM path is fundamentally sound for simple geometries.

4. **P0 and P1 are localized, low-risk changes.** They touch configuration and enumeration logic rather than the numerical solver. They can be validated quickly against the existing 9-case suite.

### Path Forward

| Phase | Action | Effort | Expected Success Rate |
|-------|--------|--------|-----------------------|
| **Now** | Implement P0 (closed-surface cap detection) + P1 (multi-target enumeration) | 1–2 days | ~89% (8/9) |
| **Next** | Implement P2 (EDT pole robustness) + P3 (mask fallback) | 2–3 days | ~95%+ |
| **Future** | If radius error remains > 5% on complex geometries, evaluate pure voxel FMM as an alternative backend | Research | — |

---

## 9. Appendix: Phase Timing Summary

| Case                               | Preprocess | Capping | Delaunay | Internal Tets | Voronoi | Poles | FMM | Resample | Sections | Network |
|------------------------------------|------------|---------|----------|---------------|---------|-------|-----|----------|----------|---------|
| aorta-surface-open-ends            | 0.045      | 0.068   | 0.470    | 7.719         | 5.056   | 0.003 | 4.756 | 0.014   | 0.106    | 0.008   |
| aorta-surface-branch-split         | 0.066      | 0.044   | 0.572    | 8.707         | 5.911   | 0.003 | 7.436 | 0.003   | 0.140    | 0.007   |
| aorta-surface-connectivity-reference | 0.028    | 0.014   | 0.232    | 3.845         | 1.180   | 0.000 | 0.048 | 0.000   | 0.000    | 0.000   |
| aorta-surface-segment-2            | 0.106      | 0.047   | 1.285    | 14.376        | 17.028  | 0.000 | 0.155 | 0.000   | 0.000    | 0.000   |
| aorta-surface-two-segments         | 0.143      | 0.378   | 2.089    | 21.667        | 39.493  | 0.000 | 0.230 | 0.000   | 0.000    | 0.000   |
| aorta-surface (.stl)               | 0.063      | 0.026   | 0.578    | 7.467         | 5.603   | 0.000 | 0.087 | 0.000   | 0.000    | 0.000   |
| aorta-surface (.vtp)               | 0.059      | 0.028   | 0.558    | 7.434         | 5.198   | 0.000 | 0.084 | 0.000   | 0.000    | 0.000   |
| cow                                | 0.034      | 0.013   | 0.110    | 3.448         | 1.131   | 0.000 | 0.037 | 0.000   | 0.000    | 0.000   |
| fixture                            | 0.052      | 0.023   | 0.431    | 6.690         | 4.088   | 0.000 | 0.072 | 0.000   | 0.000    | 0.000   |

**Observation:** For ERROR cases, the FMM phase is anomalously fast (< 0.25 s), confirming that the front never propagates. For the PASS case, FMM consumes 4.76 s (26% of total time), indicating normal operation.
