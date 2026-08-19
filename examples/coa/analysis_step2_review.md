# Step 2 Review: VMTK Continuous Tracing Implementation

**Date:** 2026-08-19  
**Goal:** Verify Step 2 changes and explain why geometric accuracy did not improve.  
**Benchmark:** Hausdorff 23.848 mm (no change), Length error 65.71% (no change), Points 81 (no change), fast_marching time 8.123 s (increased from 3.6 s).

---

## 1. Code Review

### 1.1 `_trace_centerline_steepest_descent` — Is it correct?

**Short answer: No. The function is fundamentally flawed and contains at least one fatal runtime bug.**

#### Fatal Bug: `NameError` on `python_fmm` backend

In `vmtkfastmarching_local.py` lines 558–559:

```python
if self.Backend in ("python_eikonal", "python_fmm") and len(path) >= 2:
    result = _trace_centerline_steepest_descent(dists, predecessor, graph.vertices, graph.radii, graph.edges, src_vor, tgt_vor, step_size=0.2)
```

The variables `dists` and `predecessor` are **only defined** when `self.Backend == "dijkstra"` or `self.Backend == "python_eikonal"` (lines 541–548). When `self.Backend == "python_fmm"`, the code executes:

```python
elif self.Backend == "python_fmm":
    path = _true_fmm_backend(graph.vertices, graph.radii, graph.edges, src_vor, tgt_vor, self.RadiusFloor)
```

Neither `dists` nor `predecessor` is assigned. The subsequent call to `_trace_centerline_steepest_descent` raises `NameError: name 'dists' is not defined`. This means **the `python_fmm` backend is completely broken** and will crash on any path with `len(path) >= 2`.

#### Dead Code: `_python_eikonal_backend`

Lines 238–286 define `_python_eikonal_backend`, a function that attempts a custom eikonal relaxation solver. However, this function is **never called**. The `python_eikonal` branch in `Execute()` uses `scipy.sparse.csgraph.dijkstra` instead (line 547). The dead function adds maintenance burden without contributing to the pipeline.

#### Design Flaw: Constrained to graph edges

The function claims to perform "steepest descent" but it is **geometrically constrained to Voronoi edges**:

```python
for nb in neighbors:          # only direct graph neighbors
    p0 = vertices[current]
    p1 = vertices[nb]
    ...
    n_samples = max(2, int(np.ceil(d / step_size)))
    for s in range(1, n_samples + 1):
        alpha = s / (n_samples + 1)
        p = p0 + alpha * (p1 - p0)   # interpolation along the SAME edge
```

The function samples points **only along edges that already exist in the Voronoi graph**. It cannot explore regions between edges, cannot cross Voronoi faces, and cannot discover shorter paths that cut through the interior of Voronoi polygons. This is not "continuous tracing" in the geometric sense — it is **denser sampling along the same graph**.

#### Design Flaw: Greedy local search with no gradient-field following

True steepest descent follows the gradient of the eikonal field continuously. This function uses a greedy 1-step lookahead:

```python
best_grad = 0.0
for nb in neighbors:
    ...
    grad = -(t_interp - t0) / max(d_to_p, 1e-12)
    if grad > best_grad:
        best_grad = grad
        best_point = p
```

It picks the single neighbor edge with the steepest local gradient and jumps to the vertex at the other end. It does **not** use the `predecessor` array (which is passed but never referenced) and does not follow the true gradient field. In graph-based shortest paths, the predecessor array encodes the gradient direction; ignoring it means the function is not actually descending the eikonal field — it is performing a heuristic neighbor search on discrete distance values.

#### Design Flaw: Early termination flattens near source

```python
if best_grad < 1e-9:
    break
```

Near the source vertex, the distance field is nearly flat (all distances cluster near zero). The gradient `-(t_interp - t0) / d_to_p` becomes extremely small. This causes the function to break out of the loop **before reaching the source**. The final check then rejects the result:

```python
if current != source:
    return None
```

The function returns `None`, and the caller falls back to the discrete Dijkstra path. This is a **silent failure**: the function runs, spends CPU time, but contributes nothing.

#### Design Flaw: Unused `predecessor` parameter

The function signature accepts `predecessor`:

```python
def _trace_centerline_steepest_descent(
    dist: np.ndarray,
    predecessor: np.ndarray,
    vertices: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray,
    source: int,
    target: int,
    step_size: float = 0.2,
) -> Optional[np.ndarray]:
```

But `predecessor` is **never referenced** in the function body. This is a clear sign the function was written without a proper understanding of how steepest descent should use the shortest-path tree.

### 1.2 `Execute()` — Does it actually improve the path?

**Short answer: No. For `python_eikonal` it adds interpolated points along the same graph path. For `python_fmm` it crashes.**

#### For `python_eikonal`

1. Dijkstra computes the exact shortest path on the subdivided Voronoi graph.
2. `_trace_centerline_steepest_descent` receives this path and the distance field.
3. Because the function is constrained to graph edges, it can only interpolate between vertices that Dijkstra already visited.
4. The interpolated points are added **along the same edges** that Dijkstra traversed. Geometrically, the path is identical — it just has more points.
5. The function frequently returns `None` (due to the `best_grad < 1e-9` early termination near the source), causing a silent fallback to the discrete path.
6. **Net effect:** More points, same geometry. No accuracy improvement.

#### For `python_fmm`

1. `_true_fmm_backend` computes a path on the graph.
2. The code then tries to call `_trace_centerline_steepest_descent(dists, predecessor, ...)`.
3. `dists` and `predecessor` are undefined → **`NameError` crash**.

---

## 2. Performance Analysis

### Why did fast_marching time increase from 3.6 s to 8.123 s?

The time increase is caused by `_trace_centerline_steepest_descent`. For each vertex in the Dijkstra path:

1. **Neighbor iteration:** The function loops over all graph neighbors of the current vertex.
2. **Edge sampling:** For each neighbor, it computes `n_samples = max(2, int(np.ceil(d / step_size)))` where `d` is the edge length and `step_size = 0.2` mm. On the subdivided graph (target length 0.5 mm), many edges require 2–3 samples.
3. **Distance computations:** For each sample, it computes `np.linalg.norm(p - vertices[current])` and a gradient value.
4. **Vertex reassignment:** After finding the best interpolated point, it computes `d_to_current = np.array([np.linalg.norm(vertices[nb] - best_point) for nb in neighbors])` and takes an `argmin`. This is an O(degree) operation per step.

On a path of ~80 vertices with average degree ~4–6, the function performs thousands of unnecessary distance computations. None of this work improves the path geometry, so the time is **pure overhead**.

---

## 3. Root Cause Analysis: Why Hasn't Accuracy Improved?

### 3.1 Dijkstra already finds the optimal path on the graph

The Voronoi graph is a discrete weighted graph. Dijkstra's algorithm computes the **exact shortest path** on this graph. The path returned by Dijkstra is provably optimal among all paths that follow graph edges.

Fine interpolation along graph edges cannot discover a shorter path because:
- Any interpolated point lies on a straight-line segment between two graph vertices.
- The geometric length of the interpolated path is exactly the sum of the edge lengths that Dijkstra already traversed.
- You cannot "cut corners" by interpolating along existing edges.

### 3.2 The graph itself is the bottleneck

The accuracy limitation is not the tracing resolution — it is the **graph representation**. The Voronoi diagram in the current implementation is a 1D edge graph:

- **Vertices** = circumcenters of internal tetrahedra.
- **Edges** = adjacency between tetrahedra sharing a face.

This is a skeleton of the Voronoi diagram. The true Voronoi diagram is a **2D polygonal non-manifold** (faces + edges + vertices). VMTK's fast marching method solves the eikonal equation on this 2D manifold, allowing the centerline to pass through Voronoi faces, not just along edges.

Because our implementation reduces the Voronoi to a 1D graph:
- The pathfinder cannot cross Voronoi faces; it can only hop from vertex to vertex.
- At vessel bends, the graph approximates the curved medial axis with piecewise-linear segments.
- The shortest path on this graph systematically underestimates the true eikonal distance along curved segments.

### 3.3 The distance field is graph-based, not voxel-based

The current `python_eikonal` backend computes distances using Dijkstra on the Voronoi graph. The `python_fmm` backend computes distances using a heap-based FMM, but **still on the Voronoi graph** (not on a voxel grid).

VMTK's true fast marching method solves the eikonal equation on the **voxel grid** (or on the Voronoi polygonal mesh with angle-aware quadratic updates). This produces a continuous distance field that captures the true geodesic distance through the vessel. The centerline is then traced by following the gradient of this field.

Our implementation never computes a continuous distance field. The "distance" values are discrete shortest-path distances on a graph. Interpolating these discrete values along edges does not create a continuous field — it merely densifies the discrete path.

### 3.4 The `_true_fmm_backend` is also graph-constrained

Even if the `python_fmm` backend worked correctly, `_true_fmm_backend` (lines 289–398) computes FMM on the **same Voronoi graph**. It uses a min-heap narrow band and a quadratic update formula that accounts for edge angles, but it operates on graph edges only. It does not solve the eikonal equation on a 2D manifold or voxel grid.

The quadratic update in `_true_fmm_backend`:
```python
a = d1**2 + d2**2 - 2.0 * d1 * d2 * cos_theta
b = 2.0 * d2 * u * (d1 * cos_theta - d2)
c = d2**2 * (u**2 - (1.0 / F**2) * d1**2 * (1.0 - cos_theta**2))
```

This is a correct 1D quadratic solver for the eikonal equation along a 1D graph path. It improves the distance accuracy **on the graph**, but it cannot improve the graph topology itself. The path is still confined to graph edges.

### 3.5 Silent fallback masks the failure

When `_trace_centerline_steepest_descent` returns `None`, the code falls back to:

```python
pts = graph.vertices[path]
rads = graph.radii[path]
```

This means the function's failure is invisible to the user. The benchmark reports the same metrics as Step 1, with no indication that the "continuous tracing" stage failed silently.

---

## 4. Summary of Bugs and Issues

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 1 | `Execute()` line 558–559 | `NameError`: `dists` and `predecessor` undefined for `python_fmm` backend | **Fatal** |
| 2 | `vmtkfastmarching_local.py` lines 238–286 | `_python_eikonal_backend` is dead code; never called | Low |
| 3 | `_trace_centerline_steepest_descent` | Constrained to graph edges; cannot find shorter geometric paths | **Critical** |
| 4 | `_trace_centerline_steepest_descent` | Greedy 1-step lookahead; does not follow true gradient field | **Critical** |
| 5 | `_trace_centerline_steepest_descent` | Early termination `best_grad < 1e-9` near source causes silent fallback | **High** |
| 6 | `_trace_centerline_steepest_descent` | `predecessor` parameter passed but never used | Medium |
| 7 | `vmtkfastmarching_local.py` | Distance field computed on Voronoi graph, not voxel grid or 2D manifold | **Root cause** |

---

## 5. Next Steps

### Immediate Fixes (do first)

1. **Fix the `python_fmm` NameError.** Either:
   - Compute `dists` and `predecessor` inside the `python_fmm` branch and pass them to the tracer, OR
   - Refactor `_true_fmm_backend` to return `(path, dists, predecessor)` so the tracer has the inputs it needs.

2. **Remove or fix dead code.** Delete `_python_eikonal_backend` or wire it into the `python_eikonal` branch. The current `python_eikonal` backend is just Dijkstra, which makes the naming misleading.

3. **Fix the silent fallback.** If `_trace_centerline_steepest_descent` returns `None`, log a warning so the user knows the continuous tracing stage failed.

### Fundamental Fixes (required for accuracy improvement)

4. **Implement true continuous tracing on the Voronoi manifold.** To match VMTK's accuracy, the tracer must:
   - Operate on Voronoi polygon faces (2D manifold), not just the 1D edge graph.
   - Follow the gradient of the eikonal field continuously, landing at arbitrary parametric coordinates on polygon edges.
   - Use the predecessor array or a proper gradient field to guide descent.

5. **Compute the eikonal distance field on the voxel grid or polygonal mesh.** The graph-based distance is inherently limited. The true improvement comes from solving the eikonal equation on the continuous domain (voxel grid or Voronoi polys) and tracing through that field.

6. **Improve the Voronoi graph topology.** If building full Voronoi polys is too complex for now, at minimum:
   - Add more edges to the graph (e.g., connect non-adjacent Voronoi vertices that are spatially close).
   - Use a finer initial tetrahedral mesh to produce a denser Voronoi diagram.
   - This gives the pathfinder more routes to choose from, reducing the "grid approximation" error.

### What NOT to do

- **Do not increase the subdivision resolution or sampling density** as a substitute for fixing the graph topology. As this review shows, finer sampling along existing edges does not improve the path because Dijkstra already found the optimal path on that graph.
- **Do not add more iterations to the relaxation solver.** The `python_eikonal` backend uses Dijkstra, which is exact on the graph. More iterations would not change the result.

---

## 6. Expected Outcome of Next Steps

If the immediate fixes (items 1–3) are applied, the pipeline will be more robust and debuggable, but **accuracy will still not improve** because the fundamental limitation is the graph-based distance field.

To achieve measurable accuracy improvement (Hausdorff < 10 mm, length error < 20 %), the pipeline must implement one of:

- **Option A:** True fast marching on the voxel grid + continuous gradient tracing through voxels. This is the most accurate approach but requires significant implementation effort.
- **Option B:** True fast marching on the Voronoi polygonal manifold (with polys + angle-aware quadratic updates) + continuous steepest descent on polygon edges. This matches VMTK's algorithm most closely.
- **Option C:** A denser Voronoi graph (finer tetrahedral mesh + more edges) combined with graph-based shortest path. This is a pragmatic compromise that reduces the graph approximation error without requiring a full 2D manifold implementation.
