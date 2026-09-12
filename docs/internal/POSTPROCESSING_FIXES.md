# Post-Processing Fixes Documentation

## Problem Summary

Two plots in the buildingAero tutorial were broken:
- **slice_plot.png** — rendered all black
- **vector_plot.png** — arrows invisible (factor too small, all white)

Root causes were identified and fixed through systematic debugging.

## Root Causes & Fixes

### 1. Off-screen Rendering: Missing `render()` Before `screenshot()`

**File:** `src/foampilot/postprocess/openfoam_pyvista.py`

**Problem:** `export_plot()` and `plot_slice()` called `screenshot()` without `render()` first.
In VTK off-screen mode, the render window must be explicitly rendered before capture.

**Fix:** Added `plotter.render()` before `plotter.screenshot()` in `export_plot()`.
Kept the existing `render()` call in `plot_slice()`.

### 2. VTK Off-Screen Transparency Bug

**File:** `src/foampilot/postprocess/openfoam_pyvista.py`

**Problem:** Any mesh with `opacity` in the range (0, 1) renders as **all-black** in VTK off-screen mode.
This was verified systematically:

| Opacity | Result |
|---------|--------|
| 1.0     | Works (full colors) |
| 0.9     | All black |
| 0.5     | All black |
| 0.1     | All black |
| 0.01    | Works (mostly background) |

**Fix:** Replaced all `opacity=<value>` mesh rendering with `style="wireframe"` or `style="surface"` (fully opaque).

- `plot_slice`: boundaries changed from `opacity=0.5` → `style="wireframe", color="black"`
- Vector plots: removed `opacity=0.1` cell mesh background, replaced with `style="wireframe", color="lightgray"`

### 3. `show_bounds()` Interferes with Thin 2D Slices

**File:** `src/foampilot/postprocess/openfoam_pyvista.py`

**Problem:** `show_bounds()` combined with `reset_camera()` causes thin 2D slice planes to render as all-black.
This appears to be a VTK camera frustum issue where the thin slice (zero or near-zero thickness) gets clipped.

**Fix:** Removed `show_bounds()`. Added `pl.camera_position = "xy"` (for z-slices) to provide
orthogonal top-down view showing the full domain without the cut-in-half appearance.

```python
view_map = {"x": "yz", "y": "xz", "z": "xy"}
pl.camera_position = view_map.get(plane, "xy")
```

### 4. Vector Glyph Factor Too Small

**Files:** All tutorial/example `run.py` files with vector plots

**Problem:** `factor=0.0003` on a 200m domain produces arrows of 0.06m — invisible.

**Fix:** `glyph_factor = domain_length * 0.002`
- 200m domain → factor = 0.4
- At 10 m/s velocity → arrow length ≈ 4m (visible)

### 5. No Arrow Subsampling

**Problem:** Glyphing creates one arrow per mesh point (~151K arrows), producing visual noise.

**Fix:** Subsample to ≤2000 cells using `extract_cells()` before glyphing:
```python
n_cells = cell_mesh.n_cells
max_glyphs = 2000
step = max(1, n_cells // max_glyphs)
cell_mesh_reduced = cell_mesh.extract_cells(np.arange(0, n_cells, step))
```

### 6. Wrong PyVista API Parameter Name

**Problem:** `clamp=True` — `clamp` is not a valid `glyph()` parameter in PyVista 0.46.

**Fix:** `clamping=True` — the correct parameter name.

### 7. PyVista 0.46 API Changes

**Problem:** `pv.UniformGrid` renamed to `pv.ImageData`.
`point_data_to_cell_data([field])` deprecated — field names no longer accepted as positional args.

**Fix:** Use `pv.ImageData(...)` and `point_data_to_cell_data(pass_point_data=False)`.

## Test Coverage

**File:** `test/test_postprocess_cube.py`

15 tests using a synthetic 10×10×10 cube mesh with synthetic U (velocity) and p (pressure) fields:

| Test | What It Verifies |
|------|-----------------|
| `test_export_plot_writes_png` | PNG file is created with non-zero size |
| `test_export_plot_renders_nonblank` | Image has >10 unique colors (not all black/white) |
| `test_plot_slice_offscreen_writes_png` | Off-screen slice PNG is written |
| `test_plot_slice_renders_nonblank` | Slice image has >10 unique colors |
| `test_plot_slice_inline_mode` | Non-off-screen mode returns a plotter |
| `test_vector_glyph_scales_with_domain` | Glyph factor is domain-adaptive |
| `test_vector_glyph_subsample` | Subsampling reduces cell count for large meshes |
| `test_get_mesh_statistics` | Returns correct point/cell counts and bounds |
| `test_get_region_statistics_scalar` | Computes mean/min/max/std for scalar fields |
| `test_get_region_statistics_vector` | Handles vector fields |
| `test_export_region_data_to_csv` | CSV has X, Y, Z, U_0, U_1, U_2, p columns |
| `test_export_statistics_to_json` | JSON serializes numpy types correctly |
| `test_calculate_q_criterion` | Adds `q_criterion` point data |
| `test_calculate_vorticity` | Adds `vorticity` point data (shape: n_points × 3) |
| `test_numpy_encoder` | Serializes np.int64, np.float64, np.ndarray |

## Files Modified

| File | Changes |
|------|---------|
| `src/foampilot/postprocess/openfoam_pyvista.py` | `plot_slice`: background, wireframe boundaries, camera position, render, return. `export_plot`: render before screenshot. `point_data_to_cell_data`: API fix |
| `tutorials/06_buildingAero/run.py` | Domain-adaptive glyph factor, subsampling, background, no transparency |
| `tutorials/07_motorBike/run.py` | Same vector plot fixes |
| `examples/building_aero/post.py` | Same vector plot fixes |
| `examples/building_aero/with_run.py` | Same vector plot fixes |
| `examples/muffler/run_simu.py` | Same vector plot fixes |
| `test/test_postprocess_cube.py` | New test file (15 tests) |
