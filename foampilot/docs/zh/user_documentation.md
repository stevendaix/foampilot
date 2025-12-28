# `foampilot` 用户文档

**作者：** Manus AI  
**日期：** 2025年12月4日

## 1. `foampilot` 的总体工作理念

`foampilot` 模块设计为 **OpenFOAM** 的 Python 封装，旨在简化和自动化计算流体力学（CFD）模拟过程。它抽象了 OpenFOAM 文件结构和命令的复杂性，使用户能够完全通过 Python 定义、运行和后处理模拟。

`foampilot` 的设计理念如下：

1. **Python 中定义案例：** 用户通过 Python 对象（如 `Solver`、`Meshing`、`Boundary`、`Constant`、`System`）与 OpenFOAM 交互，而不是手动修改配置文件。
2. **自动生成文件：** Python 对象负责在案例目录中自动生成 OpenFOAM 配置文件（如 `controlDict`、`fvSchemes`、`transportProperties` 等）。
3. **Python 生态集成：** 模块与强大的 Python 库集成，以完成特定任务：
   * **`classy_blocks`**：用于生成结构化网格（`blockMesh`）。
   * **`pyfluid`**：管理流体属性和物理常数。
   * **`pyvista`**：高级后处理和可视化。
   * **`latex_pdf`**：生成结构化报告。

总之，`foampilot` 将手动且分散的工作流程（编辑文件、执行 shell 命令）转化为可重复的 Python 脚本工作流程。

## 2. 几何与网格选择

网格方法的选择在 CFD 中至关重要，取决于几何复杂性。`foampilot` 提供三种主要方案：

| 网格方法 | 目标几何 | `foampilot` / 库 | 描述与优势 |
| :--- | :--- | :--- | :--- |
| **`blockMesh`** | 简单几何、拉伸几何或由六面体块组成的几何 | `Meshing(..., mesher="blockMesh")`（通过 `classy_blocks`） | 适合简单几何（通道、圆柱等）或规则计算域。提供对网格质量和单元分布的完全控制。`
| **`gmsh`** | STEP 或 IGES 格式的复杂几何 | `Meshing(..., mesher="gmsh")` | 可生成复杂 CAD 几何的非结构化网格（四面体、棱柱）。需要导入几何文件（如 `.step`）。 |
| **`snappyHexMesh`** | STL 格式的复杂几何（三角化表面） | `Meshing(..., mesher="snappy")` | 适用于非常复杂几何（如车辆、建筑）。生成符合 STL 表面的六面体网格，并自动加密边界层。 |

### 2.1 使用 `blockMesh` 生成结构化网格（通过 `classy_blocks`）

步骤：
1. 定义基础几何形状（如 `cb.Cylinder`、`cb.ExtrudedRing`、`cb.Elbow`）。
2. 使用链式方法（`.chain()`、`.expand()`、`.fill()`）构建复杂几何。
3. 对每个形状定义网格（`.chop_axial()`、`.chop_radial()`、`.chop_tangential()`）。
4. 分配补丁（表面）（`.set_start_patch()`、`.set_end_patch()`）。
5. 将所有对象组合到 `cb.Mesh()` 并写出 `blockMeshDict`。

```python
mesh = cb.Mesh()
# ... 添加形状 ...
mesh.set_default_patch("walls", "wall")
mesh.write(current_path / "system" / "blockMeshDict", current_path /"debug.vtk")
```

### 2.2 使用 `gmsh` 生成非结构化网格（STEP）

1. 确保 STEP 文件可用（如 `geometry.step`）。
2. 初始化 `Meshing` 对象并指定 `mesher="gmsh"`。
3. 执行网格生成：

```python
mesh_obj = Meshing(current_path, mesher="gmsh")
mesh_obj.mesher.run(current_path / "geometry.step")
```

### 2.3 使用 `snappyHexMesh` 生成 STL 网格

1. 创建基本 `blockMeshDict` 来包围 STL 几何。
2. 确保 STL 文件在 `constant/triSurface`。
3. 初始化 `Meshing` 对象并指定 `mesher="snappyHexMesh"`。
4. 执行网格生成：

```python
mesh_obj = Meshing(current_path, mesher="snappyHexMesh")
mesh_obj.mesher.run()
```

*注意：详细的 `snappyHexMeshDict` 配置需用户自行设置，或使用 `foampilot` 高级函数。*

## 3. 求解器选择与物理模型

通过配置 `Solver` 对象，`foampilot` 根据物理属性自动选择 OpenFOAM 求解器。

### 3.1 求解器选择

```python
from foampilot.solver import Solver
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False
```

根据设置，`foampilot` 配置 `controlDict` 和相关字典，例如 `simpleFoam`、`pimpleFoam` 或 `rhoSimpleFoam`。

### 3.2 边界条件设置

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(pattern="inlet", condition_type="velocityInlet", velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)
solver.boundary.apply_condition_with_wildcard(pattern="outlet", condition_type="pressureOutlet")
solver.boundary.apply_condition_with_wildcard(pattern="walls", condition_type="wall")
solver.boundary.write_boundary_conditions()
```

### 3.3 修改字典或添加补丁

```python
solver.constant.transportProperties.nu = Quantity(1e-6, "m2/s")
solver.system.controlDict.writeInterval = 100
solver.system.write()
solver.constant.write()
```

## 4. 使用 `pyfluid` 设置 `system` 和 `constant`

```python
from foampilot.utilities.fluids_theory import FluidMechanics
fluid_mech = FluidMechanics(FluidMechanics.get_available_fluids()['Water'], temperature=Quantity(293.15, "K"), pressure=Quantity(101325, "Pa"))
properties = fluid_mech.get_fluid_properties()
solver.constant.transportProperties.nu = properties['kinematic_viscosity']
solver.system.write()
solver.constant.write()
```

## 5. 求解器运行

### 5.1 顺序运行

```python
solver.run_simulation()
```

### 5.2 并行运行

```python
solver.decompose_domain(cores=4)
solver.run_simulation(parallel=True)
solver.reconstruct_domain()
```

## 6. 使用 `pyvista` 后处理

```python
from foampilot import postprocess
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()
time_steps = foam_post.get_all_time_steps()
structure = foam_post.load_time_step(time_steps[-1])
pl_contour = pv.Plotter(off_screen=True)
pl_contour.add_mesh(structure["cell"], scalars='p', show_scalar_bar=True)
foam_post.export_plot(pl_contour, current_path / "contour_plot.png")
```


## 7. 使用 `latex_pdf` 生成报告

```python
doc = latex_pdf.LatexDocument(title="Simulation Report", author="Automated Report", filename="simulation_report", output_dir=current_path)
doc.add_table([["Statistic", "Value"]], headers=["Statistic", "Value"], caption="Mesh Quality Statistics")
doc.generate_document(output_format="pdf")
```
