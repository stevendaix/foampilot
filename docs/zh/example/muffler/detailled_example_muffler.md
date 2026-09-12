# 消声器 CFD 示例 – FoamPilot

## 概述

此示例展示了使用 **FoamPilot** 和 **OpenFOAM** 完整的 **CFD 工作流程**，用于模拟消声器几何中的不可压缩流动。该示例可作为 **参考**，展示 FoamPilot 的工作理念：

- 明确的物理建模（流体、单位）
- 参数化几何与结构化网格
- 边界条件的可靠管理
- 自动化模拟执行
- 高级后处理与可视化
- 自动生成 PDF 报告

📁 **位置**：`examples/muffler`

---

## 1. 前提条件

运行此示例前，请确保：

- 已正确安装并配置 OpenFOAM
- 已安装 FoamPilot
- 已安装以下 Python 依赖：
  - `classy_blocks`
  - `pyvista`
  - `numpy`
  - `pandas`

---

## 2. 初始化案例

定义工作目录并初始化 FoamPilot 求解器。

```python
from foampilot.solver import Solver
from pathlib import Path

current_path = Path.cwd() / "cas_test"
solver = Solver(current_path)

solver.compressible = False
solver.with_gravity = False
```

`Solver` 是核心对象，负责：

- OpenFOAM 字典管理
- 边界条件管理
- 模拟执行

---

## 3. 流体属性

使用 FoamPilot 的 `FluidMechanics` API 明确定义流体。

```python
from foampilot import FluidMechanics, ValueWithUnit

available_fluids = FluidMechanics.get_available_fluids()

fluid = FluidMechanics(
    available_fluids["Water"],
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)

properties = fluid.get_fluid_properties()
nu = properties["kinematic_viscosity"]
```

将运动粘度注入 OpenFOAM 配置：

```python
solver.constant.transportProperties.nu = nu
```

---

## 4. 写入 OpenFOAM 配置文件

```python
solver.system.write()
solver.constant.write()
solver.system.fvSchemes.to_dict()
```

FoamPilot 自动生成：

- `controlDict`
- `fvSchemes`
- `fvSolution`
- `transportProperties`

---

## 5. 几何定义 (ClassyBlocks)

### 5.1 几何参数

```python
pipe_radius = 0.05
muffler_radius = 0.08
ref_length = 0.1
cell_size = 0.015
```

### 5.2 构建几何

几何通过 **参数化形状序列** 构建：

1. 入口管道（圆柱）
2. 扩展环（消声器主体）
3. 填充部分
4. 90° 出口弯头

入口圆柱示例：

```python
import classy_blocks as cb

shapes = []

shapes.append(cb.Cylinder(
    [0, 0, 0],
    [3 * ref_length, 0, 0],
    [0, pipe_radius, 0]
))

shapes[-1].chop_axial(start_size=cell_size)
shapes[-1].chop_radial(start_size=cell_size)
shapes[-1].chop_tangential(start_size=cell_size)
shapes[-1].set_start_patch("inlet")
```

通过在几何级别定义 patch，保证网格与边界条件一致。

---

## 6. 网格生成

```python
mesh = cb.Mesh()
for shape in shapes:
    mesh.add(shape)

mesh.set_default_patch("walls", "wall")
mesh.write(
    current_path / "system" / "blockMeshDict",
    current_path / "debug.vtk"
)
```

使用 OpenFOAM 执行网格生成：

```python
from foampilot import Meshing

meshing = Meshing(current_path, mesher="blockMesh")
meshing.mesher.run()
```

---

## 7. 边界条件管理

FoamPilot 提供通用 API，通过模式匹配设置边界条件：

```python
solver.boundary.initialize_boundary()
```

### 7.1 入口速度

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(
        ValueWithUnit(10, "m/s"),
        ValueWithUnit(0, "m/s"),
        ValueWithUnit(0, "m/s")
    ),
    turbulence_intensity=0.05
)
```

### 7.2 出口压力

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)
```

### 7.3 壁面

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)
```

### 7.4 写入边界条件文件

```python
solver.boundary.write_boundary_conditions()
```

---

## 8. 模拟执行

```python
solver.run_simulation()
```

FoamPilot 自动管理求解器、执行和日志记录。

---

## 9. 残差后处理

```python
from foampilot.utilities import ResidualsPost

residuals = ResidualsPost(current_path / "log.incompressibleFluid")
residuals.process(
    export_csv=True,
    export_json=True,
    export_png=True,
    export_html=True
)
```

---

## 10. 结果可视化与分析

### 结果加载

```python
from foampilot import postprocess

foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()
time_steps = foam_post.get_all_time_steps()
latest_time_step = time_steps[-1]
structure = foam_post.load_time_step(latest_time_step)
cell_mesh = structure["cell"]
boundaries = structure["boundaries"]
```

### 可视化

- 切片 (slice)
- 压力等值线
- 速度矢量
- 网格线框

### 分析

- Q 判据
- 涡量
- 网格和字段统计
- CSV / JSON 数据导出
- 动画生成

```python
foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
foam_post.create_animation(
    scalars="U",
    filename=current_path / "animation.gif",
    fps=5
)
```

---

## 11. 自动生成 PDF 报告

```python
from foampilot import latex_pdf

doc = latex_pdf.LatexDocument(
    title="仿真报告：消声器",
    author="自动生成报告",
    output_dir=current_path
)

doc.add_title()
doc.add_toc()
doc.add_abstract(
    "本报告总结了消声器不可压缩流体 CFD 仿真结果。"
)

doc.generate_document(output_format="pdf")
```

报告包含：

- 流体属性
- 网格统计
- 字段统计
- 可视化图形
- 数据附录

---

## 12. 总结

该示例展示了完整的 **CFD 仿真链**：

- 参数化几何与结构化网格 (classy_blocks)
- 通过 FoamPilot 使用 OpenFOAM 进行 CFD 仿真
- 高级后处理与可视化 (pyvista)
- 自动化 PDF 报告生成

