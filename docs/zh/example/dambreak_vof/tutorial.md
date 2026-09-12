# 决堤 VOF — FoamPilot 教程

## 概述

本教程使用 **VOF (Volume of Fluid)** 模型和 `interFoam` 求解器，在一个二维矩形域中模拟水柱的**晃动（sloshing）**。

FoamPilot 自动化：

- VOF 相分数 (`alpha.water`) 的设置
- 两相材料属性
- 重力激活

📁 **位置**: `foampilot/tutorials/04_damBreak_multiphase/`

---

## 1. 先决条件

- 已安装 OpenFOAM
- 已安装 FoamPilot

---

## 2. 算例物理

- **域**：二维矩形水箱 (5 m × 2 m × 0.1 m)
- **相**：水（alpha = 1）和空气（alpha = 0）
- **VOF 模型**：用于界面追踪的 Volume of Fluid
- **重力**：激活（9.81 m/s²，-Y 方向）
- **湍流**：层流（低 Re）

### 2.1 VOF 传输方程

$$
\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \, \alpha) = 0
$$

相分数 `α` 在水中为 1，在空气中为 0，在界面处为 0–1。

### 2.2 动量方程

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} + \sigma \kappa \nabla \alpha
$$

其中：
- `σ` — 表面张力系数
- `κ` — 界面曲率
- `g` — 重力矢量

### 2.3 初始条件

- 水柱：位于域左侧，尺寸 2 m × 1 m
- 其余域：充满空气
- 初始时速度处处为零

---

## 3. 工作流程

### 3.1 求解器初始化

```python
from foampilot.solver import Solver
from foampilot import ValueWithUnit

solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.is_vof = True
solver.turbulence_model = "laminar"
```

设置 `solver.is_vof = True` 会自动：

- 启用 `interFoam` 求解器
- 配置两相 `transportProperties`
- 创建 `alpha.water` 场

### 3.2 边界条件

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)
```

### 3.3 执行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

### 4.1 界面追踪

VOF `alpha.water` 场用于追踪水-空气界面：

```python
from foampilot import postprocess

foam_post = postprocess.FoamPostProcessing(case_path=case_path)
foam_post.foamToVTK()
```

### 4.2 可视化

```python
import pyvista as pv

mesh = pv.read("VTK/0/cellular.vtk")
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, scalars="alpha.water", cmap="Blues")
plotter.screenshot("dam_break_interface.png")
```

### 4.3 报告生成

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="DamBreak VOF Simulation Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Two-phase flow simulation using VOF model.")
doc.add_equation(r"\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \alpha) = 0")
doc.add_section("Interface Evolution")
doc.add_figure("dam_break_interface.png", "Water-air interface at t=2.0s")
doc.generate_document(output_format="pdf")
```

---

## 5. 预期结果

| 物理量 | 预期值 |
|--------|--------|
| 水前缘速度 | ~4.4 m/s (√(2gh), h=1m) |
| 冲击右壁时间 | ~3 s |
| 波的反射 | 冲击后可见 |

---

## 6. 执行

```bash
cd foampilot/tutorials/04_damBreak_multiphase
python run.py
python report_generator.py
```
