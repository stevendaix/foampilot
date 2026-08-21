# 腔体层流 — FoamPilot 教程

## 概述

本教程演示如何使用 **FoamPilot** 和 `icoFoam` 求解器对一个上盖驱动腔体进行一次完整的**层流不可压缩流动**仿真。

FoamPilot 自动化处理：

- `blockMesh` 网格生成
- 边界条件定义
- 求解器配置（`laminar` 湍流模型，`icoFoam`）
- 残差后处理

📁 **位置**: `foampilot/tutorials/01_cavity_laminar/`

---

## 1. 前置条件

- 已安装并可访问 OpenFOAM
- 已安装 FoamPilot（`pip install -e .`）

---

## 2. 计算案例物理

- **几何**：二维方形腔体（1 m × 1 m）
- **流体**：水（不可压缩，层流）
- **边界条件**：
  - 移动上盖：`U = (1, 0, 0) m/s`
  - 固定壁面：无滑移
  - 前/后：对称

### 2.1 控制方程

连续性方程（不可压缩）：

$$
\nabla \cdot \mathbf{u} = 0
$$

Navier–Stokes（层流）：

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
$$

### 2.2 无量纲参数

基于上盖速度和腔体高度的雷诺数：

$$
Re = \frac{U L}{\nu} = \frac{1 \cdot 1}{1 \times 10^{-6}} = 10^6
$$

对于层流情形，`nu = 1e-6 m²/s` 给出 `Re ≈ 100`（标准腔体教程）。

---

## 3. 工作流程

### 3.1 求解器初始化

```python
from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit

case_path = Path.cwd()

solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "laminar"
```

### 3.2 system 和 constant 字典

```python
solver.system.write()
solver.constant.write()
```

FoamPilot 自动生成：

- `system/controlDict`
- `system/fvSchemes`
- `system/fvSolution`
- `constant/transportProperties`
- `constant/turbulenceProperties`

### 3.3 边界条件

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="movingWall",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="fixedWalls",
    condition_type="wall",
)
solver.boundary.apply_condition_with_wildcard(
    pattern="frontAndBack",
    condition_type="symmetry",
)
solver.boundary.write_boundary_conditions()
```

FoamPilot 的 `apply_condition_with_wildcard` 使用正则表达式模式匹配，根据 patch 名称分配边界条件。这映射为：

- `movingWall` → U 使用 `fixedValue`，p 使用 `zeroGradient`
- `fixedWalls` → U 使用 `noSlip`，p 使用 `zeroGradient`
- `frontAndBack` → 两者均为 `symmetry`

### 3.4 仿真执行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

FoamPilot 包含通过 `ResidualsPost` 的残差追踪：

```python
from foampilot.utilities import ResidualsPost

residuals = ResidualsPost(case_path / "log.icoFoam")
residuals.process(
    export_csv=True,
    export_json=True,
    export_png=True,
    export_html=True,
)
```

结果导出到：

- `postProcessing/residuals.csv`
- `postProcessing/residuals.json`
- `postProcessing/residuals.png`
- `postProcessing/residuals.html`

---

## 5. 报告生成

### 5.1 LaTeX 报告

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Cavity Laminar Flow Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Laminar lid-driven cavity simulation using icoFoam.")
doc.add_section("Results", "Convergence data and field statistics.")
doc.generate_document(output_format="pdf")
```

### 5.2 HTML 报告

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Cavity CHT Report",
    author="FoamPilot",
)
report.add_statistic("Re", 100, "", "Reynolds number")
report.save_html_report(filename="cavity_report.html")
```

### 5.3 Typst 报告

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Cavity Flow Analysis", "FoamPilot")
doc.add_section("Introduction", "Lid-driven cavity laminar flow.")
doc.add_equation(r"Re = \frac{UL}{\nu}", caption="Reynolds number", label="eq:re")
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 6. 预期结果

| Variable | Min | Max | Mean |
|----------|-----|-----|------|
| U_x | 0 | ~2.5 | ~1.0 |
| p | -500 | +500 | 0 |

- 腔体中心出现主涡
- 角落出现次级涡
- 在上盖出口角落附近速度达到最大

---

## 7. 运行

```bash
cd foampilot/tutorials/01_cavity_laminar
python run.py
python report_generator.py  # generates PDF/HTML/Typst reports
```
