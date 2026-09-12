# 标量输运 — FoamPilot 教程

## 概述

本教程使用 `buoyantSimpleFoam`（启用 `energy_activated`）模拟层流通道流中的被动标量输运（温度场）。

FoamPilot 自动化处理：

- 能量方程激活
- 标量边界条件
- `scalarTransportFoam` 求解器配置

📁 **位置**: `foampilot/tutorials/05_scalarTransport/`

---

## 1. 先决条件

- 已安装 OpenFOAM
- 已安装 FoamPilot

---

## 2. 案例物理

- **域**：二维通道 (1 m × 0.1 m)
- **流动**：层流、不可压缩
- **标量**：温度 T（被动标量）
- **入口温度**：300 K
- **墙面温度**：350 K（底壁加热）

### 2.1 标量输运方程

$$
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T + S_T
$$

其中：
- `α` — 热扩散率（α = ν/Pr）
- `S_T` — 源项（可选）

### 2.2 边界条件

- **入口**：固定温度 `T = 300 K`
- **出口**：零梯度 `∂T/∂n = 0`
- **墙面**：固定温度 `T = 350 K`（底部），绝热（顶部）
- **对称面**：处处零梯度

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
solver.energy_activated = True
```

设置 `energy_activated = True` 会启用：

- 在 `fvSchemes` 中启用能量方程
- 温度场 `T` 的初始化
- 如果 `with_gravity = True` 则启用浮力耦合

### 3.2 边界条件

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

### 3.3 执行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

### 4.1 温度统计

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Scalar Transport Report",
    author="FoamPilot",
)

report.add_statistic("T_inlet", 300.0, "K", "Inlet temperature")
report.add_statistic("T_wall", 350.0, "K", "Wall temperature")
report.add_statistic("Pr", 0.71, "", "Prandtl number (air)")
```

### 4.2 LaTeX 报告

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Scalar Transport Analysis",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Passive scalar transport in a laminar channel flow.")
doc.add_equation(
    r"\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T",
    caption="Scalar transport equation",
)
doc.add_section("Boundary Conditions", "")
doc.add_table(
    [["Inlet", "300", "K"], ["Wall", "350", "K"], ["Outlet", "zeroGradient", ""]],
    headers=["Patch", "Condition", "Value"],
    caption="Temperature boundary conditions",
)
doc.generate_document(output_format="pdf")
```

### 4.3 Typst 报告

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Scalar Transport", "FoamPilot")
doc.add_section("Introduction", "Passive scalar transport analysis.")
doc.add_equation(r"Pe = UL/\alpha", caption="Peclet number", label="eq:pe")
doc.add_table(
    [["Parameter", "Value"], ["Re", "100"], ["Pe", "71"]],
    caption="Flow parameters",
)
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 5. 预期结果

| 物理量 | 公式 | 预期 |
|--------|------|------|
| 体积平均温度 | $T_{bulk} = \frac{1}{L} \int_0^L T dy$ | ~325 K |
| 壁面热通量 | $q'' = -k \frac{dT}{dy}\big|_{wall}$ | ~500 W/m² |
| 出口温度分布 | — | 抛物线型剖面 |

---

## 6. 运行

```bash
cd foampilot/tutorials/05_scalarTransport
python run.py
python report_generator.py
```
