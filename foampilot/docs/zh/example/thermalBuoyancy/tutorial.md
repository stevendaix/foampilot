# 热浮力（buoyantSimpleFoam）— FoamPilot 教程

## 概述

本教程使用 `buoyantSimpleFoam` 和 **Boussinesq 近似** 在受热房间中模拟 **自然对流**。它演示了在重力作用下流体流动与传热的耦合。

FoamPilot 自动化配置：

- 重力激活和 Boussinesq 浮力
- 等温壁面补丁
- 能量方程配置

📁 **位置**: `foampilot/tutorials/08_thermalBuoyancy/`

---

## 1. 前提条件

- 已安装 OpenFOAM
- 已安装 FoamPilot

---

## 2. 案例物理

- 域：房间（4 m × 4 m × 3 m）
- 流体：空气（不可压缩 Boussinesq）
- 热墙：350 K（左壁）
- 冷墙：300 K（右壁）
- 其他壁面：绝热（零梯度）
- 重力：9.81 m/s²（-Z 方向）

### 2.1 Boussinesq 近似

密度变化建模为：

$$
\rho = \rho_0 [1 - \beta (T - T_0)]
$$

其中：
- `ρ₀` — 参考密度
- `β` — 热膨胀系数
- `T₀` — 参考温度

动量方程中的浮力项：

$$
\frac{\partial (rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p_{rgh} + \nabla \cdot (\mu_{eff} \nabla \mathbf{u}) + \rho \mathbf{g}
$$

### 2.2 Rayleigh 数

$$
Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}
$$

对于本例（ΔT = 50 K，L = 4 m）：

$$
Ra = \frac{9.81 \times 3.2 \times 10^{-3} \times 50 \times 4^3}{1.5 \times 10^{-5} \times 2.2 \times 10^{-5}} \approx 9.7 \times 10^9
$$

这处于 **湍流自然对流** 区域（Ra > 1e9），确认需要湍流模型（k-epsilon）。

### 2.3 控制方程

能量方程：

$$
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T
$$

压力（经水静力修正）：

$$
p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}
$$

---

## 3. 工作流程

### 3.1 求解器初始化

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = True
solver.turbulence_model = "kEpsilon"
```

设置 `solver.with_gravity = True` 将启用：

- `buoyantSimpleFoam` solver
- 动量方程中的 Boussinesq 密度
- `p_rgh` 压力变量

### 3.2 边界条件

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(0.1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)

# 热墙 350 K
solver.boundary.set_raw_condition("hotWall", "T", {"type": "fixedValue", "value": "350"})
# 冷墙 300 K
solver.boundary.set_raw_condition("coldWall", "T", {"type": "fixedValue", "value": "300"})
```

FoamPilot 的 `set_raw_condition` 允许针对复杂情况直接指定 OpenFOAM 字典。

### 3.3 运行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

### 4.1 自然对流环流单元

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Thermal Buoyancy Report",
    author="FoamPilot",
)
report.add_statistic("Ra", 9.7e9, "", "Rayleigh number")
report.add_statistic("T_hot", 350.0, "K", "Hot wall temperature")
report.add_statistic("T_cold", 300.0, "K", "Cold wall temperature")
report.save_html_report(filename="buoyancy_report.html")
```

### 4.2 LaTeX 报告

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Natural Convection in a Heated Room",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Boussinesq buoyancy simulation with buoyantSimpleFoam.")
doc.add_section("Governing Equations", "")
doc.add_equation(
    r"Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}",
    caption="Rayleigh number",
)
doc.add_equation(
    r"p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}",
    caption="Modified pressure",
)
doc.add_section("Boundary Conditions", "")
doc.add_table(
    [["hotWall", "350", "K"], ["coldWall", "300", "K"], ["Other walls", "adiabatic", ""]],
    headers=["Patch", "Temperature", "Condition"],
    caption="Wall boundary conditions",
)
doc.generate_document(output_format="pdf")
```

### 4.3 Typst 科学文档

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Natural Convection", "FoamPilot")
doc.add_section("Introduction", "Buoyancy-driven flow analysis.")
doc.add_equation(
    r"Ra = g \beta \Delta T L^3 / (\nu \alpha)",
    caption="Rayleigh number",
    label="eq:rayleigh",
)
doc.add_table(
    [["T_hot", "350 K"], ["T_cold", "300 K"], ["g", "9.81 m/s²"]],
    headers=["Parameter", "Value"],
    caption="Simulation parameters",
)
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 5. 预期结果

| 数量 | 预期 |
|----------|----------|
| 自然对流单元 | 2–4 个环流单元 |
| 热空气上升速度 | 约 0.1–0.3 m/s |
| 中平面温度分布 | 从 350 K 到 300 K 线性变化 |
| 热墙附近速度 | 向上（0.05–0.15 m/s） |

---

## 6. 执行

```bash
cd foampilot/tutorials/08_thermalBuoyancy
python run.py
python report_generator.py
```
