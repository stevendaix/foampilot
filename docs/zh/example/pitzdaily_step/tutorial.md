# PitzDaily 后向台阶 — FoamPilot 教程

## 概述

本教程使用 **FoamPilot** 和 `simpleFoam`（k-omega SST）模拟**流经后向台阶的湍流**。该算例用于验证回流区和再附着长度。

📁 **位置**: `foampilot/tutorials/03_pitzDaily_step/`

---

## 1. 先决条件

- 已安装 OpenFOAM
- 已安装 FoamPilot

---

## 2. 算例物理

- **几何**：二维通道，带后向台阶（台阶高度 H = 0.012 m）
- **流动**：不可压、湍流、稳态
- **入口速度**：1 m/s
- **湍流模型**：k-omega SST
- **湍流强度**：5%

### 2.1 关键物理

由于流动分离，后向台阶在台阶下游产生一个**回流区**。在逆向流重新附着到下游壁面处形成一个**再附着点**。

### 2.2 无量纲参数

$$
Re_H = \frac{U H}{\nu} = \frac{1 \times 0.012}{1.5 \times 10^{-5}} \approx 800
$$

对于该 Re 下的湍流，回流泡长度为：

$$
L_r \approx 6.5 H \approx 0.078 \text{ m}
$$

---

## 3. 工作流程

### 3.1 求解器初始化

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 边界条件

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
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

### 3.3 运行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

### 4.1 回流区分析

回流区可由轴向速度为负值来识别：

$$
u_x < 0 \quad \text{in the recirculation region}
$$

再附着长度是在台阶下游壁面上满足 $u_x = 0$ 的位置处测得。

### 4.2 报告生成

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Backward-Facing Step Report",
    author="FoamPilot",
)

report.add_statistic("Re_H", 800, "", "Hydraulic Reynolds number")
report.add_statistic("L_r_expected", 6.5, "H", "Expected reattachment length ratio")

report.save_html_report(filename="step_report.html")
```

### 4.3 LaTeX/Typst 报告

```python
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

# LaTeX
doc = LatexDocument("Backward-Facing Step", "FoamPilot",
                    output_dir=case_path)
doc.add_title()
doc.add_section("Recirculation Zone", "Length and velocity analysis.")
doc.generate_document(output_format="tex")

# Typst
tdoc = ScientificDocument("BFS Analysis", "FoamPilot")
tdoc.add_equation(r"L_r = 6.5 H", caption="Reattachment length", label="eq:reattachment")
r = TypstRenderer()
r.render(tdoc)
```

---

## 5. 预期结果

| 量 | 预期 |
|----------|----------|
| 回流长度 (L_r/H) | 6.0–7.0 |
| 再附着点 x/H | 6.5 |
| 速度恢复 | 在 x/H ≈ 20 处恢复 |

---

## 6. 运行

```bash
cd foampilot/tutorials/03_pitzDaily_step
python run.py
python report_generator.py
```
