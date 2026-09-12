# MotorBike External Aerodynamics — FoamPilot Tutorial

## Overview

本教程使用 `simpleFoam` (k-omega SST) 模拟**高速绕摩托车的外部流动**，演示壁面解析网格划分和尾迹预测。

FoamPilot 自动化：

- 高速入口设置 (30 m/s)
- 墙面和移动地面边界条件
- 阻力和升力监测

📁 **位置**: `foampilot/tutorials/07_motorBike/`

---

## 1. Prerequisites

- OpenFOAM 已安装
- FoamPilot 已安装

---

## 2. Case Physics

- **几何**: 带道路表面的摩托车模型
- **流动**: 不可压、湍流、稳态
- **速度**: 30 m/s（108 km/h 高速）
- **湍流模型**: k-omega SST
- **湍流强度**: 5%

### 2.1 Dimensionless Parameters

基于车辆长度 (L = 2.0 m) 的雷诺数：

$$
Re_L = \frac{U L}{\nu} = \frac{30 \times 2}{1.5 \times 10^{-5}} = 4 \times 10^6
$$

阻力系数：

$$
C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}
$$

其中：
- `Fd` — 阻力
- `A` — 迎风面积（摩托车约为 0.7 m²）

### 2.2 Wake Prediction

摩托车下游的尾迹表现为：

- 速度亏损
- 湍流混合
- 压力恢复

$$
T_{aw} = T_\infty \left[ 1 + r \frac{\gamma - 1}{2} M_\infty^2 \right]
$$

(高速流的恢复温度公式)

---

## 3. Workflow

### 3.1 Solver Initialization

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 Boundary Conditions

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(30, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*wheels.*|.*moving.*",
    condition_type="wall",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*road.*",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

FoamPilot 的通配符模式系统处理复杂的补片：

- `.*wheels.*` — 匹配所有车轮补片
- `.*moving.*` — 匹配移动表面
- `.*road.*` — 匹配地面

### 3.3 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Force Coefficients

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="MotorBike Aerodynamics",
    author="FoamPilot",
)
report.add_statistic("Re_L", 4e6, "", "Reynolds number")
report.add_statistic("Cd_expected", 0.35, "", "Expected drag coefficient")
```

### 4.2 LaTeX Report

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="MotorBike External Aerodynamics",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Aerodynamic analysis of motorcycle at 30 m/s.")
doc.add_section("Method", "")
doc.add_equation(r"Re_L = \frac{UL}{\nu}")
doc.add_section("Results", "")
doc.add_table(
    [["Drag coeff", "0.35"], ["Lift coeff", "0.05"]],
    headers=["Coefficient", "Value"],
    caption="Aerodynamic coefficients",
)
for img in ["pressure_contour.png", "velocity_vectors.png"]:
    doc.add_figure(img, caption=img.replace("_", " ").title())
doc.generate_document(output_format="pdf")
```

---

## 5. Expected Results

| 物理量 | 期望值 |
|--------|--------|
| 阻力系数 (Cd) | 0.30–0.40 |
| 迎风阻力 | ~200–250 N |
| 尾迹尺寸 | ~3–5 车长 |
| 尾部压力恢复 | ~70–80% |

---

## 6. Execution

```bash
cd foampilot/tutorials/07_motorBike
python run.py
python report_generator.py
```
