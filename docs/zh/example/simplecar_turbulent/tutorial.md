# SimpleCar Turbulent Flow — FoamPilot 教程

## 概述

本教程演示了使用 **FoamPilot** 和 `simpleFoam` 求解器，采用 **k-omega SST** 湍流模型，对简化汽车几何体进行的**稳态 RANS 湍流**模拟。

FoamPilot 自动化处理：

- 使用湍流强度设置的湍流边界条件
- 函数对象（场平均、运行时控制）
- 力和压力系数监测

📁 **位置**: `foampilot/tutorials/02_simpleCar_turbulent/`

---

## 1. 先决条件

- 已安装 OpenFOAM
- 已安装 FoamPilot
- `classy_blocks`（可选，用于几何体）

---

## 2. 计算案例物理

- **几何**：简化汽车外部空气动力学
- **流动**：不可压、湍流、稳态
- **进口速度**：30 m/s（108 km/h 迎风）
- **湍流模型**：k-omega SST
- **湍流强度**：5%

### 2.1 控制方程

采用 Boussinesq 近似的 RANS：

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nabla \cdot \left[ \nu_{eff} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right]
$$

### 2.2 k-omega SST 模型

湍流动力学能：

$$
\frac{\partial (\rho k)}{\partial t} + \frac{\partial (\rho u_j k)}{\partial x_j} = P_k - \beta^* \rho k \omega
$$

比耗散率：

$$
\frac{\partial (\rho \omega)}{\partial t} + \frac{\partial (\rho u_j \omega)}{\partial x_j} = \alpha S_\omega
$$

### 2.3 无量纲参数

基于汽车长度 L = 4.5 m 的风洞雷诺数：

$$
Re_L = \frac{U L}{\nu} = \frac{30 \times 4.5}{1.5 \times 10^{-5}} \approx 9 \times 10^6
$$

---

## 3. 工作流程

### 3.1 求解器设置

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
    velocity=(ValueWithUnit(30, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
)
```

FoamPilot 的 `velocityInlet` 与 `turbulence_intensity` 会自动计算入口处的 `k` 和 `omega` 值：

$$
k = \frac{3}{2} (I \cdot U)^2, \quad \omega = \frac{\sqrt{k}}{L_{ref} \cdot 0.016}
$$

### 3.3 函数对象

FoamPilot 支持添加用于监测的函数对象：

```python
solver.system.functions.velocity_field_average = {
    "type": "fieldAverage",
    "enabled": True,
    "fields": [("U", "U_mean", "U_rms")],
}
```

### 3.4 运行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

### 4.1 力系数

FoamPilot 通过函数对象监测阻力和升力系数：

```
forces {
    type            forces;
    functionObjectLibs ("libforces.so");
    patches          (car body walls);
    rho            rhoInf;  // 不可压缩
    liftDir        (0 0 1);
    dragDir        (1 0 0);
    CofR           (0 0 0);
}
```

### 4.2 压力系数

$$
C_p = \frac{p - p_\infty}{\frac{1}{2} \rho U_\infty^2}
$$

### 4.3 报告生成

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="SimpleCar Aerodynamics Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("External aerodynamics simulation of a simplified car.")
doc.add_section("Drag Coefficient", f"Cd = {cd_value:.4f}")
doc.add_section("Pressure Distribution", "")
for img in ["pressure_contour.png", "velocity_contour.png"]:
    doc.add_figure(img, caption=img.replace("_", " ").title())
doc.generate_document(output_format="pdf")
```

---

## 5. 预期结果

| Quantity | Expected Value |
|----------|----------------|
| Drag coefficient (Cd) | 0.25–0.35 |
| Lift coefficient (Cl) | 0.1–0.2 |
| Max. Cp | ~1.2 |
| Reattachment length behind car | ~2–3 car lengths |

（表头可译为“物理量 / 预期值”，但此处保留原表格列名以保持与原始文档一致。）

---

## 6. 执行

```bash
cd foampilot/tutorials/02_simpleCar_turbulent
python run.py
python report_generator.py
```
