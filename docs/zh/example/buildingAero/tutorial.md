# 建筑气动学 — FoamPilot 教程

## 概览

本教程使用 `simpleFoam`（k-omega SST）模拟城市环境中建筑物周围的 **湍流** 流动。它演示了使用 `topoSet` 和 `createPatch` 的高级网格操作。

FoamPilot 自动化：

- `topoSet` 和 `createPatch` 的执行
- 城市边界条件（来流、建筑物）
- 风荷载分析

📁 **位置**: `foampilot/tutorials/06_buildingAero/`

---

## 1. 先决条件

- 已安装 OpenFOAM
- 已安装 FoamPilot

---

## 2. 工况物理

- **域**：包含多座建筑的城市峡谷（10 m × 10 m × 3 m）
- **流动**：不可压、湍流、稳态
- **入口速度**：10 m/s（50% 城市湍流强度）
- **湍流模型**：k-omega SST
- **重力**：关闭（压力驱动，稳态 RANS）

### 2.1 城市边界层剖面

入口速度（对数剖面）：

$$
u(y) = u_* \frac{\ln(y / y_0)}{\kappa}
$$

其中：
- `u*` — 摩擦速度
- `κ` — 冯·卡门常数 (0.41)
- `y0` — 粗糙度高度

FoamPilot 使用 `velocityInlet` 和 `turbulence_intensity` 简化该设置：

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.15,
)
```

### 2.2 峡谷风效应

建筑物在下游会产生 **街道峡谷涡旋**。峡谷的长宽比（建筑高度 / 街道宽度）决定流动状态：

$$
AR = \frac{H_{building}}{W_{street}}
$$

对于 AR ≈ 1（本教程），流动处于“临界”状态，峡谷内部出现强烈回流。

---

## 3. 工作流程

### 3.1 求解器初始化

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 使用 topoSet + createPatch 的网格操作

FoamPilot 封装了 OpenFOAM 的拓扑工具：

```python
# topoSet for defining building cell zones
solver.system.run_topoSet()

# createPatch for renaming boundary patches
solver.system.run_createPatch()
```

这会执行：

- `system/topoSetDict` — 为建筑定义单元/区域集合
- `system/createPatchDict` — 将面重命名为命名的边界面

### 3.3 边界条件

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.15,
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*building.*",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

### 3.4 执行

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. 后处理

### 4.1 峡谷流场可视化

使用 PyVista 可视化速度场：

```python
import pyvista as pv
from pathlib import Path

mesh = pv.read(str(Path("VTK/latest/cellular.vtk")))
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh.slice("z"), scalars="U", cmap="viridis")
plotter.screenshot("canyon_velocity.png")
```

### 4.2 报告生成

```python
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.report_generator import CFDReportGenerator

# HTML report
report = CFDReportGenerator(
    case_path=case_path,
    title="Building Aero Report",
    author="FoamPilot",
)
report.add_statistic("U_inlet", 10.0, "m/s", "Inlet velocity")
report.add_statistic("I_inlet", 0.15, "", "Turbulence intensity")
report.save_html_report(filename="building_report.html")

# LaTeX report
doc = LatexDocument(
    title="Urban Building Aerodynamics",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Wind flow simulation around urban buildings.")
doc.add_section("Canyon Flow", "")
doc.add_figure("canyon_velocity.png", "Velocity field in urban canyon")
doc.generate_document(output_format="pdf")
```

---

## 5. 预期结果

| 物理量 | 预期 |
|--------|------|
| 屋顶高度处的风速提升 | 1.2–1.5× U_inlet |
| 峡谷回流区 | 在每栋建筑后方可见 |
| 压强系数 Cp | -0.5 到 +1.0 |
| 行人高度速度 | < 0.2 U_inlet |

---

## 6. 执行

```bash
cd foampilot/tutorials/06_buildingAero
python run.py
python report_generator.py
```
