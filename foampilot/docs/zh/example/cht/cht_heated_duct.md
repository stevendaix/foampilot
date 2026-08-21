# CHT 热交换器 – FoamPilot 教程

## 概述

本教程演示使用 **FoamPilot** 和 **OpenFOAM 13** (`chtMultiRegionFoam`) 的一个完整的、用于管壳式换热器的 **CHT（共轭传热）工作流**。它模拟了加热管道内的稳态、层流、可压缩空气流动，并与固体铜壁耦合。

FoamPilot 的 **CHT 模块** (`foampilot.cht`) 提供专用类：

- `ChtSolver` — 多区域共轭传热求解器
- `FluidRegion` — 流体域 (heRhoThermo, 可压缩)
- `SolidRegion` — 固体域 (heSolidThermo)
- `CoupledInterface` — 流体-固体热耦合接口
- `FixedTemperatureBC`, `CoupledTemperatureBC`, 等 — 边界条件辅助类
- `calc_nusselt_number`, `calc_heat_transfer_coefficient`, `calc_thermal_resistance` — 后处理函数

FoamPilot 报告引擎

`CFDReportGenerator` 集成了：

- `LatexDocument` — 通过 PyLaTeX 生成 LaTeX/PDF 报告
- `ScientificDocument` / `TypstRenderer` — 基于 Typst 的科学文档

`CFDReportGenerator` 提供：

- `add_statistic()` — 注册标量统计量（Re、Nu、h 等）
- `add_figure()` — 注册图像
- `add_table()` — 注册数据表
- `collect_time_series()` — 收集跨时间步的场统计
- `collect_region_statistics()` — 每区域场统计
- `save_html_report()` — 包含 Plotly 的交互式 HTML 报告
- `save_latex_report()` — 带表格和图像的 LaTeX 报告
- `save_typst_report()` — Typst 科学文档

📁 **位置**: `foampilot/tutorials/09_CHT_heatedDuct/`

---

## 1. 先决条件

- 已安装并可访问 OpenFOAM 13
- 已安装 FoamPilot（`pip install -e .`）
- Python 依赖：`pyvista`、`numpy`、`pandas`、OpenFOAM 运行时

---

## 2. 案例物理

- **几何**：管壳式换热器，包含三部分（流体-内侧、流体-外侧、固体）
  - **域尺寸**：-0.649 × 0.649 × (-3.45 to 3.45) m
  - **Fluid-Inner**：类似水的流体（ρ₀=1027 kg/m³，Cp=4195 J/kg·K，Pr=2.289，μ=3.645e-4 Pa·s）
  - **Fluid-Outer**：相同水性质，不同进口温度（353 K vs 283 K）
  - **Solid**：使用 CalculiX 的管壁结构求解器
- **流动**：稳态，层流（基于内径 0.025 m 的 Re ≈ 13,000）
- **热力学**：heRhoThermo，hConst，perfectFluid 状态方程
- **耦合**：preCICE 使用最近邻映射进行耦合
  - 接口：Solid-to-Fluid-Inner 和 Solid-to-Fluid-Outer
  - 数据交换：Sink-Temperature、Heat-Transfer-Coefficient（隐式耦合）

### 2.1 控制方程

**连续方程（不可压，Boussinesq）：**

$$
\nabla \cdot \mathbf{u} = 0
$$

**动量方程（buoyantSimpleFoam）：**

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p_{rgh} + \nabla \cdot \left[ \mu_{eff} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right] + \rho \mathbf{g}
$$

**能量方程：**

$$
\frac{\partial (\rho h)}{\partial t} + \nabla \cdot (\rho h \mathbf{u}) = \nabla \cdot \left( \frac{\kappa}{Pr} \nabla h \right)
$$

**修正压力：**

$$
p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}
$$

### 2.2 边界条件

| 补丁 | 场 | 边界条件 | 值 |
|------|----|---------|----|
| inlet (inner) | U | fixedValue | (0, 0, -0.002) m/s |
| inlet (inner) | T | fixedValue | 283 K |
| inlet (outer) | T | fixedValue | 353 K |
| outlet | T | zeroGradient | — |
| interface | T | mixed | refValue=293 K, frac=0.5 |
| adiabatic | T | zeroGradient | — |

### 2.3 preCICE 配置

preCICE 配置使用 **隐式耦合方案**：

- **交换的数据**：Sink-Temperature、Heat-Transfer-Coefficient
- **映射**：nearest-neighbor（保持一致性约束）
- **收敛**：并行显式耦合（伪时间步进以达到稳态）

---

## 3. 工作流

### 3.1 使用 CHT 求解器的案例设置

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion, CoupledInterface

fluid_region = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
    thermophysical_model="heRhoThermo",
    equation_of_state="perfectGas",
)

solid_region = SolidRegion(
    name="solid",
    temperature=350.0,
    thermal_conductivity=380.0,   # W/(m·K) — copper
    density=8960.0,               # kg/m³
    specific_heat=385.0,          # J/(kg·K)
)

solver = ChtSolver(
    case_path=case_path,
    solver_name="chtMultiRegionFoam",
    regions=[fluid_region, solid_region],
    interfaces=[CoupledInterface(...)],
)

solver.system.controlDict.start_time = 0
solver.system.controlDict.end_time = 1.0
solver.system.controlDict.delta_t = 5e-4
solver.system.controlDict.application = "chtMultiRegionFoam"

solver.setup_case()
solver.write_case()
```

### 3.2 网格生成

网格通过 JSON 配置生成：

```python
from foampilot import Meshing

mesh = Meshing(case_path, mesher="blockMesh")
mesh.mesher.load_from_json(case_path / "block_mesh.json")
mesh.mesher.write(file_path=case_path / "system" / "blockMeshDict")
solver.run_command(["blockMesh"], log_filename="log.blockMesh")
```

### 3.3 多区域设置

```python
solver.run_command(["createZones"], log_filename="log.createZones")
solver.run_command(["splitMeshRegions", "-cellZones", "-defaultRegionName", "fluid"],
                   log_filename="log.splitMeshRegions")
solver.run_command(["foamSetupCHT"], log_filename="log.foamSetupCHT")
```

### 3.4 模拟执行

```python
solver.run_simulation(nb_proc=1)
```

### 3.5 VTK 转换

```python
solver.run_command(["foamToVTK", "-region", "fluid", "-latestTime",
                    "-fields", "(T U p k omega)"],
                   log_filename="log.foamToVTK_fluid")
solver.run_command(["foamToVTK", "-region", "solid", "-latestTime",
                    "-fields", "(T)"],
                   log_filename="log.foamToVTK_solid")
```

---

## 4. 后处理

后处理脚本（`run_post.py`）使用 foampilot CHT 分析函数：

```python
from foampilot.cht import (
    calc_nusselt_number,
    calc_heat_transfer_coefficient,
    calc_thermal_resistance,
    calc_total_heat_balance,
    calc_temperature_contour,
)
```

### 4.1 主要结果

| 指标 | 值 | 参考 |
|------|----|------|
| Interface T (fluid side) | 293.00 K | preCICE reference |
| Interface T (solid side) | 293.00 K | preCICE reference |
| Heat transfer coefficient h | Variable | Coupled via preCICE |
| Mass flow rate (inner) | ~0.005 | kg/s |
| Mass flow rate (outer) | ~0.15 | kg/s |
| Temperature difference ΔT | 70 K | 353−283 K |

### 4.2 温度统计

| 区域 | T_min (K) | T_max (K) | T_mean (K) |
|------|-----------|-----------|------------|
| Fluid-Inner | 283.00 | 353.00 | ~293 |
| Fluid-Outer | 283.00 | 353.00 | ~318 |
| Solid | 283.00 | 353.00 | ~303 |

### 4.3 网格统计

| 属性 | 内侧流体 | 外侧流体 |
|------|----------|----------|
| Cells | ~100,000 | ~150,000 |
| Points | 37,894 (inner) | 95,000+ (outer) |
| Faces | ~1,084,000 | ~1,700,000 |
| Patches | inlet, outlet, interface, adiabatic | inlet, outlet, interface, adiabatic |

---

## 5. 报告生成

FoamPilot 的 `CFDReportGenerator` 自动化了报告创建。CHT 教程生成了一个综合报告，示例用法如下：

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="CHT Heated Duct Report",
    author="FoamPilot",
)

# Add key statistics
report.add_statistic("Nu", 0.2597, "", "Nusselt number")
report.add_statistic("h", 3.38, "W/(m²·K)", "Heat transfer coefficient")
report.add_statistic("R_th", 0.2963, "K/W", "Thermal resistance")
report.add_statistic("T_interface", 350.0, "K", "Interface temperature")

# Add figures
report.add_figure("postProcessing/fluid_temperature_contour.png",
                  "Temperature contour (fluid)")
report.add_figure("postProcessing/solid_temperature_contour.png",
                  "Temperature contour (solid)")
report.add_figure("postProcessing/cht_temperature_contour.png",
                  "CHT temperature overlay")

# Generate LaTeX report
report.save_latex_report(compile_pdf=True)

# Generate interactive HTML report
report.save_html_report()
```

### 5.1 报告类型

| 方法 | 输出 | 特性 |
|------|------|------|
| `save_latex_report()` | `.tex` / `.pdf` | 通过 PyLaTeX 生成 LaTeX，包含表格和图像 |
| `save_typst_report()` | `.typ` | Typst 科学文档 |
| `save_html_report()` | `.html` | 交互式 Plotly 图像，嵌入表格 |

---

## 6. 文件

| 文件 | 描述 |
|------|------|
| `run.py` | 教程主脚本（CHT 设置、网格、模拟） |
| `run_post.py` | 后处理与分析 |
| `block_mesh.json` | 用于 `BlockMesher` 的几何配置 |
| `README.md` | 本文档 |

---

## 7. 执行

```bash
cd foampilot/tutorials/09_CHT_heatedDuct
python run.py
python run_post.py
```

---

## 8. 预期输出

```
postProcessing/
├── temperature_statistics.csv
├── temperature_profile.csv
├── temperature_profile_combined.csv
├── fluid_temperature_contour.png
├── solid_temperature_contour.png
├── cht_temperature_contour.png
└── CHT_Report.md
```
