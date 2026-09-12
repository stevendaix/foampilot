# 高级工作流程

This page gathers workflows that are present in the package but were not previously described in the English documentation. They are more specialised than the basic cavity and external-aerodynamics examples and should be considered experimental unless the corresponding tutorial has been validated with the target OpenFOAM release.

## 共轭传热

`foampilot.cht` 包用于为 `chtMultiRegionFoam` 和相关求解器构建多区域案例。主要对象为 `ChtSolver`、`FluidRegion`、`SolidRegion` 和 `CoupledInterface`。边界条件工厂覆盖固定温度、热通量、进口/出口温度、对称、全温、辐射耦合和耦合界面条件。

一个 CHT 案例按区域组织：

```text
case/
├── 0/
│   ├── fluid/
│   └── solid/
├── constant/
│   ├── fluid/
│   ├── solid/
│   └── regionInterfaces/
└── system/
```

生成的 `controlDict` 包含区域求解器映射。串行运行会启动独立的 CHT 可执行程序；并行运行会对所有区域进行分区、启动 MPI，并重构所有区域。

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion

fluid = FluidRegion(name="fluid", temperature=300.0)
solid = SolidRegion(name="solid", temperature=350.0)
solver = ChtSolver(
    case_path="case",
    solver_name="chtMultiRegionFoam",
    regions=[fluid, solid],
)
solver.setup_case()
solver.run_simulation(nb_proc=1)
```

确切的构造函数和材料参数取决于案例使用的 OpenFOAM 版本。将 `09_CHT_heatedDuct` 教程作为可执行参考，并在运行前检查生成的区域字典。

CHT 后处理辅助工具可以计算热通量、界面热通量、努塞尔数、热边界层厚度、换热系数、总热平衡、温度等值线和热阻。

## Windkessel 和血流动力学实用工具

顶层包中提供 `WindkesselModel` 用于降阶心血管边界建模。它应与明确的压/流约定耦合，并在投入生产使用前针对目标 OpenFOAM 边界条件进行验证。

实用工具包还包含血管和医疗几何的辅助工具，包括 NIfTI 到 STL 的转换、主动脉表面清理、网格优化和 CSV foam 积分器。这些工具可能需要可选包，例如 NiBabel、Trimesh、VMTK、PyFQMR 或 PyACVD。

## 天气与大气输入

`WeatherFileEPW` 可读取 EnergyPlus Weather (EPW) 文件。它可用于提取室外温度、风、辐射和其他时间序列输入，然后将其转换为 FoamPilot 边界条件或大气强迫。将 EPW 文件视为输入数据集，并在案例元数据中记录其来源、位置和时区。

`foampilot.utilities.wind_profile` 和 `foampilot.postprocess.wind_analysis` 模块提供风廓线和风集合辅助工具。它们对于比较多个风向或大气边界层假设很有用，但不能替代对大气边界条件的物理校准。

## 城市 CFD

`foampilot.urban` 包是一个用于城市尺度 CFD 的实验性管道。它暴露了建筑、地形、道路和 CFD 域的数据模型；几何简化与清理；基于 Gmsh 或表面的四分域构建器；网格尺寸与尾流细化对象；补丁分配；大气边界层剖面；以及几何/网格验证。

一个高层工作流程为：

```python
from foampilot.urban import (
    UrbanModel,
    CFDSimplifier,
    MeshConfig,
    ABLProfile,
    GeometryValidator,
)

# 从支持的读取器加载或构建 UrbanModel。
# 为 CFD 简化几何、构建域、设定网格尺寸、
# 分配补丁、验证，然后导出到 OpenFOAM 工作流。
```

OSM 和 LiDAR 读取器为可选项，因为它们依赖地理空间库和外部数据集。在导入它们之前安装额外依赖：

```bash
pip install -e ".[urban]"
```

城市案例应记录坐标参考系、度量换算、风框架、地形来源、建筑高度假设、简化容差、网格预算和大气剖面。这些细节对可重复性至关重要，不能仅从生成的网格中安全推断。

## MakeHuman 与体温调节

仓库包含一个用于体温调节实验的 MakeHuman-to-STL 工作流。该工作流导出身体模型、选择主要皮肤表面、创建 JOS-3 表面分区，并为后续耦合写入分区映射。它是一个外部工作流，而不是通用的 FoamPilot 求解器功能，因此英文文档应指向其 README 并明确说明其外部依赖。

使用此工作流时，应记录 MakeHuman 版本、模型姿势、导出的表面组、JOS-3 区域映射以及用于耦合的 OpenFOAM 案例。在未检查表面拓扑和区域分配之前，不要将生成的 STL 解释为已验证的生理模型。
