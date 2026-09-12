# 架构与工作流程

FoamPilot 是围绕 OpenFOAM 的 Python 编排层。它不会替代 OpenFOAM 的求解器、网格工具或文件格式；相反，它提供用于创建、检查、执行和后处理 OpenFOAM 算例的 Python 对象。

> 应将一个 FoamPilot 算例视为可复现的构建工件：Python 脚本是唯一可信来源，而 `0/`、`constant/` 与 `system/` 是生成的输入与仿真输出。

## 端到端工作流程

典型工作流程包含六个阶段：

| 阶段 | FoamPilot 职责 | 主要输出 |
| --- | --- | --- |
| 定义 | 创建一个 `Solver`、物性参数以及边界条件对象。 | Python 配置 |
| 网格 | 使用 `blockMesh`、Gmsh、snappyHexMesh 或直接的 OpenFOAM 网格来生成或导入网格。 | `constant/polyMesh` 与网格字典 |
| 配置 | 写入 `controlDict`、离散化方案、线性求解器、输运性质、湍流、重力，以及可选的 function objects。 | `system/` 与 `constant/` |
| 运行 | 启动串行或并行的 OpenFOAM 求解器，并将日志保存在算例目录中。 | 时间目录与日志文件 |
| 检查 | 直接读取原生 OpenFOAM 结果，或将其转换为 VTK 供 PyVista 使用。 | PyVista 网格与派生场 |
| 报告 | 生成图表、仪表板、CSV 汇总、LaTeX PDF 或 Typst 文档。 | 图、表与报告 |

这些阶段被有意设计为显式可控。脚本可以在网格生成后停止、修改已生成的字典，或在不重建算例的情况下仅重新运行后处理。

## 包映射

公共包按职责而非按 OpenFOAM 可执行程序进行组织：

| 包 | 用途 |
| --- | --- |
| `foampilot.base` | 算例路径、文件抽象与网格生成编排。 |
| `foampilot.solver` | 求解器选择、算例设置、执行、分解与重组。 |
| `foampilot.boundaries` | Patch 分配、标准边界条件、原始字典以及基于 CSV 的条件。 |
| `foampilot.constant` | 流体、湍流、重力、相、辐射与材料字典。 |
| `foampilot.system` | `controlDict`、`fvSchemes`、`fvSolution`、function objects、约束、模型与分解。 |
| `foampilot.cht` | 多区域共轭传热，包含流体/固体区域与界面条件。 |
| `foampilot.mesh` 与 `foampilot.openfoam` | 网格生成、直接网格导出、Gmsh 与 snappyHexMesh 辅助工具。 |
| `foampilot.postprocess` | 基于 PyVista 的后处理、原生 OpenFOAM 读取器、风环境分析与网页展示。 |
| `foampilot.report` | 网格报告、收敛性报告、并行研究、LaTeX 与 Typst 渲染。 |
| `foampilot.urban` | 实验性的城市 CFD 数据模型、简化、几何、网格生成、Patch、验证与 OSM 读取器。 |
| `foampilot.utilities` | 单位、流体性质、残差、气象文件、几何转换与耦合工具。 |

## 生成文件与验证

FoamPilot 通过 Python 文件对象写入 OpenFOAM 字典。每次生成步骤后，都应检查生成的文件，而不是仅依赖内存中的属性。尤其需要确认 `system/controlDict`、`0/` 中的所有初始场、`constant/polyMesh` 下的网格，以及相关材料字典均已写出。

对于不可压缩算例，`constant/transportProperties` 必须包含求解器使用的动态量，包括运动黏度 `nu`。如果某个值被动态赋予但未出现在生成的字典中，应将该算例视为无效，并在启动 OpenFOAM 之前检查对应的 constant 目录写入器。

最小验证序列为：

```bash
checkMesh -case path/to/case
foamDictionary path/to/case/constant/transportProperties -entry nu
foamDictionary path/to/case/system/controlDict -entry application
```

具体的验证命令取决于 OpenFOAM 发行版。FoamPilot 可以生成文件，但在字典语法、网格有效性以及求解器兼容性方面，OpenFOAM 仍是权威。

## 可选依赖

基础安装与可选扩展在 `pyproject.toml` 中分离管理。`docs` 扩展安装 MkDocs，`dev` 扩展安装测试与代码检查工具，`gnn` 包含图学习依赖项，`urban` 包含 OSMnx、GeoPandas、Rasterio 以及 LAS/LAZ 支持等地理空间读取器。

```bash
pip install -e ".[dev,docs]"
# 可选的城市工作流
pip install -e ".[urban]"
```

部分专用工具还需要系统级应用或外部数据集。在干净环境中运行工作流程之前，请查看相关示例。
