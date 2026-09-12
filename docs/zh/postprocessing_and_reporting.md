# 后处理与报告

FoamPilot 支持两条互补的后处理路径。传统路径使用 `foamToVTK` 将算例转换为 VTK，然后用 PyVista 载入该 VTK 输出。直接路径则无需生成中间的 VTK 树，直接将原生 OpenFOAM 网格与场读入 PyVista。

## 原生 OpenFOAM 读取器

当算例已包含有效的 `constant/polyMesh`，且希望避免外部转换步骤时，使用直接读取器。

```python
from foampilot.postprocess import OpenFOAMDirectReader

reader = OpenFOAMDirectReader("/path/to/case")
mesh = reader.to_pyvista(fields=["U", "p"], time_step="latest")
print(mesh.n_points, mesh.n_cells)
```

该读取器可从 OpenFOAM 场文件头自动识别点场与单元场，支持惰性加载、字段缓存，并可读取压缩场文件。便捷函数对于小型脚本很有用：

```python
from foampilot.postprocess import read_openfoam

mesh = read_openfoam(
    "/path/to/case",
    fields=["U", "p"],
    time_step="latest",
)
```

对于共轭传热（CHT）算例，使用 `CHTDirectReader`。它会自动识别流体与固体区域，并返回一个 PyVista 的 `MultiBlock` 结构。

```python
import pyvista as pv
from foampilot.postprocess import CHTDirectReader

reader = CHTDirectReader("/path/to/cht-case")
print(reader.region_names)
blocks = reader.get_all_meshes(fields=["T"], time_step="latest")

plotter = pv.Plotter(off_screen=True)
for region_name, region_mesh in blocks.items():
    plotter.add_mesh(region_mesh, scalars="T", name=region_name)
plotter.screenshot("temperature.png")
plotter.close()
```

当存在具名的区域界面时，可直接检查界面温度：

```python
interface = reader.get_interface_temperatures(
    "fluid_to_solid", time_step="latest"
)
print(interface["fluid_T"])
print(interface["solid_T"])
print(interface["T_interface"])
```

## PyVista 后处理

当既有工作流依赖 `foamToVTK`、时间步点发现机制或更高层的绘图辅助器时，`FoamPostProcessing` 仍然十分有用。

```python
from foampilot.postprocess import FoamPostProcessing

post = FoamPostProcessing(case_path="/path/to/case")
post.foamToVTK()
time = post.get_all_time_steps()[-1]
mesh = post.load_time_step(time)["cell"]
```

常见操作包括切片、等值面、矢量图、涡结构分析、网格统计、图像导出以及动画导出。在无显示环境中，可使用 `off_screen=True` 创建图像，或使用 FoamPilot 的渲染辅助工具以检测可用的离屏后端。

## 交互式网页展示

`foampilot.postprocess.web_presentation` 模块提供用于速度、压力与温度场的 Plotly 构建器，以及一个用于交互探索的 `CFDDashboard`。一个最小用法模式如下：

```python
from foampilot.postprocess.web_presentation import (
    plotly_velocity_magnitude,
    plotly_pressure_contour,
    CFDDashboard,
)

velocity_figure = plotly_velocity_magnitude(mesh)
pressure_figure = plotly_pressure_contour(mesh)
# Pass the figures to the dashboard or to a Plotly/Streamlit application.
```

该仪表板用于探索与沟通。要获得可复现实的工程记录，请将输入脚本、生成的字典、求解器日志、图像与报告一起保存。

## 仿真与网格报告

`foampilot.report` 包含针对网格质量、收敛性与求解器研究的结构化报告。报告 API 设计为在仿真结束后运行，以便记录失败或未完成的计算，而非被静默忽略。

当需要 PDF 形式的计算说明时，LaTeX API 很适合：

```python
from foampilot.report import latex_pdf

document = latex_pdf.LatexDocument(
    title="OpenFOAM simulation report",
    author="FoamPilot",
    filename="simulation_report",
    output_dir="postProcessing/report",
)
document.add_section("Purpose", "Summary of the simulated case.")
document.add_figure("postProcessing/velocity.png", caption="Velocity magnitude")
document.generate_document(output_format="pdf")
```

若无需 LaTeX 工具链进行文档生成，Typst 渲染器提供了章节、方程、图、表、代码块与参考文献等结构化构件。当项目已使用 `.typ` 模板或对确定性排版要求较高时，优先考虑 Typst。

## 并行研究

`ParallelStudy` 可自动化比较不同的处理器分解。它可以写出 `decomposeParDict`，运行基线与并行算例，解析日志、收集计时与网格度量，并导出处理器边界的可视化。需要保证 OpenFOAM 与 MPI 运行时可在 `PATH` 中访问。

在启动研究之前，请复制算例或使用一次性的输出目录。并行运行会通过创建处理器目录与重构输出来修改算例。

## 推荐的结果布局

可复现的项目可采用如下布局：

```text
case_project/
├── run.py
├── case/
│   ├── 0/
│   ├── constant/
│   └── system/
├── logs/
├── postProcessing/
│   ├── figures/
│   ├── tables/
│   └── reports/
└── README.md
```

将生成的输出与源几何和 CSV 输入分开存放。这样就可以删除算例目录并通过脚本重建，而不丢失该次运行的科学可追溯性。

## 场类型与派生量

后处理应区分“点数据”“单元数据”和“表面数据”。存储在单元中心的速度向量不能与插值到顶点的数值直接互换。表面压力与壁面剪切应力必须在实际壁面补丁上积分，而体积平均需要单元体积。

常见的派生量包括：

| 量 | 典型定义或用途 |
| --- | --- |
| 速度大小 | $|\mathbf{U}|$，用于速度图与阈值区域。 |
| 涡量 | $\nabla\times\mathbf{U}$，用于识别旋转结构。 |
| Q 判据 | 识别旋转占优于应变的区域。 |
| 壁面剪切应力 | 壁面上的切向牵引；对近壁网格敏感。 |
| 压力系数 | $C_p=(p-p_\infty)/(\tfrac12\rho U_\infty^2)$，用于外流。 |
| 热通量 | 表面法向导热或总热通量。 |
| 努塞尔数 | 基于给定特征长度的无量纲传热强度。 |
| 相分数 | VOF 中的界面位置与液相体积分数诊断。 |
| 标量混合指数 | 被输运浓度场的均匀度或方差指标。 |

每个导出的量都必须附带定义、参考态、符号约定与平均操作说明。

## 残差与收敛

残差是迭代过程中衡量离散方程满足程度的代数量；它并不自动等价于对所关心物理量的误差估计。一个算例可能残差很小，但阻力系数、热量收支或出口流量分配仍然不正确。

因此，一个稳健的后处理报告应包含：

1. 各区域、各场的求解器残差历史；
2. 受监测的力、通量、温度或标量平均；
3. 连续性误差与体积守恒；
4. 最终网格统计；
5. 最终物理时间、时间步长、Courant 数与迭代次数；
6. 工程输出的收敛判据。

`ResidualsPost` 可将求解器日志转换为 CSV、JSON、PNG 或 HTML 工件。请保留原始日志文件，因为解析后的摘要可能会掩盖警告、浮点异常或求解器重启信息。

## 边界与补丁分析

对补丁（patch）层级的分析对于外部空气动力学、生物医学流动与 CHT 至关重要。一个可靠的补丁报告应明确补丁名称、补丁类型、面积、面数、场量的最小/最大/平均值，并在适用时给出积分通量或力。

对于车辆，应按补丁与方向分别报告力。对于血管模型，应在每个入口/出口报告流量与压力，并校验守恒。对于 CHT，应分别在界面的流体侧与固体侧报告热通量，并明确法向约定。

## 风场集合与多算例

风场分析模块提供 `WindRose`、`WindCaseResult`、`WindEnsemble`、`LawsonProcessor` 与 `LawsonVisualizer` 等对象。它们可以组织多种风向或大气工况，并将其结果汇总为方向性统计。它们并不替代入口剖面的物理定义或舒适性判据的选择。

一个风场集合应为每个算例记录：

| 元数据 | 示例 |
| --- | --- |
| 方向 | 气象学或笛卡尔约定，需明确说明。 |
| 参考速度 | 高度与平均时段。 |
| 大气剖面 | 对数律、幂律、实测剖面或前驱场。 |
| 稳定度 | 中性、稳定、不稳定或未知。 |
| 求解器/模型 | RANS 关闭、壁函数、时间步长与离散格式。 |
| 权重 | 分配给该算例的频率或概率。 |

## CHT 后处理

对于 CHT 算例，应在同一物理时间载入所有区域。将一个时刻的流体场与另一时刻的固体场进行比较会造成伪界面不匹配。直接的 `CHTDirectReader` 可将温度场载入为 `MultiBlock` 对象；CHT 工具还可计算界面温度、热通量、热阻、传热系数与努塞尔数。

一份最小的 CHT 报告应包含：

- 流体与固体区域名称；
- 材料属性及其随温度的变化；
- 界面补丁配对；
- 界面温度的连续性；
- 界面热通量的连续性；
- 进入、离开与储存的总热量；
- 局部与积分努塞尔数；
- 法向于界面的网格分辨率；
- 收敛历史。

## 数据导出与可追溯性

当将场导出到 CSV、JSON、VTK 或图像时，应编写一份包含以下内容的元数据文件：

```text
case identifier
OpenFOAM version
FoamPilot commit
mesh cell count
physical time
field names and locations
units
coordinate system
filter/interpolation operation
reference values
```

这对生物医学与城市算例尤为重要，因为可视化结果可能会与原始几何、坐标参考系或患者/环境输入数据相分离。

## 报告类型

FoamPilot 支持多层级的报告：

| 报告 | 最佳用途 |
| --- | --- |
| 残差 CSV/PNG/HTML | 开发阶段的快速数值健康检查。 |
| 网格质量报告 | 求解前的几何与离散化审查。 |
| 仿真报告 | 可复现实的算例摘要，包含图表与表格。 |
| 并行研究报告 | 处理器数量对比与分解诊断。 |
| LaTeX PDF | 正式的计算说明或发表风格的报告。 |
| Typst 文档 | 无需 LaTeX 工作流的结构化科学文档。 |
| Streamlit/Plotly 仪表板 | 面向工程师与协作者的交互式探索。 |

不要仅用仪表板作为唯一归档。交互状态可能丢失；算例脚本、字典、求解器日志、原始数据与静态摘要才是可复现实的记录。
