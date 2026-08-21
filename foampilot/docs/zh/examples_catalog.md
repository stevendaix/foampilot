# 完整示例与教程目录

此页面列出当前随 FoamPilot 分发的可执行示例的地图。每个教程都足够小以便检查，但每个教程也遵循相同的可重复模式：在 Python 中定义案例，生成 OpenFOAM 字典，创建或导入网格，应用边界条件，运行求解器，后处理结果，并存档生成的工件。

## 如何阅读教程

每个教程应从三个层面来阅读。物理层面解释守恒定律、本构假设和无量纲数。OpenFOAM 层面识别求解器、场、字典、边界条件和函数对象。FoamPilot 层面展示哪些 Python 对象生成这些文件以及如何重现或参数化工作流。

在运行教程之前，检查其 README 中命名的 OpenFOAM 版本、脚本所需的外部几何或教程数据，以及可选 Python 依赖项的可用性。生成的案例在解释图形之前必须用 `checkMesh` 和求解器日志进行检查和验证。

## 汇总矩阵

| Tutorial | Main physics | Solver family | Mesh strategy | Main outputs |
| --- | --- | --- | --- | --- |
| `01_cavity_laminar` | Incompressible laminar recirculation | `icoFoam` / incompressible transient flow | `blockMesh` | Velocity, pressure, residuals, figures, report |
| `02_simpleCar_turbulent` | Steady external turbulent aerodynamics | `simpleFoam` / incompressible RANS | Gmsh or imported geometry with boundary patches | Velocity, pressure, wall forces, report |
| `03_pitzDaily_step` | Backward-facing-step separation and reattachment | `simpleFoam` | Gmsh or structured channel geometry | Recirculation length, residuals, profiles |
| `04_damBreak_multiphase` | Transient water-air free surface | `interFoam` | Gmsh/block-style 2-D domain | Interface evolution, phase fraction, animation |
| `05_scalarTransport` | Passive scalar or temperature-like transport | `scalarTransportFoam` function object | Channel mesh | Scalar contours, time histories, CSV data |
| `06_buildingAero` | Urban external wind and wakes | `simpleFoam` | `blockMesh` background plus `snappyHexMesh` | Wind field, turbulence, wake and building statistics |
| `07_motorBike` | External vehicle aerodynamics | `simpleFoam` or OpenFOAM-13 `incompressibleFluid` path | `blockMesh` plus `snappyHexMesh` | Drag, pressure, wake, animation, report |
| `08_thermalBuoyancy` | Natural convection with buoyancy | Boussinesq/compressible thermal flow | `blockMesh` | Temperature, `U`, `p_rgh`, residuals, thermal report |
| `09_CHT_heatedDuct` | Conjugate heat transfer in fluid and solid regions | `chtMultiRegionFoam` | `blockMesh`, zones, region splitting | Region temperatures, heat flux, Nusselt number, balance |
| Muffler case study | Internal flow, pressure loss, acoustic/fluidic analysis | Case-specific OpenFOAM workflow | JSON/geometry-driven setup | Pressure, velocity, acoustic or flow report |
| SimpleCar case study | Scripted external-flow case with JSON mesh configuration | Case-specific incompressible workflow | JSON-based mesh generation | Case dictionaries, fields, figures, and report |

## 1. Lid-driven cavity: laminar transient flow

### 目的

腔体是粘性不可压流的典型验证案例。一个方形腔体中包含流体，上壁以规定速度运动，剩余壁面固定不动。该案例孤立出粘性扩散、压力-速度耦合、壁面无滑移条件以及回流单元的发展。

### 数学模型

本教程求解不可压 Navier–Stokes 方程：

$$
\nabla\cdot\mathbf{U}=0,
$$

$$
\frac{\partial\mathbf{U}}{\partial t}+\nabla\cdot(\mathbf{U}\mathbf{U})
=-\nabla p+\nu\nabla^2\mathbf{U}.
$$

雷诺数通常保持足够低以获得层流参考解。问题是瞬态的，因为流动从静止开始并趋向稳态回流状态。

### FoamPilot 工作流程

脚本创建一个二维 `blockMesh` 案例，写入 `controlDict`、`fvSchemes`、`fvSolution` 和 `transportProperties`，然后施加移动盖速度和无滑移墙面。它运行瞬态求解器，提取残差，并生成图表或报告。

### 需要验证的内容

主要的验证量是中心线速度剖面、回流单元的数量和位置、残差衰减，以及对网格加密和时间步长的敏感性。仅凭目视上合理的涡旋不足以证明正确性：应将中心线剖面与已发表的基准或 OpenFOAM 参考案例进行比较。

## 2. SimpleCar: steady turbulent external aerodynamics

### 目的

该案例介绍简化车辆周围的外部流动。它演示如何将物体几何放置在类风洞域中，如何规定入口湍流，以及如何从车体表面提取压力和剪切力。

### 模型与假设

流动为不可压且湍流。RANS 闭合（在该案例系列中常用 `kOmegaSST`）用涡粘性模型替代未解析的湍流应力。SST 模型之所以被选用，是因为它将 $k$–$\omega$ 系列对近壁面的敏感性与远壁面的更强自由流容忍性相融合。它是对钝体或流线型体周围分离的实用折衷；但不能替代模型验证。

阻力系数由下式得到：

$$
C_D=\frac{F_D}{\tfrac12\rho U_\infty^2 A_\mathrm{ref}},
$$

其中 $F_D$ 是流向力，$\rho$ 是密度，$U_\infty$ 是参考风速，$A_\mathrm{ref}$ 是选定的参考面积。

### 网格与边界条件

背景域在上游和下游必须足够长，以避免污染车辆的压力场。汽车表面需要一个命名的 wall patch。入口速度、湍动能以及湍流耗散或比耗散应一致地给定；出口应避免人工反射；地面处理必须与车辆是固定、移动还是以移动地面参考系表示相匹配。

### 后处理

提取表面压力、壁面剪切应力、积分力系数、分离区和尾流速度剖面。始终在报告 $C_D$ 时同时给出参考面积、参考速度、密度、湍流模型、壁面处理和网格统计信息。

## 3. PitzDaily: backward-facing step

### 目的

后退阶梯是用于研究剪切层发展、回流、再附着和湍流模型敏感性的分离内流案例。

### 物理模型

入口流经过突变的膨胀。阶梯后会形成分离气泡，再附着长度取决于雷诺数、入口剖面、湍流模型、壁面分辨率和数值格式。该案例在其名义求解器配置下通常是稳态的，但如果网格、时间步长或模型允许，分离流可能表现出非稳态行为。

### 主要诊断量

最重要的输出是再附着长度，通常以阶梯高度为基准表示。补充诊断包括壁面压力、壁面剪切应力、中心线速度、回流区长度和残差历史。结果不应仅由残差判断，因为稳态求解器可能会收敛到数值稳定但物理有偏差的解。

## 4. DamBreak: transient multiphase VOF

### 目的

DamBreak 案例演示一个瞬态自由面问题。一个水柱在重力作用下坍塌并置换空气。界面由相分数场表示，通常为 `alpha.water`。

### 控制模型

VOF 方法求解相分数的输运方程：

$$
\frac{\partial\alpha}{\partial t}+\nabla\cdot(\alpha\mathbf{U})=0,
$$

并带有界面压缩和有界性控制。混合物密度和粘度由相分数重建。重力驱动坍塌，压力应与静水部分一致地解释。

### 数值重点

库朗数、界面压缩、有界的相分数和时间步长自适应比简单增加迭代次数更为重要。在若干时刻检查 `alpha.water`，确认 $0\leq\alpha\leq1$，并验证液体体积在预期数值容差内守恒。

### 输出结果

该教程适合导出界面快照、动画、自由面高度历史、压力场和残差。同一模式可复用于摇摆、灌注、排水或波冲击案例，但每个应用都需要单独验证表面张力、润湿和接触线假设。

## 5. Scalar transport

### 目的

该案例在通道中输运一个被动标量。该标量可以表示浓度、示踪物、污染物，或在刻意简化能量方程时表示与温度类似的量。

### Equation

对于常量扩散率 $D$：

$$
\frac{\partial C}{\partial t}+\nabla\cdot(\mathbf{U}C)
=\nabla\cdot(D\nabla C)+S_C.
$$

除非加入耦合的浮力、密度、反应或源模型，否则标量不会改变流动。这种分离使该案例适用于测试对流-扩散数值方法和基于 CSV 的边界条件。

### Diagnostics

将标量剖面与预期的对流-扩散长度尺度比较。报告标量 Peclet 数、入口剖面、扩散率、出口处理、有界性和数值格式。如果标量代表温度，请明确说明它是被动场还是完全耦合的热模型。

## 6. Building aerodynamics: external urban wind

### 目的

建筑案例介绍一组障碍物在大气或类风洞域中的情况。它说明背景六面体网格与基于表面的局部加密之间的差异。

### 物理模型

流动通常为不可压且湍流。对于第一个工程模型，稳态 RANS 闭合（如 $k$–$\epsilon$ 或 $k$–$\omega$ SST）常被选用，因为它在平均风场和尾流预测上具有可管理的成本。该模型不解析所有瞬态涡旋；它通过湍流粘度预测其平均效应。

对于大气应用，当物理问题为受控风洞时，均匀入口才可接受。对于真实的大气边界层，入口速度和湍流场必须随高度变化并相互一致。参见 [户外风理论](theory_applied.md#outdoor-wind-and-atmospheric-boundary-layers)。

### 网格工作流程

典型顺序为：

```text
background blockMesh
→ surfaceFeatureExtract
→ snappyHexMesh castellated mesh
→ snap to building surfaces
→ optional boundary layers
→ checkMesh and patch validation
```

建筑物、地面、入口、出口、侧边界和顶部边界必须具有稳定的名称。加密应集中在建筑边缘、屋脊线、峡谷通道和尾流区域，而不是均匀施加。

### 输出结果

有用的输出包括行人高度的风速、速度矢量、压力、湍动能、屋顶和街区峡谷回流，以及选定建筑补丁的统计信息。报告入口剖面、粗糙度假设、壁面函数、域范围、加密级别和单元数量。

## 7. MotorBike: complex external geometry

### 目的

MotorBike 示例是一个更具挑战性的基于表面的外部气动案例。它测试几何导入、特征提取、表面贴合、局部加密、壁面补丁、力积分和动画制作。

### 模型选择

存储库脚本和 README 包含与版本相关的引用。运行前检查实际脚本：某些配置使用 `simpleFoam`/`incompressibleFluid` 路径，而脚本文档也提到 Spalart–Allmaras RAS 模型。所选求解器、湍流模型、壁面处理和几何来源必须记录在生成的案例中。

Spalart–Allmaras 对于附着或中度分离的外部气动流动有吸引力，因为它相对便宜并求解一个输运的湍流变量。当分离行为和在不利压强梯度下的鲁棒性更重要时，可能更偏好 $k$–$\omega$ SST。两者都没有普遍优越性；网格和验证数据通常主导不确定度。

### 网格与验证

先使用粗网格验证几何方向和补丁名称，然后加密前缘、车轮、整流罩、地面接触和尾流。验证表面无泄漏或法线反转，并且局部单元尺寸支持预期的壁面处理。在力参考量确定后，再比较阻力和压力分布。

## 8. Thermal buoyancy: natural convection

### 目的

thermal-buoyancy 示例模拟一个带有重力、温差和浮力驱动流动的加热房间或腔体。

### Boussinesq approximation

对于适度的温差，可以在连续性和惯性项中忽略密度变化，仅在浮力项中保留。典型的关系为：

$$
\rho\approx\rho_0[1-\beta(T-T_0)],
$$

浮力贡献与 $\rho_0\beta(T-T_0)\mathbf{g}$ 成正比。这比完全可压缩的理想气体处理计算开销更低，但当密度变化很大、压缩性重要或线性近似因温度范围而失效时不应使用。

### 边界条件与诊断量

该案例规定冷热墙、绝热或隔热表面、重力和热湍流模型。监控 $T$, $U$, `p_rgh`, $k$, $\epsilon$ 或 $\omega$, 以及 `alphat`（如适用）。主要无量纲群为 Rayleigh 数、Prandtl 数和 Nusselt 数。若可能，应将温差、环流单元和传热率与基准进行验证。

## 9. Heated duct: conjugate heat transfer

heated-duct 案例在 [CHT case setup](cht_workflow.md) 中有详细文档。它是流体-固体区域创建、区域特定场、材料属性、耦合界面、`chtMultiRegionFoam`、直接或 VTK 后处理以及热平衡报告的参考示例。

## 10. Muffler case study

muffler 案例是一个更大、更面向应用的示例。它演示了 FoamPilot 如何结合几何处理、内部流动建模、压降分析、声学或流体后处理以及报告生成。相关页面为 [Detailed muffler example](example/muffler/detailled_example_muffler.md)。

重要的建模决策包括内部体积和穿孔或连接通道、入口和出口的压强/流量数据、壁面粗糙度假设、可压缩性或不可压缩性，以及如果解释声学量时的频率范围。单独的压力场并不是声学预测；声学假设和采样策略必须记录。

## 11. SimpleCar case study

详细的 SimpleCar 页面补充了可执行的湍流教程。它侧重于基于 JSON 的案例设置、网格配置、OpenFOAM 字典操作、边界条件和自动报告。学习如何使用项目级脚本生成完整案例（而不仅仅是重现小型基准）时请使用它。

## 12. Thermal and geometry add-on examples

存储库还包含围绕几何转换、主动脉表面处理、天气/EPW 输入、风廓线、人体几何、MakeHuman/JOS-3 体温调节 和 CSV 耦合的专业示例和实用程序。这些并不全等同于求解器教程：其中一些是预处理或数据交换工作流。因此它们的文档必须说明输入数据格式、坐标系、外部软件、生成的工件和验证检查。

## 教程产物与可复现性

教程目录可能包含运行脚本、几何文件、残差导出、图像、动画和生成的报告。在调整教程时将生成的结果与源输入分开保存。记录 OpenFOAM 版本、Python 环境、单元数量、求解器设置、收敛准则以及对生成字典所做的任何手动更改。

## 教程不能证明什么

教程演示一个工作流；它并不建立工业精度。准确性需要网格收敛、时间步或库朗数研究、模型敏感性、守恒性检查、与解析或实验数据的比较以及不确定性声明。几何或生理学越复杂，这些检查就越重要。
