# 共轭传热：数据设置、执行与验证

共轭传热 (CHT) 将固体中的热导与一个或多个流体区域中的热传输耦合在一起。FoamPilot 提供用于区域定义、热物性字典、温度场、接口、控制字典、执行和热后处理的 Python 对象。OpenFOAM 仍负责求解耦合的有限体积方程。

## 1. 物理问题

一个 CHT 工况至少包含一个流体区域和一个固体区域。在流体中，求解器解算动量、连续性和能量方程。在固体中，解算热传导；固体中不存在流体速度场。

对于可压缩流体，能量方程使用所选的焓或内能形式以热力学一致的方式书写。简化的温度形式为：

$$
\rho c_p\left(\frac{\partial T}{\partial t}+\mathbf{U}\cdot\nabla T\right)
=\nabla\cdot(k_f\nabla T)+S_T,
$$

其中 $\rho$ 为密度，$c_p$ 为比热容，$k_f$ 为流体导热率，$S_T$ 表示源项。在静止固体中：

$$
\rho_s c_{p,s}\frac{\partial T_s}{\partial t}
=\nabla\cdot(k_s\nabla T_s)+S_s.
$$

在理想的流体-固体界面：

$$
T_f=T_s,
$$

且

$$
-k_f\nabla T_f\cdot\mathbf{n}
=-k_s\nabla T_s\cdot\mathbf{n}.
$$

第一个条件表示温度连续性。第二个条件表示热通量连续性。如果接触热阻、辐射、粗糙度或薄涂层在物理上重要，则必须改变接口模型以显式考虑它们，而不是在网格调整中隐蔽处理。

## 2. FoamPilot 中的数据模型

CHT API 围绕四个概念组织：

| Object | Role |
| --- | --- |
| `FluidRegion` | 流体区域的初始温度和速度、湍流模型、热模型、状态方程、混合物和传输模型。 |
| `SolidRegion` | 导热区域的初始温度和固体材料属性。 |
| `CoupledInterface` | 流体补丁与固体补丁的配对以及接口字典。 |
| `ChtSolver` | 工况路径、求解器可执行文件、区域、接口、区域到求解器的映射、设置以及串行/并行执行。 |

最小设置示例如下：

```python
from foampilot.cht import (
    ChtSolver,
    FluidRegion,
    SolidRegion,
    CoupledInterface,
)

fluid = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

solid = SolidRegion(
    name="solid",
    temperature=350.0,
)

interface = CoupledInterface(
    name="fluid_to_solid",
    fluid_region="fluid",
    solid_region="solid",
)

solver = ChtSolver(
    case_path="case",
    solver_name="chtMultiRegionFoam",
    regions=[fluid, solid],
    interfaces=[interface],
)
solver.setup_case()
solver.run_simulation(nb_proc=1)
```

必须为目标问题填写具体的材料参数。不要使用作为验证示例的默认值作为真实材料数据。

## 3. 工况目录与区域数据

多区域工况不是在单区域工况上简单增加一个温度场。目录布局携带物理含义：

```text
case/
├── 0/
│   ├── fluid/
│   │   ├── T
│   │   ├── U
│   │   ├── p
│   │   ├── p_rgh
│   │   └── turbulence fields
│   └── solid/
│       └── T
├── constant/
│   ├── fluid/
│   │   ├── thermophysicalProperties
│   │   └── turbulenceProperties
│   ├── solid/
│   │   ├── thermophysicalProperties
│   │   └── transportProperties
│   └── regionInterfaces/
│       └── fluid_to_solid.dict
└── system/
    ├── controlDict
    ├── regionProperties or region solver mapping
    ├── fvSchemes
    ├── fvSolution
    └── createZonesDict
```

具体的字典名称随 OpenFOAM 版本和求解器家族而异。仓库中的 `09_CHT_heatedDuct` 教程是其目标版本的参考。始终检查生成的字典，因为 `foamSetupCHT`、`splitMeshRegions` 和区域特定约定在不同版本间存在差异。

## 4. 几何与区域定义

heated-duct 教程使用背景网格和单元区 (cell-zone) 定义来区分流体与受热固体。概念性顺序为：

```text
blockMesh
→ createZones
→ splitMeshRegions -cellZones
→ create region dictionaries and fields
→ run chtMultiRegionFoam
```

单元区定义必须覆盖预期的固体体积，不得有间隙或重叠。失败的区域划分可能会创建一个看似包含固体目录但并不代表预期物理域的工况。

使用以下命令验证区域拓扑：

```bash
checkMesh -case case -region fluid
checkMesh -case case -region solid
splitMeshRegions -cellZones -overwrite -case case
```

使用与已安装 OpenFOAM 发行版相匹配的命令，并且不要在唯一的源工况副本上执行破坏性的 `-overwrite` 操作。

## 5. 流体区域数据

流体数据必须指定足够的信息以闭合动量和能量方程：

| Data | Meaning | Typical choices |
| --- | --- | --- |
| Density law | Relation between density, pressure, and temperature. | Perfect gas, constant density, Boussinesq where supported. |
| Heat capacity | Energy storage. | Constant or temperature-dependent $c_p$. |
| Conductivity | Molecular heat diffusion. | Constant or temperature-dependent $k_f$. |
| Viscosity | Momentum diffusion. | Constant or temperature-dependent dynamic viscosity. |
| Equation of state | Pressure-density relation. | Ideal gas or incompressible approximation. |
| Turbulence | Closure for unresolved flow. | Laminar, $k$–$\epsilon$, $k$–$\omega$ SST, or version-specific RAS model. |
| Initial velocity | Starting flow field. | Zero or prescribed bulk velocity. |
| Initial temperature | Starting thermal field. | Inlet temperature, wall temperature, or a uniform estimate. |

对于空气管道，流体区域通常在 CHT 中被视为可压缩，因为温度会影响密度和能量。如果温度范围较小且所用求解器支持，则不可压缩的热力学表述可能更合适，但必须有物理理由来支持该简化。

## 6. 固体区域数据

固体区域至少需要：

- 密度 $\rho_s$；
- 比热容 $c_{p,s}$；
- 热导率 $k_s$；
- 初始温度；
- 外部和接口的热边界条件。

对于铜壁，高导热率会导致固体中相对于低导热绝热层较小的温度梯度。但这并不意味着可以移除固体：其厚度和热阻仍然决定热通量。

如果求解器和生成的字典支持，固体模型可以扩展为温度依赖属性、辐射、热源或各向异性热导率。

## 7. 边界条件数据

必须为每个区域分别定义边界条件。常见的流体边界条件有：

| Boundary | Fluid fields |
| --- | --- |
| Inlet | Velocity, temperature, pressure, turbulence quantities. |
| Outlet | Pressure and compatible velocity/temperature outflow conditions. |
| Fluid-solid interface | Coupled temperature and heat-flux condition. |
| Symmetry | `symmetryPlane` for all compatible fields. |
| Adiabatic wall | Zero heat flux, with no-slip or the selected wall velocity treatment. |

常见的固体边界条件有：

| Boundary | Solid field |
| --- | --- |
| Coupled interface | Temperature and heat flux coupled to the fluid region. |
| Fixed-temperature wall | Prescribed `T`. |
| Heat-flux wall | Prescribed normal heat flux. |
| Adiabatic wall | Zero gradient or solver-specific insulated condition. |
| Radiation wall | Radiation-coupled condition when radiation is included. |

FoamPilot 暴露了诸如 `get_coupled_temperature_bc`、`get_fixed_temperature_bc`、`get_heat_flux_bc`、`get_inlet_outlet_bc` 和 `get_symmetry_bc` 等工厂函数。生成的条件仍需针对已安装的 OpenFOAM 版本进行检查。

（注：在中文技术语境中，将“boundary conditions”一致译为“边界条件”。）

## 8. 求解器执行

CHT 求解器在常见的 OpenFOAM 工作流中是一个独立的多区域可执行文件。FoamPilot 的串行路径使用工况路径启动求解器。并行路径对所有区域进行分解，运行 MPI，并重构所有区域。

```python
solver.run_simulation(
    nb_proc=1,
    log_filename="log.chtMultiRegionFoam",
)
```

并行执行示例：

```python
solver.run_simulation(nb_proc=8)
```

在并行运行之前，确认 MPI、`decomposePar` 和 `reconstructPar` 与 OpenFOAM 构建兼容。保存串行基线结果，因为它对于区分分解问题与物理或数值问题是必要的。

## 9. 收敛性与守恒

只有在满足若干条件时，CHT 运行才能被视为收敛：

1. 残差收敛至选定的容差；
2. 区域温度和热通量达到稳定趋势；
3. 接口温度连续性在可接受范围内；
4. 耦合系统的总热量进出保持平衡；
5. 对时间步长合理减小或迭代次数增加，结果不敏感；
6. 感兴趣的工程量保持稳定。

`foampilot.cht.postprocess` 辅助工具包含区域热通量、接口热通量、努塞尔数、热边界层厚度、换热系数、总热平衡、温度等值线和热阻的计算。

## 10. heated-duct 示例数据

仓库教程使用初始接近 300 K 的流体区域和初始接近 350 K 的受热固体区域。其输入数据包括 `block_mesh.json`、一个 Python 工况构建器、一个后处理脚本、区域特定的场文件和材料字典。期望的输出包括区域温度统计、流体温度剖面、组合剖面以及流体/固体温度等值线。

教程中的数值用于演示。对于真实的管道，请用测量或设计值替换以下项目：

- 质量流量或入口速度；
- 入口温度与压力；
- 壁厚与材料；
- 外部热源或壁面温度；
- 流体成分和属性关联式；
- 接触热阻（如适用）；
- 足够的域长以满足热发展区要求。

## 11. 换热解读

局部对流换热系数常定义为：

$$
 h=\frac{q''}{T_w-T_b},
$$

其中 $q''$ 是壁面热通量，$T_w$ 是壁面温度，$T_b$ 是适当选取的流体平均温度。努塞尔数定义为：

$$
 Nu=\frac{hL}{k_f}.
$$

特征长度 $L$ 必须说明：液压直径、管道高度、距入口的局部距离或其他物理有意义的尺度。缺少长度尺度与边界条件的努塞尔数是不完整的。

## 12. 常见 CHT 失败模式

| Symptom | Likely cause |
| --- | --- |
| Missing region fields | Region directories were not created or the setup was interrupted. |
| Solver cannot find a material property | Region-specific `thermophysicalProperties` is incomplete or named differently for the OpenFOAM release. |
| Interface temperature jumps | Wrong patch pairing, non-conformal mapping, or incompatible interface conditions. |
| Heat balance is not closed | Insufficient convergence, incorrect flux sign, missing boundary heat loss, or an unintended source. |
| Fluid temperature is unstable | Time step, energy relaxation, thermo model, or boundary condition is inconsistent. |
| Solid temperature is uniform when it should not be | Conductivity, thickness, source, or region geometry is wrong; check the generated solid mesh. |
| Parallel case differs from serial case | Decomposition, reconstruction, processor boundary treatment, or insufficient convergence. |

## 参考文献

[1]: https://doc.openfoam.com/2306/tools/processing/solvers/rtm/heat-transfer/chtMultiRegionFoam/ "OpenFOAM 文档: chtMultiRegionFoam"

[2]: https://openfoamwiki.net/index.php/Getting_started_with_chtMultiRegionSimpleFoam_-_planeWall2D "OpenFOAM Wiki: multi-region 换热入门"
