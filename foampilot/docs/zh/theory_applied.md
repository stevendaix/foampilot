# 应用理论：血流动力学、户外风环境与体温调节

本章解释为何选择某个模型、它表示哪条定律、需要哪些数据，以及在何种情况下模型会不可靠。内容刻意比求解器配方更详细。一个 CFD case 并非仅由可执行文件名定义；它由几何、守恒定律、本构关系、边界数据、湍流闭合、数值离散和验证策略共同定义。

## 1. 模型选择原则

一个有用的模型不是可用的最复杂模型，而是在实验或应用条件下解析感兴趣量所需的最简模型。选择应由下列问题来证明：

| Question | Consequence |
| --- | --- |
| Is the flow compressible? | Select the density and pressure formulation. |
| Is it steady or transient? | Select a steady RANS, transient RANS, LES, or time-dependent laminar model. |
| Is heat coupled to momentum? | Use passive scalar transport, buoyancy, or a full energy/thermophysical model. |
| Are wall gradients important? | Choose wall resolution, wall functions, and mesh targets consistently. |
| Does viscosity depend on shear rate? | Use Newtonian or non-Newtonian rheology. |
| Are the boundaries measured or idealised? | Use data-driven profiles, tables, or analytical functions and quantify uncertainty. |
| Is the geometry patient-specific or geospatial? | Preserve coordinate systems, units, topology, and provenance. |

模型应始终说明其适用领域（domain of validity）。特别地，一次成功的教程运行并不能为生物医学或环境预测验证物理假设。

# 2. 生物医学 CFD

## 2.1 模型对象是什么？

生物医学 CFD 可指非常不同的问题：动脉内的血流、呼吸道气流、人体周围的换热、医疗器械内的流动，或多孔组织内的传输。守恒定律和边界数据在这些问题间不同。本节关注血管流动以及病人特定几何、血流动力学模型与 FoamPilot 实用工具之间的接口。

对于固定的流体域，基本方程是质量守恒和动量守恒：

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0,
$$

$$
\rho\left(\frac{\partial\mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)
=-\nabla p+\nabla\cdot\boldsymbol{\tau}+\mathbf{f},
$$

其中 $\boldsymbol{\tau}$ 是粘性应力张量，$\mathbf{f}$ 包括体力或模型化源项。

对于不可压缩牛顿流体：

$$
\boldsymbol{\tau}=2\mu\mathbf{D},
\qquad
\mathbf{D}=\frac12\left(\nabla\mathbf{u}+\nabla\mathbf{u}^{T}\right),
$$

具有常数动力粘度 $\mu$。对于血液，这是一个近似，其适用性取决于血管尺寸、剪切率、血细胞比容（haematocrit）和感兴趣输出量。

## 2.2 牛顿型与非牛顿型血液

在大动脉和高剪切区，血液常被视为牛顿流体，因为表观粘度趋于近似常数值。此简化降低了成本并使收敛更容易。在目标是整体压降且区域大部分剪切率较高时，这种简化可辩护。

血液还具有剪切变稀行为：在低剪切率时表观粘度增加，而随着剪切增大而降低。在回流区、动脉瘤、停滞附近、远端血管以及当壁面剪切应力（WSS）或停留时间为主要关注量时，非牛顿模型更为重要。一项关于颅内狭窄模型的比较研究发现，牛顿与非牛顿假设对压力比的影响可能很小，但在低-WSS 区域，尤其在舒张期会产生更明显的差异 [1]。

### 牛顿定律

$$
\mu=\mu_0.
$$

这是最简单的模型。必须说明粘度的单位和温度假设。

### Carreau–Yasuda 定律

一种常见的剪切变稀形式为：

$$
\mu(\dot\gamma)=\mu_\infty+(\mu_0-\mu_\infty)
\left[1+(\lambda\dot\gamma)^a\right]^{(n-1)/a},
$$

其中 $\dot\gamma$ 是剪切率大小，$\mu_0$ 和 $\mu_\infty$ 是极限粘度，$\lambda$ 是时间尺度，$a$ 控制过渡，且 $n<1$ 导致剪切变稀。

### Casson 定律

Casson 模型是另一种用于血液的经验流变学模型：

$$
\sqrt{\tau}=\sqrt{\tau_y}+\sqrt{\mu_c\dot\gamma},
$$

其中 $\tau_y$ 是类屈服应力参数，$\mu_c$ 控制高剪切行为。低剪切率处的确切正则化和实现方式很重要。

### 如何选择

| Quantity of interest | First model to test | Additional sensitivity study |
| --- | --- | --- |
| Bulk flow rate or rough pressure ratio | Newtonian | Carreau–Yasuda or Casson if low-shear regions matter. |
| Wall shear stress | Newtonian and non-Newtonian comparison | Report low-WSS and oscillatory-WSS sensitivity. |
| Residence time or thrombosis-related indicator | Non-Newtonian candidate | Check rheological parameters and near-wall resolution. |
| Large artery with high shear | Newtonian may be sufficient | Verify against a non-Newtonian run. |
| Small vessel or strong recirculation | Non-Newtonian is more defensible | Include diameter, haematocrit, temperature, and patient variability. |

FoamPilot 的基础 `transportProperties` 路径天然适合常属性问题。非牛顿定律需要一个实际根据局部剪切率评估粘度的求解器和字典配置。仅分配一个描述性 Python 变量而不核查生成的 OpenFOAM 字典不会激活流变模型。

## 2.3 脉动性与 Womersley 数

血流通常是脉动的。Womersley 数比较了非稳惯性与粘性扩散：

$$
\alpha=R\sqrt{\frac{\omega\rho}{\mu}},
$$

其中 $R$ 是血管半径，$\omega$ 是心动波形的角频率。低 $\alpha$ 导致更接近准稳抛物线型的流型。更高的 $\alpha$ 导致流芯更平坦并在压力梯度与壁面响应之间产生更强的相位滞后。

对于脉动模拟，入口应由测量或合成流量波形表示。必须使用实际入口面积将波形转换为速度剖面，并在可能时采用发展或与 Womersley 一致的剖面。在急剧弯曲或分支动脉的入口处使用均匀速度会产生人为的入口效应，污染感兴趣区域。

## 2.4 血管 CFD 的边界条件

在生物医学中，最重要的不确定性往往是边界条件，而非内部离散化。

| Boundary | Common data | Physical issue |
| --- | --- | --- |
| Inlet | Flow rate, velocity profile, pressure, or patient waveform | Measured plane may be far from the computational inlet. |
| Outlet | Fixed pressure, traction, resistance, impedance, or Windkessel | Downstream vasculature is truncated. |
| Wall | No-slip rigid wall, moving wall, or fluid-structure coupling | Wall compliance can change pressure and WSS. |
| Branch | Flow split or pressure relation | Patient-specific downstream resistance is uncertain. |

### Windkessel 出口模型

Windkessel 模型表示截断出口下游血管的阻力和顺应性。常见的三元件模型结合了近端电阻 $R_1$、顺应 $C$ 和远端电阻 $R_2$。在压强-流量形式下：

$$
C\frac{dP_c}{dt}=Q-\frac{P_c-P_d}{R_2},
$$

$$
P=P_c+R_1Q,
$$

其中 $Q$ 是出口流量，$P$ 是出口压强，$P_c$ 是电容器压强，$P_d$ 是远端参考压强。选择该模型的原因是固定压强出口无法重现下游网络的蓄能与延迟响应。

FoamPilot 将 `WindkesselModel` 作为模型附加项暴露。在使用前，需定义符号约定、单位、初始电容压强、压强参考和耦合时间步。根据测量的压-流数据或文献生理假设对 $R_1$, $R_2$, 和 $C$ 进行校准。Windkessel 模型是一个降阶的边界表示；它并不是完整的心血管循环模型。

## 2.5 病人特定几何与数据溯源

生物医学 case 通常始于 CTA、MRI、CT、NIfTI、STL、VTP 或其他分割后的表面。流程应记录：

1. 成像模态、分辨率、采集日期和方向；
2. 分割方法和阈值决策；
3. 平滑与闭孔操作；
4. 入口/出口延长长度；
5. 表面重网格容差和三角形数量；
6. 转换为米制单位；
7. 分支标签和 patch 名称；
8. 网格质量与体积守恒；
9. 边界条件来源与校准；
10. 匿名化和数据治理。

FoamPilot 实用工具包括 NIfTI-to-STL 和血管表面清理辅助工具。这些是几何处理工具，而非临床分割验证器。求解前请对输出进行目视和定量检查。

## 2.6 生物医学验证量

下列量常被报告：

- 压降和病变间压比；
- 时均壁面剪切应力；
- 振荡剪切指数；
- 相对停留时间；
- 回流体积；
- 分支处的流量分配；
- 收缩期峰值和舒张末值；
- 出口处的流量守恒。

在没有网格收敛与近壁研究的情况下，不要单独解读某个局部 WSS 峰值。WSS 是壁处的导数，对表面平滑、网格间距、时间分辨率与流变学尤其敏感。

# 3. 户外风与大气边界层

## 3.1 为何均匀入口常常不正确

建筑或城市流动模拟并非仅仅把汽车案例竖直旋转。近地面处，平均风速随高度增加且湍流随高度变化。建筑扰动入射的大气边界层，产生角落加速、屋顶分离、街峡回流和尾流等现象。

均匀入口对于受控风洞或简化方法研究可接受。但除非域和边界条件被有意构造以使剖面在感兴趣区域之前发展，否则它通常不符合大气边界层。

## 3.2 控制方程

对于低速户外风，空气通常被视为不可压缩：

$$
\nabla\cdot\mathbf{U}=0,
$$

$$
\frac{\partial\mathbf{U}}{\partial t}+\mathbf{U}\cdot\nabla\mathbf{U}
=-\frac{1}{\rho}\nabla p+\nabla\cdot[(\nu+\nu_t)\nabla\mathbf{U}],
$$

湍流粘性 $\nu_t$ 由诸如 $k$–$\epsilon$、realizable $k$–$\epsilon$、RNG $k$–$\epsilon$ 或 $k$–$\omega$ SST 等闭合模型提供。

正确选择取决于输出。稳态 RANS 对平均风速和平均压强是高效的。当相干非稳性重要时需要 URANS。当瞬态涡旋和湍流波动为主要输出时，LES 或混合方法更合适，但其网格和时间步代价高得多。

## 3.3 对数风速定律

中性大气边界层常用对数律近似：

$$
U(z)=\frac{u_*}{\kappa}\ln\left(\frac{z+z_0}{z_0}\right),
$$

其中 $u_*$ 是摩擦速度，$\kappa\approx0.4$ 是 von Kármán 常数，$z_0$ 是空气动力学粗糙度长度。如果参考风速已知于高度 $z_r$：

$$
U(z)=U(z_r)\frac{\ln[(z+z_0)/z_0]}{\ln[(z_r+z_0)/z_0]}.
$$

对数律之所以被选用，是因为在中性、水平均匀的假设下它代表表面层的平均速度。它并不自动代表热分层、林冠、复杂地形或强瞬态天气。

OpenFOAM 提供基于对数律型剖面和湍流量的边界条件。其文档描述了 `atmBoundaryLayerInletVelocity`, `atmBoundaryLayerInletK`, `atmBoundaryLayerInletEpsilon`, 和 `atmBoundaryLayerInletOmega`，以及大气壁面函数和源项 [2]。

## 3.4 幂律剖面

工程替代是：

$$
U(z)=U_r\left(\frac{z}{z_r}\right)^\alpha,
$$

其中 $\alpha$ 是经验剪切指数。当风数据在两个高度可用或风工程标准提供指数时，幂律方便使用。它与粗糙度长度的直接联系不如对数律，应避免在未说明换算关系时混用粗糙度长度。

## 3.5 湍流入口数据

入口必须定义的不仅是速度，还有湍流。对于 $k$–$\epsilon$ 模型，一个常见估计是：

$$
 k=\frac32(UI)^2,
$$

其中 $I$ 是湍流强度。可以使用长度尺度 $L$ 来估算：

$$
\epsilon=C_\mu^{3/4}\frac{k^{3/2}}{L},
$$

对于 $k$–$\omega$ 模型：

$$
\omega\approx\frac{\sqrt{k}}{C_\mu^{1/4}L}.
$$

这些公式是建模假设，而非测量。$U$, $k$, $\epsilon$, 或 $\omega$ 的剖面应相互兼容；否则大气边界层可能在到达建筑物前漂移、加速或衰减。

## 3.6 稳定性与浮力

中性流忽略热分层。稳定或不稳定的大气条件需要温度、浮力和湍流产生的假设。浮力项的符号和大小影响垂直混合、尾流恢复与行人高度处的风。

对于简化的城市热例，若温差较小可使用 Boussinesq 近似。对于更大的分层或密度变化，应采用可压缩或可变密度模型。选择必须与可用的天气数据和求解器的热物性表述一致。

## 3.7 计算域与壁面建模

户外域应提供足够的上游 fetch、下游尾流长度、侧面间距和顶部间隙。地面不是另一个普通壁面：其粗糙度决定速度剖面和湍流产生。壁面函数、粗糙度参数、首单元高度和大气入口剖面必须作为一个系统被选择。

使用特定定律或边界条件的主要原因是“平衡一致性”（equilibrium consistency）。如果入口剖面暗含一种粗糙度而地面壁面又暗含另一种，剖面会人工演化。因此首要任务是在加入建筑物前验证一个前置或空域案例。

## 3.8 城市输出

相关输出包括行人高度处的平均风速、若有瞬态数据则的超出概率、立面压强、风舒适指标、屋顶加速度、街峡环流、湍流强度，以及当耦合标量方程时的污染物传输。

风场结果应始终说明参考高度、粗糙度、风向、大气稳定性、湍流模型、计算域尺寸、网格数量、壁面处理和平均区间。

# 4. 体温调节（Human thermoregulation）

## 4.1 耦合层级

体温调节可在若干层级上表示：

| Level | Description | Suitable use |
| --- | --- | --- |
| Convective boundary condition | Prescribed heat-transfer coefficient or skin temperature. | Simple thermal CFD around a body. |
| Multi-node physiology | Core, blood, muscle, fat, and skin temperatures with regulatory responses. | Coupling CFD environment with human thermal response. |
| Detailed local physiology | Segment-level metabolism, perfusion, sweating, clothing, radiation, and posture. | Research studies requiring local response. |
| Fully coupled human-fluid model | Physiological state changes alter surface fluxes and flow conditions. | Advanced research; requires careful time coupling and validation. |

FoamPilot 的 MakeHuman/JOS-3 工作流属于几何加生理耦合层级。MakeHuman 提供人体表面；JOS-3 提供多结点热响应；OpenFOAM 解析周围的流动和热传。

## 4.2 JOS-3 模型概念

JOS-3 是一个数值人体体温调节模型，预测诸如核心温度、皮肤温度、出汗、血流与 17 个体段及整个人体的热响应等量 [3] [4]。它衍生自早期多结点模型，并使用组织隔室与调节信号的生理网络。

该模型包含通过体组织的热储存与传递、血液灌注、代谢产生、呼吸损失、传导、对流、辐射与蒸发。调节响应可以包括血管舒张、血管收缩、出汗、发抖、非颤抖产热，以及与活动或姿势相关的变化。

该模型应被视为一个集总或多结点生理模型，而不是解析的血管 CFD 模型。CFD 场可提供局部空气温度、风速、与湿度相关的输入和辐射条件，而 JOS-3 返回分段级皮肤温度与耗热信号。

## 4.3 人体热平衡

一个简化的人体热平衡为：

$$
M-W=Q_{sk}+Q_{res}+S,
$$

其中 $M$ 是代谢热产生，$W$ 是外部功，$Q_{sk}$ 是皮肤总散热，$Q_{res}$ 是呼吸散热，$S$ 是体内热储存。皮肤散热可分解为：

$$
Q_{sk}=Q_{conv}+Q_{rad}+Q_{cond}+Q_{evap}.
$$

CFD 模型解析或近似对流传热。辐射可用辐射求解器建模或由平均辐射温度表示。蒸发依赖于湿度、服装、皮肤湿润度和蒸汽压差；它并非仅由空气速度决定。

## 4.4 为何局部 17 区数据重要

单一的平均体温掩盖了局部暴露。一个人可能同时面部偏热、手被冷却、躯干保温并有不对称气流。JOS-3 接受 17 个体段的局部环境与服装值。FoamPilot 的几何工作流创建相应的表面 patch 和一个 `zone_mapping.csv`，以便 CFD 结果能够一致地被聚合。

映射必须记录：

- 精确的体部名称和顺序；
- STL 或 OpenFOAM 网格中的表面 patch 名称；
- 每个 patch 所代表的面积；
- 某个 patch 是否暴露、着衣或被遮挡；
- 局部速度、温度和辐射的平均方法；
- 热通量的符号约定；
- CFD 与生理之间的时间插值。

## 4.5 围绕人体的对流定律

对流热通量常写为：

$$
q''_{conv}=h_c(T_{skin}-T_a),
$$

其中 $h_c$ 是局部对流换热系数，$T_a$ 是空气温度。在 CFD 耦合中，可从解析到的壁面热通量估计 $h_c$：

$$
 h_c=\frac{q''_{conv}}{T_{skin}-T_a},
$$

或从基于局部速度、特征长度和朝向的经验相关式得到。CFD 得到的路径空间分辨率更高，但依赖于网格质量、壁面处理、表面温度边界条件和湍流建模。

对于简化相关式，Nusselt 数可能依赖于雷诺数和普朗特数：

$$
Nu=\frac{h_cL}{k_a}=f(Re_L,Pr),
$$

其中

$$
Re_L=\frac{U L}{\nu_a},
\qquad
Pr=\frac{\nu_a}{\alpha_a}.
$$

应明确选择相关式还是 CFD。相关式成本低且对初步设计有用；当出现分离、回流、姿势、服装几何或空间不对称时，CFD 更有用。

## 4.6 辐射与平均辐射温度

辐射不等同于空气温度。人体可以处在凉爽空气中，但同时从周围表面接收强的长波或太阳辐射。因此实用的生理耦合会提供空气温度 $T_a$、平均辐射温度 $T_r$、空气速度 $V_a$、相对湿度、服装绝热、活动水平与姿势。

如果 CFD case 不求解辐射，应使用有文档记录的 $T_r$ 输入，而不是默默地将 $T_r=T_a$。如果太阳负荷重要，应区分短波太阳吸收与长波交换并记录表面发射率和吸收率。

## 4.7 CFD 与 JOS-3 之间的数据交换

一个稳健的耦合循环为：

```text
MakeHuman surface
→ surface cleanup and JOS-3 patch generation
→ CFD mesh and patch mapping
→ OpenFOAM temperature/velocity/radiation solution
→ area-weighted segment averages
→ JOS-3 physiological update
→ updated skin temperature or heat-flux boundary data
→ next CFD interval
```

耦合可以是单向或双向：

| Coupling | CFD receives | Physiology receives | Use |
| --- | --- | --- | --- |
| One-way | Fixed skin temperature or heat flux | CFD air conditions | Initial feasibility study. |
| Loose two-way | Updated segment skin temperature or flux | Local CFD temperature, speed, radiation, humidity proxy | Practical transient coupling. |
| Strong two-way | Iterated thermal boundary condition within each timestep | Converged local environmental state | Expensive research coupling. |

时间步必须同时解析 CFD 的瞬态与生理响应。生理更新每个 CFD 迭代可能没有必要；非常大的耦合间隔可能错过快速暴露变化。将耦合间隔作为数值参数进行测试。

## 4.8 体温调节的验证

在评判耦合系统之前，应分别比较生理模型和周围的 CFD 模型以进行验证。对于 CFD 侧，验证速度、温度、壁面通量与网格收敛。对于生理侧，验证基线皮肤/核心温度、代谢响应、出汗、血流，以及对受控热暴露的预期响应。

诸如平均皮肤温度之类的体温调节输出是带有生理不确定性的模型预测。未经实验比较，不应将其作为临床诊断或已验证的人体反应来呈现。

## 参考文献

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8450390/ "Liu et al., Comparison of Newtonian and Non-newtonian Fluid Models in Blood Flow Simulation"

[2]: https://www.openfoam.com/news/main-news/openfoam-v20-06/boundary-conditions "OpenFOAM: atmospheric boundary-layer boundary conditions"

[3]: https://github.com/TanabeLab/JOS-3 "TanabeLab/JOS-3: Joint system thermoregulation model"

[4]: https://doi.org/10.1016/j.enbuild.2020.110575 "Takahashi et al., Thermoregulation model JOS-3 with new open source code"
