# 网格案例与网格策略

网格并不是可以与物理过程分离的预处理细节。它决定可以解析哪些梯度，哪些壁面处理有效，力和热通量如何被积分，以及多少数值扩散进入解。FoamPilot 协调若干网格策略，但用户仍需负责选择几何表示、加密目标、补丁拓扑和质量准则。

## 网格策略选择

| Geometry or objective | Recommended route | Why |
| --- | --- | --- |
| Rectangular cavity, channel, duct, or 2-D benchmark | `blockMesh` | Explicit topology, predictable cells, excellent for verification. |
| Structured multi-block geometry | `classy_blocks` / `blockMesh` | Strong control of grading, blocks, arcs, and named patches. |
| CAD solid or STEP geometry | Gmsh | Flexible unstructured surface/volume meshing and CAD import. |
| STL, OBJ, building, vehicle, or biological surface | Background `blockMesh` + `snappyHexMesh` | Local refinement and snapping around complex triangulated surfaces. |
| Existing OpenFOAM mesh | Direct mesh reader/exporter | Avoids remeshing and enables a Python-controlled post-processing path. |
| Large urban data set | Urban readers + simplification + Gmsh/surface builder | Controls geometry complexity, metric coordinates, and cell budget. |
| CHT fluid/solid case | `blockMesh` or Gmsh + cell zones + region splitting | The mesh must represent both regions and their coupled interface. |

## 1. 结构化 `blockMesh` 情况

`blockMesh` 是验证案例的首选路径，因为其拓扑是显式的。用户控制顶点、块、边、网格分级和边界补丁（patch）。这使其适合腔体、标量通道、热浮力室、后向分离台阶和加热管道的背景网格。

一个结构化案例应定义：

1. 坐标系和尺寸；
2. 块连接性和单元数；
3. 每个方向的分级比（grading ratio）；
4. 补丁名称和 OpenFOAM 补丁类型；
5. 尺度尺寸（dimensional scale）；
6. 目标壁面分辨率和对称假设。

主要风险是顶点排序错误、面法线不一致、过度分级以及补丁与边界条件代码不匹配。在撰写案例其余部分之前，先运行 `blockMesh` 和 `checkMesh`。

## 2. `classy_blocks` 和多块（multi-block）几何

当几何自然由圆柱、挤出、环、弯头或链式块组装时，`classy_blocks` 很有用。FoamPilot 用户指南演示了形状构建、链式连接、扩展、填充、定向切分和补丁分配。

优点是几何控制强；缺点是用户必须理解块如何相接以及单元分级如何在块接口上传播。将其用于拓扑已知的几何；不要用它来掩盖对 CAD 表面理解不足的问题。

## 3. Gmsh 情况

Gmsh 适用于 STEP/IGES/CAD 类几何以及倾向于使用非结构四面体或混合网格的域。一个 Gmsh 案例必须记录：

| Input | Required decision |
| --- | --- |
| CAD units | Confirm whether the source is in metres, millimetres, or another unit system. |
| Physical groups | Define inlet, outlet, walls, symmetry, interfaces, and solid regions explicitly. |
| Element order | Choose linear or higher-order elements consistently with the solver pipeline. |
| Surface quality | Remove duplicate, self-intersecting, or badly oriented faces. |
| Volume closure | Confirm that each fluid or solid volume is watertight. |
| Conversion | Check how the generated mesh is converted to OpenFOAM and how patch names survive. |

Gmsh 的加密应由物理过程驱动：狭窄缝隙、高曲率表面、分离边缘、热界面和边界层需要比均匀区域更多的单元。

## 4. `snappyHexMesh` 情况

标准的复杂几何序列是：

```text
background blockMesh
→ surfaceFeatureExtract
→ castellatedMesh
→ snap
→ addLayers (optional)
→ checkMesh
```

背景网格定义外域。表面几何放置在 `constant/triSurface` 或配置的几何目录下。`snappyHexMesh` 会根据几何相交删除或加密单元，将点捕捉到表面，并可以添加棱柱层。

### 加密区域

在以下区域使用局部加密：

- 前缘和后缘；
- 建筑角落和屋脊线；
- 车辆车轮、整流罩和底盘缝隙；
- 钝体后方的尾迹区域；
- 热界面和狭窄流体通道；
- 医学狭窄、动脉瘤颈、分叉以及进出口延伸。

加密等级必须与湍流模型和壁面处理相平衡。细致的表面网格但边界层未充分解析，并不自动成为好的 CFD 网格。

### 表面与特征检查

在运行复杂案例之前，在查看器中检查表面并核查：

| Check | Typical consequence if it fails |
| --- | --- |
| Closed and orientable surface | Leaks, missing cells, incorrect inside/outside classification. |
| Consistent scale | Geometry is too large or too small relative to velocity and viscosity. |
| Feature extraction | Sharp edges are rounded or patches are merged unexpectedly. |
| Patch names | Boundary conditions are applied to the wrong surface. |
| Surface normals | Wall orientation or flux signs are incorrect. |
| Layer feasibility | Prism layers collapse or create non-orthogonal cells. |

如果检查失败，典型后果包括泄漏、缺失单元、内外部分类错误、特征被平滑或补丁意外合并、补丁名称错误导致边界条件应用错误、表面法线错误导致壁面方向或通量符号错误，以及棱柱层不可行导致坍塌或产生非正交单元。

## 5. 城市与大气网格

城市 CFD 在进入 OpenFOAM 网格化之前需要地理空间处理阶段。将数据转换为公制坐标系，定义风向参考系，移除无关对象，简化建筑平面，分配高度，并建立地形与域边界。urban 包含建筑、道路、地形、CFD 域、几何简化、清理、网格尺寸、尾迹加密、边界层、补丁分配和验证的模型。

网格域的尺寸应由来流的大气边界层和下游尾迹来证明。域太短会将压力和湍流扰动再循环到感兴趣区域；域横向过小会限制风场并夸大阻挡效应。

## 6. 生物医学表面与体积网格

生物医学网格需要额外注意，因为几何是病人特异性的，并且感兴趣的量常依赖导数：壁面剪切应力（WSS）、压降、停留时间或热传递。工作流程通常包括图像分割或表面导入、清理、孔洞封闭、受控公差的平滑、进出口延伸、表面重网格、体积网格生成以及在适当情况下的边界层加密。

几何处理操作绝不应仅描述为“清理”。记录所用算法、公差、目标边长、三角形数、平滑迭代次数，以及该操作是否改变腔体体积或分支直径。将最终网格与原始影像导出的表面进行验证。

对于血流动力学，需在高曲率、狭窄、分叉、回流和预计高壁面剪切梯度的区域加密。足够延长出口以减少人工边界条件对感兴趣区域的影响。

## 7. CHT 网格与区域界面

CHT 网格必须区分流体单元与固体单元，并且必须保持共形或其他正确耦合的界面。教程使用背景网格和 cell zones 定义，然后在将案例拆分为 `fluid` 和 `solid` 区域之前进行区分。

界面需要：

- 匹配的或正确映射的面；
- 区域特定的温度场；
- 每个区域的热物性参数；
- 耦合的温度与热通量边界条件；
- 一致的法线方向和界面命名约定；
- 在热边界层和固体传导路径上足够的分辨率。

最小单元尺寸应同时由动量和热梯度来证明。一个网格可以解析速度但欠解析温度，或反之。使用热边界层估算和局部 Prandtl 数来指导初始网格，然后进行加密研究。

## 8. 网格质量指标

`checkMesh` 是必要的但不足够。至少报告以下指标：

| Indicator | Interpretation |
| --- | --- |
| Non-orthogonality | Large values increase discretisation error and may require correction or a different mesh. |
| Skewness | High skewness degrades gradient and flux reconstruction. |
| Aspect ratio | High ratios can be valid in boundary layers but harmful in poorly aligned regions. |
| Volume ratio | Abrupt cell-size changes can produce numerical stiffness. |
| Negative or zero volume | Invalid mesh; stop before solving. |
| Boundary-layer count | Determines whether the wall model or low-Re treatment is appropriate. |
| $y^+$ distribution | Must be compatible with the selected wall treatment. |
| Cell count by region | Important for CHT balances and parallel decomposition. |

## 9. 壁面分辨率与 $y^+$

目标 $y^+$ 取决于壁面处理。低-Re 方法旨在解析黏性亚层，通常 $y^+$ 接近 1。壁面函数方法将第一层单元放在对数区，并需要与特定壁面函数和湍流模型一致的目标范围。确切目标并非通用。

使用：

$$
 y^+ = \frac{u_\tau y}{\nu},
$$

其中 $u_\tau=\sqrt{\tau_w/\rho}$ 为摩擦速度，$y$ 为从壁面到单元中心的距离。由于 $u_\tau$ 初始未知，可用平板或管流相关式估算，创建初步网格，运行案例，然后检查实际的 $y^+$ 场。

## 10. 网格收敛性协议

一个可辩护的网格研究每次只改变一个分辨率参数，并比较重要的工程输出：压降、阻力系数、再附着长度、传热系数、Nusselt 数、WSS 或标量混合指标。比较全局量和局部剖面。较小的残差并不能证明网格独立。

对于瞬态案例，单独进行时间步长研究。对于多相流，监控界面分辨率和体积守恒。对于 CHT，将总热平衡和界面温度连续性纳入收敛判据。
