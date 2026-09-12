# foampilot 模块的概念和理论概述

`foampilot` 模块被设计为一个面向对象的 Python 层，用于计算流体力学 (CFD) 平台 **OpenFOAM**。它的主要目的是将传统依赖文本字典文件的 OpenFOAM 案例配置复杂性抽象化，提供一个直观且强大的 Python 编程接口（API）。

`foampilot` 的架构与 OpenFOAM 案例结构紧密对应，每个子模块管理模拟的基本方面。

## 1. foampilot.solver：物理和数值核心

`solver` 子模块是模拟的核心控制。它不仅运行 OpenFOAM 求解器，还根据用户定义的物理和数值属性处理 **动态求解器选择**。

| OpenFOAM 概念 | foampilot 中的作用 | 理论说明 |
| :--- | :--- | :--- |
| **求解器** | **`Solver` 类** | 作为 **智能求解器管理器**。通过修改布尔属性（如 `compressible`、`transient`、`is_vof`），它会自动选择对应的 OpenFOAM 求解器，解决所需的物理方程（不可压流的 Navier-Stokes 方程、可压流方程等）。 |
| **物理配置** | **Solver 属性** | 定义问题类型（稳态或瞬态，单相或多相 VOF，有无重力/能量）。这种抽象保证用户只需关注问题的物理，而无需关心具体的 OpenFOAM 求解器名称。 |

## 2. foampilot.constant：物理介质定义

`constant` 子模块管理 OpenFOAM 的 `constant` 目录，包含流体属性和网格信息。

| OpenFOAM 概念 | foampilot 中的作用 | 理论说明 |
| :--- | :--- | :--- |
| **流体属性** | **`transportProperties`, `physicalProperties` 类** | 定义流体的基本属性（运动粘度 $\nu$、密度 $\rho$、比热 $C_p$ 等）。这些属性对于封闭 Navier-Stokes 方程和传输建模至关重要。 |
| **湍流模型** | **`turbulenceProperties` 类** | 管理湍流模型的选择和配置（如 $k-\epsilon$、$k-\omega$ SST）。这些模型通过增加输运方程来模拟湍流效应，封闭 RANS（雷诺平均 Navier-Stokes）方程系统。 |
| **重力** | **`gravityFile` 类** | 可激活并定义重力向量 $\mathbf{g}$，在浮力（Boussinesq）或自由表面流模拟中至关重要。 |

## 3. foampilot.system：数值和时间控制

`system` 子模块管理 OpenFOAM 的 `system` 目录，该目录决定方程的离散和求解方式。

| OpenFOAM 概念 | foampilot 中的作用 | 理论说明 |
| :--- | :--- | :--- |
| **模拟控制** | **`controlDictFile` 类** | 定义时间参数（时间步长 $\Delta t$、起始/结束时间）、结果写入频率，以及执行函数（如自动停止的 `runTimeControl`）。 |
| **数值格式** | **`fvSchemesFile` 类** | 管理方程项的离散（时间导数、对流项、扩散项）。数值格式选择（如时间欧拉法，对流使用 `upwind` 或 `Gauss linear`）直接影响数值稳定性和精度。 |
| **代数求解器** | **`fvSolutionFile` 类** | 配置离散后线性方程系统的矩阵求解器（如压力使用 `PCG`，速度使用 `BiCGStab`）。同时定义收敛准则（容差）和欠松弛策略。 |

## 4. foampilot.mesh：网格生成

`mesh` 子模块负责网格创建，即计算域的空间离散化。

| OpenFOAM 概念 | foampilot 中的作用 | 理论说明 |
| :--- | :--- | :--- |
| **结构化网格** | **`BlockMeshFile` 类** | 封装 OpenFOAM 的 `blockMesh` 工具。通过六面体块定义域几何，是处理简单或参数化几何的高效方法。 |
| **非结构化网格** | **`gmsh_mesher`, `snappymesh` 类** | 处理复杂几何（如 STL 文件）与高级网格工具（`Gmsh`、`snappyHexMesh`）的集成，并生成这些工具所需的配置文件。 |

## 5. foampilot.boundaries：物理边界条件

`boundaries` 子模块用于定义流体与环境的交互。

| OpenFOAM 概念 | foampilot 中的作用 | 理论说明 |
| :--- | :--- | :--- |
| **边界条件** | **`Boundary` 类** | 管理每个物理场 ($\mathbf{U}$, $p$, $k$, $\epsilon$ 等) 在网格 patch 上的边界条件。理论上，这些条件提供缺失边界信息，确保偏微分方程的唯一解。 |
| **条件类型** | **`Boundary` 方法** | 提供常用物理条件方法：`set_velocity_inlet`（速度 Dirichlet），`set_pressure_outlet`（压力 Neumann），`set_wall`（无滑或滑动）等。 |
| **壁面函数** | **自动集成** | 为湍流模型自动应用适当的壁面函数，实现边界层模拟，无需极细壁面网格。 |

## 6. foampilot.postprocess 和 foampilot.report：结果分析

这些子模块管理模拟后的数据提取、分析和展示。

| 子模块 | 概念角色 | 理论说明 |
| :--- | :--- | :--- |
| **`postprocess`** | **可视化和分析** | 使用 **PyVista** 加载 OpenFOAM 结果 (VTK 文件) 并进行常用后处理（切片、等值线、矢量、流线）。可视化物理场并提取派生量（如 Q 准则、涡度）。 |
| **`report`** | **报告生成** | 自动生成结构化模拟报告（如 PDF），整合关键数据（输入参数、收敛残差、后处理图像），保证可追溯性和可重复性。 |

## 7. foampilot.utilities 和 foampilot.commons：跨模块工具

这些模块为整个框架提供支持功能。

| 子模块 | 概念角色 | 理论说明 |
| :--- | :--- | :--- |
| **`utilities.manageunits`** | **单位管理** | 使用 `ValueWithUnit` 类确保输入的量纲一致。这在物理和工程中至关重要，可避免单位转换错误，并使代码独立于使用的单位制（SI、英制等）。 |
| **`utilities.dictonnary`** | **OpenFOAM 字典处理** | 提供创建和操作复杂数据结构的工具，匹配 OpenFOAM 字典文件（如 `topoSetDict`、`createPatchDict`）。 |
| **`commons`** | **通用工具** | 包含类序列化、网格文件读取（`polyMesh/boundary`）及其他底层操作，实现与 OpenFOAM 文件格式接口。 |

本文概述了 `foampilot` 如何将 OpenFOAM 模拟的基本概念结构化并映射到模块化、直观的 Python 架构中。
