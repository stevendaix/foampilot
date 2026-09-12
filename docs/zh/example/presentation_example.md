# FOAMPilot 示例

FOAMPilot 是一个 Python 库，旨在简化 OpenFOAM 仿真的创建、配置和执行。它提供模块化且直观的方法来管理 CFD 工程案例、网格生成、边界条件、函数对象以及结果后处理。

本节介绍不同的示例，展示 FOAMPilot 在自动化 CFD 工作流程以及使用 Python 学习 OpenFOAM 方面的优势和灵活性。

## 示例目标

这些示例的目标包括：

- 演示如何从 Python 初始化 OpenFOAM 案例。
- 展示如何使用 JSON 文件生成和修改网格。
- 说明如何定义流体属性和边界条件。
- 设置 `functionObjects` 以监控物理量（力、压力、场平均等）。
- 创建和管理 OpenFOAM 特定字典（`topoSetDict`、`createPatchDict` 等）。
- 运行仿真并自动化后处理。
- 提供可复现的示例用于学习和原型开发。

## 示例列表

随着更多测试的开发，本部分将不断更新。

- [消声器](muffler/detailled_example_muffler.md)：详细示例，展示汽车消声器的复杂网格生成、边界条件设置以及声学和流体结果分析。  
- [简单汽车](simplecar/detailled_example.md)：基于官方 OpenFOAM 教程 [SimpleCar](https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar) 的示例，演示使用 JSON 网格生成、边界条件应用和空气动力学力监测进行简单汽车气流仿真。

## 说明

每个示例都附带独立的 Python 脚本，功能包括：

1. 定义案例路径 (`current_path`)。
2. 初始化流体属性（密度、粘度、压力、温度等）。
3. 初始化 FOAMPilot 求解器以及 system/constant 文件夹。
4. 从 JSON 文件生成网格。
5. 添加所需的 `functionObjects`（场平均、参考压力、运行时控制等）。
6. 操作 OpenFOAM 字典以创建 patch 和定义区域。
7. 使用现代 API 应用边界条件。
8. 运行仿真。
9. 自动后处理结果并导出 CSV、JSON、PNG 和 HTML 文件。

这些示例设计为模块化，易于适应各种 CFD 案例研究。
