<!-- Logo -->
<img src="foampilot/images/logo.png" alt="FoamPilot Logo" width="250">

# foampilot 🚀

🌍 **语言：**  
[English](README.md) | [Français](README.fr.md) | [中文](README.zh.md)

**foampilot** 是一个 Python 平台，旨在 *完全管理 OpenFOAM 仿真*，  
使用 Python 作为唯一可信源 —— 从案例定义和网格划分，到执行、后处理和报告生成。

它面向希望获得 **可复现、可脚本化和可维护 CFD 工作流** 的工程师和研究人员，  
无需手动编辑 OpenFOAM 字典文件。

---

## 背景与动机

OpenFOAM 功能强大，但管理仿真通常涉及：
- 手动编辑多个字典文件，
- 脆弱的案例复制，
- 临时脚本进行后处理，
- 不同研究间的复现性有限。

**foampilot** 通过将 Python 放在工作流的核心解决了这些问题：  
OpenFOAM 案例变为 *生成的工件*，而不是手动维护的输入文件。

---

## 主要功能

- **Python 优先工作流**  
  直接在 Python 中定义网格、求解器、边界条件和控制参数。

- **自动生成 OpenFOAM 案例**  
  程序化生成 `system`、`constant` 和 `0/` 文件，保证一致性和可复现性。

- **网格管理**  
  原生支持 `blockMesh` 和 `snappyHexMesh`，架构可扩展。

- **仿真控制**  
  直接通过 Python 启动和管理 OpenFOAM 求解器。

- **现代化后处理**  
  使用 PyVista 进行 3D 可视化，自动导出图像和动画。

- **自动化报告**  
  生成 PDF 计算报告（LaTeX）或交互式仪表板（Streamlit）。

---

## 设计理念

- Python 是 **唯一可信源**
- OpenFOAM 字典文件由 **程序生成**，不手动编辑
- 优先保证可复现性和可追踪性，而非 GUI 工作流
- 配置明确，可检查
- 为自动化、参数化研究和工程工作流设计

---

## foampilot 不是

- 不是 CFD 求解器  
- 不是 OpenFOAM 替代品  
- 不是基于 GUI 的工具  
- 不打算隐藏 OpenFOAM 概念  

foampilot 假定用户 **具备 OpenFOAM 和 CFD 基础知识**。

---

## 支持平台

- **Linux**（原生）  
- **Windows via WSL2**（推荐）  
- **macOS**（通过官方 OpenFOAM 构建）

OpenFOAM 安装和系统设置另行文档说明。

---

## 文档

📘 完整文档，包括安装指南和详细使用说明：

**https://stevendaix.github.io/foampilot/zh/**

文档内容包括：
- OpenFOAM & WSL 安装指南
- 项目结构和概念
- 网格划分、求解器控制和后处理
- 报告生成和可视化工作流

---

## 项目状态

⚠️ **状态：** 开发阶段 / 测试版

公共 API 可能会变化。  
欢迎反馈、讨论和贡献。

---

## 许可

本项目遵循 **MIT 许可** 发布。
