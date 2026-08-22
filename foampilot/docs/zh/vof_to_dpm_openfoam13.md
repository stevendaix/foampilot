# OpenFOAM 13 中的 VOF 到 DPM

本页面介绍 foampilot 中完整集成的 VOF–DPM 项目，包括 Python 转换器、OpenFOAM 13 原生 C/C++ 源码、不可压缩和可压缩 `fvModel`、验证算例以及 PDF 技术报告生成器。

> 当前实现区分离线碎片提取和求解器–粒子云耦合。在 VOF 体积分数被实际扣除、粒子包被事务性插入之前，不能将其描述为完整的实时守恒转换。

## 1. 安装要求

以下命令假设 Ubuntu 中的 OpenFOAM 13 安装在 `/opt/openfoam13`：

```bash
sudo apt update
sudo apt install -y git build-essential python3 python3-pip
. /opt/openfoam13/etc/bashrc
foamVersion
```

在 foampilot 根目录安装 Python 依赖：

```bash
cd foampilot
sudo pip3 install -r requirements.txt
sudo pip3 install pytest
```

如果只使用转换器，目标模块测试主要需要 NumPy 和 pytest。完整导入 foampilot 时，还可能需要几何和后处理相关的可选依赖。

## 2. 文件位置

完整的 OpenFOAM 13 VOF–DPM 项目位于 `foampilot/examples/openfoam13/vof_to_dpm/`。

| 路径 | 内容 |
|---|---|
| `src/foampilot/utilities/vof_to_dpm.py` | OpenFOAM ASCII 读取、连通区域提取和输出写入 |
| `test/test_vof_to_dpm.py` | Python 单元测试 |
| `examples/course_vof_to_dpm.py` | Python 教学练习 |
| `examples/generate_vof_to_dpm_technical_note.py` | PDF 技术报告生成器 |
| `src/foampilot/report/typst_pdf.py` | 技术报告使用的 Typst 引擎 |
| `examples/openfoam13/vof_to_dpm/applications/vofToDpm` | 原生 C++ 离线提取器 |
| `examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds` | 不可压缩 `fvModel` |
| `examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds` | 可压缩 `fvModel` |
| `examples/openfoam13/vof_to_dpm/test/openfoam13` | OpenFOAM 13 验证算例和 `Allrun` 脚本 |
| `docs/fr/vof_to_dpm_technical_note.pdf` | 已生成的技术报告 |

完整的 `statisticalDPMFoam` 源码，包括 `.C`、`.H`、`Make/files` 和 `Make/options`，位于 `examples/openfoam13/vof_to_dpm/statisticalDPMFoam/`。

## 3. 运行 Python 测试

从 `foampilot` 目录运行：

```bash
PYTHONPATH=src/foampilot/utilities python -m pytest -q test/test_vof_to_dpm.py
```

测试覆盖分离和连接的液体碎片、`alpha × V` 加权、无效索引、显式过滤器、OpenFOAM ASCII 字段读取和输出文件生成。

运行教学示例：

```bash
PYTHONPATH=src python examples/course_vof_to_dpm.py
```

程序会输出碎片数量、源体积、转换体积、体积残差以及转换前后的加权动量。

## 4. 编译 OpenFOAM C/C++ 组件

在终端中加载 OpenFOAM 环境：

```bash
. /opt/openfoam13/etc/bashrc
cd foampilot/examples/openfoam13/vof_to_dpm
```

分别编译三个组件：

```bash
wmake applications/vofToDpm
wmake applications/incompressibleVoFClouds
wmake applications/compressibleVoFClouds
```

每个组件的 `Make/files` 和 `Make/options` 都保存在源代码目录中。`Make/linux64*` 下的编译对象由 `wmake` 生成，不纳入版本控制。

编译随 foampilot 提供的 `statisticalDPMFoam` 源码：

```bash
cd examples/openfoam13/vof_to_dpm/statisticalDPMFoam
./Allwmake
```

## 5. 运行 C++ 离线提取器

原生提取器读取串行算例、`alpha` 字段、可选的 `U` 字段和网格连接关系。示例：

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/vofToDpmSingleCell
. /opt/openfoam13/etc/bashrc
../../../../applications/vofToDpm/Make/linux64GccDPInt32Opt/vofToDpm \
    -alpha alpha.liquid -U U -threshold 0.5 -rhoLiquid 1000
```

可执行文件的路径可能根据编译器、精度和 OpenFOAM 选项变化。转换器生成位置文件、碎片属性文件以及选中体积、转换体积和丢弃体积报告。液体体积采用 `sum(alpha_i × V_i)`，不对界面单元进行重新归一化。

## 6. 运行不可压缩验证算例

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/incompressibleVoFCloudsDamBreak
./Allrun
```

脚本准备算例，启用 `fvModels` 和动量预测器，加载 `incompressibleVoFClouds` 并检查粒子云活动。当前测试使用受控的手动注入，用于验证求解器–粒子云路径；它还不是自动 VOF 碎片转换测试。

## 7. 运行可压缩验证算例

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/compressibleVoFCloudsDamBreak
./Allrun
```

该 smoke test 验证 `compressibleVoFClouds` 的运行时选择、粒子云机械演化和动量源项。它还不是热力学守恒的转换，因为质量、焓和压力一致性仍需要专门的传递实现。

## 8. 生成技术 PDF

报告生成器使用 foampilot 的 `ScientificDocument` 和 `TypstRenderer`：

```bash
cd foampilot
python examples/generate_vof_to_dpm_technical_note.py
```

从仓库根目录执行时，输出位于 `report/`：

```text
report/vof_to_dpm_technical_note.pdf
report/vof_to_dpm_technical_note.typ
report/vof_to_dpm.bib
```

报告包括转换判据、守恒方程、代码审计以及实时转换的建议架构。

## 9. 当前科学范围

Python 转换器和 C++ 离线工具能够计算连通区域体积、质心、体积加权速度和等效球直径。两个 `fvModel` 能够推进 `parcelCloudList`，并将粒子云的机械源项返回到载体动量方程。

完整的生产级自动转换还需要有界的 `alpha` 消耗、动态粒子插入、稳定的碎片 ID、MPI 连通区域合并、防止重复转换，以及可压缩情况下质量和能量的一致传递。

## 10. 多语言文档

| 语言 | 安装和运行指南 | 详细材料 |
|---|---|---|
| English | `docs/en/vof_to_dpm_openfoam13.md` | `docs/en/vof_to_dpm.md` |
| Français | `docs/fr/vof_to_dpm_openfoam13.md` | `docs/fr/cours_vof_to_dpm.md`, `docs/fr/audit_implementation_vof_to_dpm.md` |
| 中文 | `docs/zh/vof_to_dpm_openfoam13.md` | `docs/zh/vof_to_dpm.md` |
