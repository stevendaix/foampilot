# VOF→DPM 完整项目地图

完整实现位于 `examples/openfoam13/vof_to_dpm/`。

| 模块 | 路径 |
|---|---|
| Python 转换器 | `src/foampilot/utilities/vof_to_dpm.py` |
| Python 测试 | `test/test_vof_to_dpm.py` |
| 教学示例 | `examples/course_vof_to_dpm.py` |
| PDF 生成器 | `examples/generate_vof_to_dpm_technical_note.py` |
| C++ 离线提取器 | `examples/openfoam13/vof_to_dpm/applications/vofToDpm/` |
| 不可压缩耦合 | `examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds/` |
| 可压缩耦合 | `examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds/` |
| statisticalDPMFoam 源码 | `examples/openfoam13/vof_to_dpm/statisticalDPMFoam/` |
| OpenFOAM 13 测试 | `examples/openfoam13/vof_to_dpm/test/openfoam13/` |
| 技术报告和参考文献 | `docs/fr/vof_to_dpm_technical_note.pdf`, `docs/fr/vof_to_dpm.bib` |

主要算例是 `vofToDpmSingleCell`、`vofToDpmParcelInBox`、`incompressibleVoFCloudsDamBreak` 和 `compressibleVoFCloudsDamBreak`。请先阅读 [安装与运行指南](vof_to_dpm_openfoam13.md)，再运行 Python 测试，最后编译 OpenFOAM 组件。

英文和法文版本分别位于 `docs/en/vof_to_dpm_openfoam13.md` 和 `docs/fr/vof_to_dpm_openfoam13.md`。
