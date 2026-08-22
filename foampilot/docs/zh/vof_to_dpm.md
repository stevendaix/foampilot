# VOF→DPM 转换器

`VofToDpmConverter` 是 foampilot 中的确定性 Python 转换器。它根据 `alpha >= alpha_threshold` 选择单元，按照网格面连接关系识别连通液体区域，并使用物理权重 `alpha × V` 计算每个碎片的体积、质心和平均速度。

```text
V_fragment = sum(alpha_i × V_i)
centroid = sum(alpha_i × V_i × centroid_i) / V_fragment
U_fragment = sum(alpha_i × V_i × U_i) / V_fragment
d_eq = (6 V_fragment / pi)^(1/3)
```

转换器读取 OpenFOAM ASCII 字段，明确拒绝未解码的二进制字段，并写出位置文件、碎片属性文件和 JSON 审计报告。当前版本不会修改 `alpha`，也不会直接向正在运行的粒子云插入 parcel。

```python
from foampilot.utilities.vof_to_dpm import VofToDpmConverter

converter = VofToDpmConverter(alpha_threshold=0.5)
fragments = converter.extract_case("case", time_directory="0.01")
outputs = converter.write_openfoam_outputs(fragments, "case/constant")
```

OpenFOAM 13 的安装和运行命令请参阅[完整指南](vof_to_dpm_openfoam13.md)。关于科学范围和代码检查，请参阅英法文技术审计材料。
