# OpenFOAM 示例使用 FoamPilot

本示例演示如何使用 Python 库 FoamPilot 来设置、网格生成、求解和后处理一个简单的不可压缩流动案例。

示例参考：https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar

---

## 1. 定义案例路径
```python
from pathlib import Path
current_path = Path.cwd() / "cases"
```
这里将当前工作目录下的 `cases` 文件夹作为案例的根目录。

---

## 2. 流体属性
```python
from foampilot import FluidMechanics, Quantity
available_fluids = FluidMechanics.get_available_fluids()
fluid = FluidMechanics(
    available_fluids["Air"],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)
nu = fluid.get_fluid_properties()["kinematic_viscosity"]
```
使用 FoamPilot 的现代 API 获取空气的运动粘度，并设置温度和压力。

---

## 3. 初始化求解器
```python
from foampilot.solver import Solver
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False
solver.constant.transportProperties.nu = nu
solver.system.write()
solver.constant.write()
```
这里创建求解器对象，禁用可压缩性和重力，并将粘度写入 OpenFOAM 文件结构。

---

## 4. 网格生成（blockMesh JSON）
```python
from foampilot import Meshing
from pathlib import Path
mesh = Meshing(current_path, mesher="blockMesh")
data_path = Path.cwd() / "block_mesh.json"
mesh.mesher.load_from_json(data_path)
mesh.mesher.write(file_path=current_path / "system" / "blockMeshDict")
mesh.mesher.run()
```
- **load_from_json**: 从 JSON 文件加载 blockMesh 配置。方便使用程序化方式或从其他工具生成的网格配置。
- **write**: 将 Python 中的网格配置写入 `blockMeshDict`。
- **run**: 调用 OpenFOAM `blockMesh` 命令生成网格。

---

## 5. 函数对象 (Function Objects)
### 5.1 计算 fieldAverage
```python
from foampilot import utilities
name_field, field_average_dict = utilities.Functions.field_average("fieldAverage")
utilities.Functions.write_function_field_average(name_field, field_average_dict, base_path=current_path, folder='system')
```
`fieldAverage` 用于在计算中统计指定区域的平均场值。

### 5.2 设置参考压力
```python
name_field_ref, reference_dict = utilities.Functions.reference_pressure("referencePressure")
utilities.Functions.write_function_reference_pressure(name_field_ref, reference_dict, base_path=current_path, folder='system')
```
参考压力用于设置压力求解的基准点。

### 5.3 运行时控制 (runTimeControl)
```python
conditions1 = {
    "condition1": {
        "type": "average",
        "functionObject": "forceCoeffs1",
        "fields": "(Cd)",
        "tolerance": "1e-3",
        "window": "20",
        "windowType": "exact"
    }
}
name_field_rt1, rt1_dict = utilities.Functions.run_time_control("runTimeControl", conditions=conditions1)
utilities.Functions.write_function_run_time_control(name_field=name_field_rt1, name_condition="runTimeControl1", function_dict=rt1_dict, base_path=current_path, folder='system')
solver.system.write_functions_file(includes=["fieldAverage", "runTimeControl"])
```
`runTimeControl` 用于在模拟运行中根据特定条件停止或记录数据。

---

## 6. 字典操作 (patches & topoSet)
### 6.1 创建 patch
```python
patch_names = ["airIntake"]
patches_dict = utilities.dictonnary.dict_tools.create_patches_dict(patch_names)
create_patch_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='createPatchDict', **patches_dict)
create_patch_dict_file.write("createPatchDict", current_path)
```
用于在网格中创建或修改边界面。

### 6.2 创建 topoSet
```python
actions = [
    utilities.dictonnary.dict_tools.create_action(name="porousCells", action_type="cellSet", action="new", source="boxToCell", box=[(2.05, 0.4, -1), (2.1, 0.85, 1)]),
    utilities.dictonnary.dict_tools.create_action(name="porousZone", action_type="cellZoneSet", action="new", source="setToCellZone", set="porousCells"),
    utilities.dictonnary.dict_tools.create_action(name="airIntake", action_type="faceSet", action="new", source="patchToFace", patch="body"),
    utilities.dictonnary.dict_tools.create_action(name="airIntake", action_type="faceSet", action="subset", source="boxToFace", box=[(2.6, 0.75, 0), (2.64, 0.8, 0.1)])
]
actions_dict = utilities.dictonnary.dict_tools.create_actions_dict(actions)
create_topo_set_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='topoSetDict', **actions_dict)
create_topo_set_dict_file.write("topoSetDict", current_path)
solver.system.run_topoSet()
solver.system.run_createPatch()
```
`topoSet` 用于创建 cellSets、faceSets 和 cellZoneSets，为后续函数和模拟条件提供便利。

---

## 7. 边界条件
```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(pattern="inlet", condition_type="velocityInlet", velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)
solver.boundary.apply_condition_with_wildcard(pattern="airIntake", condition_type="velocityInlet", velocity=(Quantity(1.2, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)
solver.boundary.apply_condition_with_wildcard(pattern="outlet", condition_type="pressureOutlet")
solver.boundary.apply_condition_with_wildcard(pattern="walls", condition_type="wall")
solver.boundary.write_boundary_conditions()
```
设置速度入口、压力出口以及壁面边界条件，并将其写入 OpenFOAM 文件。

---

## 8. 运行模拟
```python
solver.run_simulation()
```
执行 OpenFOAM 模拟。

---

## 9. 后处理
```python
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
```
处理残差并导出为 CSV、JSON、PNG 和 HTML 格式。

---

## 完成
```python
print("Simulation and post-processing completed.")
```
表示模拟及后处理完成。

