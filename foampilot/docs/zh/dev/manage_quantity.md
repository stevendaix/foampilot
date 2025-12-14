```md
# `Quantity` 类：方法详细说明

本文档详细说明了 `Quantity` 类的各个方法，该类用于处理带有单位的物理量。

## 介绍

`Quantity` 类是对 Pint 库的封装，可用于操作带有物理单位的数值。它简化了单位转换、显示、序列化以及算术运算。

## 构造函数

### 描述

构造函数用于使用数值和单位初始化一个新的 `Quantity` 实例。

- **参数**：
  - `value`：物理量的数值。
  - `unit`：单位字符串（例如："m/s"、"Pa"、"kg"）。

## 主要方法

### `set_quantity`

#### 描述

更新物理量的数值和单位。

- **参数**：
  - `value`：新的数值。
  - `unit`：新的单位。

### `get_in`

#### 描述

将物理量转换为目标单位，并返回数值。

- **参数**：
  - `target_unit`：目标单位。
- **返回**：目标单位下的数值。
- **异常**：如果无法转换，将引发错误。

### `to`

#### 描述

将物理量转换为目标单位，并返回一个新的 `Quantity` 对象。

- **参数**：
  - `target_unit`：目标单位。
- **返回**：包含转换后数值的新 `Quantity` 对象。
- **异常**：如果无法转换，将引发错误。

## 序列化

### `as_dict`

#### 描述

返回物理量的字典表示。

- **返回**：包含 `value` 和 `unit` 键的字典。
- **示例输出**：`{"value": 10.0, "unit": "meter / second"}`

### `from_dict`

#### 描述

从字典创建一个 `Quantity` 对象。

- **参数**：
  - `data`：包含 `value` 和 `unit` 的字典。
- **返回**：新的 `Quantity` 对象。

### `from_pint`

#### 描述

从 Pint Quantity 对象创建一个 `Quantity` 对象。

- **参数**：
  - `pint_quantity`：Pint Quantity 对象。
- **返回**：新的 `Quantity` 对象。

## 表示方法

### `__repr__` 和 `__str__`

#### 描述

- **`__repr__`**：返回对象的正式表示，用于调试。  
- **`__str__`**：返回可读性较高的对象表示。

## 算术运算

### 描述

该类支持与其他 `Quantity` 对象或标量的 `+`、`-`、`*`、`/` 运算。

- **加法 (`__add__`)**：两个物理量相加或物理量与标量相加。
- **减法 (`__sub__`)**：两个物理量相减或物理量与标量相减。
- **乘法 (`__mul__`)**：两个物理量相乘或物理量与标量相乘。
- **除法 (`__truediv__`)**：两个物理量相除或物理量与标量相除。

- **返回**：一个新的 `Quantity` 对象，表示运算结果。

## 使用示例

1. **创建对象**：
   - `speed = Quantity(10, "m/s")`
   - `pressure = Quantity(101325, "Pa")`

2. **单位转换**：
   - `speed.get_in("km/h")`
   - `pressure.get_in("atm")`

3. **算术运算**：
   - `d1 + d2`
   - `force / area`

4. **序列化**：
   - `data = {"speed": speed.as_dict(), "pressure": pressure.as_dict()}`
   - `json_str = json.dumps(data, indent=2)`

5. **反序列化**：
   - `speed_loaded = Quantity.from_dict(loaded_data["speed"])`

## 结论

`Quantity` 类简化了 Python 中带单位物理量的处理，利用 Pint 进行单位转换，同时增加了序列化和算术运算功能。
```
