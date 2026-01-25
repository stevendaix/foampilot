from __future__ import annotations
from typing import Any, Type, TypeVar, get_origin, get_args, Union, Optional

T = TypeVar("T", bound="Serializable")

from foampilot.utilities.manageunits import ValueWithUnit

class Serializable:
    """
    Base class for serialization/deserialization.
    Works with nested Serializable classes, ValueWithUnit,
    lists and dicts containing them, including Optional/Union.
    """

    def as_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Serializable):
                result[key] = value.as_dict()
            elif isinstance(value, ValueWithUnit):
                result[key] = {"magnitude": value.magnitude, "units": value.units}
            elif isinstance(value, list):
                result[key] = [
                    v.as_dict() if isinstance(v, Serializable) else
                    {"magnitude": v.magnitude, "units": v.units} if isinstance(v, ValueWithUnit) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                result[key] = {
                    k: v.as_dict() if isinstance(v, Serializable) else
                    {"magnitude": v.magnitude, "units": v.units} if isinstance(v, ValueWithUnit) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        obj = cls.__new__(cls)  # bypass __init__
        annotations = getattr(cls, "__annotations__", {})

        for key, expected_type in annotations.items():
            value = data.get(key, None)  # assign None if missing

            if value is None:
                setattr(obj, key, None)
                continue

            origin = get_origin(expected_type)
            args = get_args(expected_type)

            # Case: Optional or Union
            if origin is Union:
                # Take the first non-None type
                non_none_types = [a for a in args if a is not type(None)]
                if non_none_types:
                    expected_type = non_none_types[0]
                    origin = get_origin(expected_type)
                    args = get_args(expected_type)

            # Case: ValueWithUnit
            if expected_type == ValueWithUnit and isinstance(value, dict):
                setattr(obj, key, ValueWithUnit.from_dict(value))
                continue

            # Case: nested Serializable
            if isinstance(value, dict) and hasattr(expected_type, "from_dict"):
                setattr(obj, key, expected_type.from_dict(value))
                continue

            # Case: list of Serializable
            if origin in (list, list[Any]):
                inner_type = args[0] if args else Any
                if hasattr(inner_type, "from_dict"):
                    setattr(obj, key, [inner_type.from_dict(v) if isinstance(v, dict) else v for v in value])
                else:
                    setattr(obj, key, value)
                continue

            # Case: dict with Serializable values
            if origin in (dict, dict[Any, Any]):
                inner_type = args[1] if len(args) > 1 else Any
                if hasattr(inner_type, "from_dict"):
                    setattr(obj, key, {k: inner_type.from_dict(v) if isinstance(v, dict) else v for k, v in value.items()})
                else:
                    setattr(obj, key, value)
                continue

            # Default: assign raw value
            setattr(obj, key, value)

        # Assign any extra keys not in annotations
        for key, value in data.items():
            if key not in annotations:
                setattr(obj, key, value)

        return obj