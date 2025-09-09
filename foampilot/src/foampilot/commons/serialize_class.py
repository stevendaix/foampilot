from __future__ import annotations
from typing import Any, Type, TypeVar, get_origin, get_args

T = TypeVar("T", bound="Serializable")


class Serializable:
    """
    Base class for serialization/deserialization.
    Works with nested Serializable classes, Quantity,
    lists and dicts containing them.
    """

    def as_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Serializable):
                result[key] = value.as_dict()
            elif isinstance(value, list):
                result[key] = [v.as_dict() if isinstance(v, Serializable) else v for v in value]
            elif isinstance(value, dict):
                result[key] = {k: v.as_dict() if isinstance(v, Serializable) else v for k, v in value.items()}
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        obj = cls.__new__(cls)  # bypass __init__
        annotations = getattr(cls, "__annotations__", {})

        for key, value in data.items():
            expected_type = annotations.get(key)

            if expected_type is None:
                setattr(obj, key, value)
                continue

            # Case: Quantity
            if expected_type.__name__ == "Quantity" and isinstance(value, dict):
                setattr(obj, key, Quantity.from_dict(value))
                continue

            # Case: nested Serializable
            if isinstance(value, dict) and hasattr(expected_type, "from_dict"):
                setattr(obj, key, expected_type.from_dict(value))
                continue

            # Case: list of Serializable
            if get_origin(expected_type) is list:
                inner_type = get_args(expected_type)[0]
                if hasattr(inner_type, "from_dict"):
                    setattr(obj, key, [inner_type.from_dict(v) if isinstance(v, dict) else v for v in value])
                    continue

            # Case: dict with Serializable values
            if get_origin(expected_type) is dict:
                inner_type = get_args(expected_type)[1]
                if hasattr(inner_type, "from_dict"):
                    setattr(obj, key, {k: inner_type.from_dict(v) if isinstance(v, dict) else v for k, v in value.items()})
                    continue

            # Default: assign raw value
            setattr(obj, key, value)

        return obj