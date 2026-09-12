from enum import Enum


class CFDLOD(str, Enum):
    LOD0 = "lod0"
    LOD1 = "lod1"
    LOD2 = "lod2"
    LOD3 = "lod3"
    LOD4 = "lod4"


class RoofType(str, Enum):
    FLAT = "flat"
    GABLE = "gable"
    HIP = "hip"
    PYRAMID = "pyramid"
    UNKNOWN = "unknown"
