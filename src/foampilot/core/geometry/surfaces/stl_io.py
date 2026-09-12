import pyvista as pv
import numpy as np
from pathlib import Path
from typing import Union, Optional


class SurfaceReader:
    @staticmethod
    def read_stl(path: Union[str, Path], scale: Optional[float] = None) -> pv.PolyData:
        mesh = pv.read(str(path))
        if scale is not None and scale != 1.0:
            mesh.scale([scale] * 3, inplace=True)
        return mesh

    @staticmethod
    def read_vtp(path: Union[str, Path]) -> pv.PolyData:
        return pv.read(str(path))

    @staticmethod
    def read_stl_bytes(data: bytes, scale: Optional[float] = None) -> pv.PolyData:
        import io
        stream = io.BytesIO(data)
        mesh = pv.read(stream)
        if scale is not None and scale != 1.0:
            mesh.scale([scale] * 3, inplace=True)
        return mesh


class SurfaceWriter:
    @staticmethod
    def write_stl(mesh: pv.PolyData, path: Union[str, Path], binary: bool = True) -> None:
        mesh.save(str(path), binary=binary)

    @staticmethod
    def write_vtp(mesh: pv.PolyData, path: Union[str, Path]) -> None:
        mesh.save(str(path))

    @staticmethod
    def write_stl_bytes(mesh: pv.PolyData, binary: bool = True) -> bytes:
        import io
        stream = io.BytesIO()
        mesh.save(stream, binary=binary)
        return stream.getvalue()