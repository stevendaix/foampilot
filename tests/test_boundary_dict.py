import pytest
from pathlib import Path
import tempfile

from foampilot.core.dictionaries import BoundaryDict


class TestBoundaryDict:
    def test_writes_vol_vector_field_for_U(self, tmp_path):
        bd = BoundaryDict("U", {"inlet": {"type": "fixedValue", "value": "uniform (10 0 0)"}}, base_path=tmp_path)
        out = tmp_path / "U"
        bd.write(out)
        content = out.read_text()
        assert "class     volVectorField;" in content
        assert "object     U;" in content

    def test_writes_vol_scalar_field_for_p(self, tmp_path):
        bd = BoundaryDict("p", {"outlet": {"type": "fixedValue", "value": "uniform 0"}}, base_path=tmp_path)
        out = tmp_path / "p"
        bd.write(out)
        content = out.read_text()
        assert "class     volScalarField;" in content
        assert "object     p;" in content

    def test_writes_dimensions_for_k(self, tmp_path):
        bd = BoundaryDict("k", {"walls": {"type": "kqRWallFunction", "value": "uniform 0.375"}}, base_path=tmp_path)
        out = tmp_path / "k"
        bd.write(out)
        content = out.read_text()
        assert "dimensions [0 2 -2 0 0 0 0];" in content

    def test_writes_default_internal_field_when_not_provided(self, tmp_path):
        bd = BoundaryDict("U", {"walls": {"type": "noSlip"}}, base_path=tmp_path)
        out = tmp_path / "U"
        bd.write(out)
        content = out.read_text()
        assert "internalField uniform (0 0 0);" in content

    def test_writes_custom_internal_field(self, tmp_path):
        bd = BoundaryDict("p", {"outlet": {"type": "fixedValue", "value": "uniform 0"}}, internal_field="uniform 1e5", base_path=tmp_path)
        out = tmp_path / "p"
        bd.write(out)
        content = out.read_text()
        assert "internalField uniform 1e5;" in content

    def test_writes_boundary_field_with_patches(self, tmp_path):
        boundaries = {
            "inlet": {"type": "fixedValue", "value": "uniform (10 0 0)"},
            "walls": {"type": "noSlip"},
        }
        bd = BoundaryDict("U", boundaries, base_path=tmp_path)
        out = tmp_path / "U"
        bd.write(out)
        content = out.read_text()
        assert "boundaryField" in content
        assert "inlet" in content
        assert "walls" in content
        assert "fixedValue" in content
        assert "noSlip" in content

    def test_includes_etc_by_default(self, tmp_path):
        bd = BoundaryDict("U", {"inlet": {"type": "fixedValue", "value": "uniform (10 0 0)"}}, base_path=tmp_path)
        out = tmp_path / "U"
        bd.write(out)
        content = out.read_text()
        assert '#includeEtc "caseDicts/setConstraintTypes"' in content

    def test_omits_include_etc_when_disabled(self, tmp_path):
        bd = BoundaryDict("U", {"inlet": {"type": "fixedValue", "value": "uniform (10 0 0)"}}, include_etc=False, base_path=tmp_path)
        out = tmp_path / "U"
        bd.write(out)
        content = out.read_text()
        assert "includeEtc" not in content

    def test_compressible_p_dimension_override(self, tmp_path):
        bd = BoundaryDict("p", {"outlet": {"type": "fixedValue", "value": "uniform 0"}}, compressible=True, base_path=tmp_path)
        out = tmp_path / "p"
        bd.write(out)
        content = out.read_text()
        assert "dimensions [1 -1 -2 0 0 0 0];" in content

    def test_writes_footer_when_requested(self, tmp_path):
        bd = BoundaryDict("U", {"inlet": {"type": "fixedValue", "value": "uniform (10 0 0)"}}, base_path=tmp_path)
        out = tmp_path / "U"
        bd.write(out, footer=True)
        content = out.read_text()
        assert "// ************************************************************************* //" in content

    def test_creates_parent_directory(self, tmp_path):
        target = tmp_path / "0" / "U"
        bd = BoundaryDict("U", {"inlet": {"type": "fixedValue", "value": "uniform (10 0 0)"}}, base_path=tmp_path)
        bd.write(target)
        assert target.exists()
