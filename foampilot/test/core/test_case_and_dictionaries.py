from __future__ import annotations

from pathlib import Path

import pytest

from foampilot.core.case import CaseLayout, create_case_structure
from foampilot.core.dictionaries import DictionaryWriter


def test_case_layout_creates_standard_and_extra_directories(tmp_path: Path) -> None:
    case = CaseLayout(tmp_path / "case")
    assert case.ensure(("constant", "geometry")) == (tmp_path / "case").resolve()
    assert case.validate() == ()
    assert (tmp_path / "case" / "geometry").is_dir()


def test_case_layout_rejects_escape_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CaseLayout(tmp_path / "case").ensure(("../outside",))


def test_dictionary_writer_renders_openfoam_header_and_values(tmp_path: Path) -> None:
    path = DictionaryWriter(
        "transportProperties",
        {"nu": 1e-5, "model": "Newtonian", "enabled": True},
    ).write(tmp_path / "constant" / "transportProperties")
    content = path.read_text(encoding="utf-8")
    assert "object      transportProperties;" in content
    assert "nu 1e-05;" in content
    assert "enabled true;" in content


def test_dictionary_writer_rejects_path_in_object_name() -> None:
    with pytest.raises(ValueError):
        DictionaryWriter("../controlDict")


def test_legacy_case_function_matches_core_layout(tmp_path: Path) -> None:
    result = create_case_structure(tmp_path / "case", extra_dirs=("geometry",))
    assert result == (tmp_path / "case").resolve()
    assert (result / "system").is_dir()
