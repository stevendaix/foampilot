#!/usr/bin/env python3
"""Tests unitaires pour le module CHT (Conjugate Heat Transfer) de foampilot.

Couvre :
- ChtSolver (création, validation, setup_case)
- FluidRegion / SolidRegion (champs T, U, propriétés thermophysiques)
- CoupledInterface
- Conditions aux limites CHT (classes et fonctions)
- regionSolvers round-trip dans ControlDictFile
- CaseFieldsManager multi-région
- Fonctions de post-traitement CHT
- SOLVER_MODULES (chtMultiRegionFoam, chtMultiRegionSimpleFoam)
"""

import sys
import tempfile
import ast
from pathlib import Path

import numpy as np

# Ensure foampilot is importable
sys.path.insert(0, str(Path(__file__).parent / "foampilot" / "src"))

# Import all CHT components at module level
from foampilot.cht import (
    ChtSolver, FluidRegion, SolidRegion, CoupledInterface,
    CoupledTemperatureBC, ExternalTemperatureBC, HeatFluxBC,
    FixedTemperatureBC, InletOutletTemperatureBC, SymmetryBC,
    TotalTemperatureBC, RadiationCoupledTemperatureBC,
    calc_region_heat_flux, calc_interface_heat_flux,
    calc_nusselt_number, calc_thermal_boundary_layer_thickness,
    calc_heat_transfer_coefficient, calc_total_heat_balance,
    calc_temperature_contour, calc_thermal_resistance,
)


# ---------------------------------------------------------------------------
# Step 1 — Module cht/ imports
# ---------------------------------------------------------------------------

def test_cht_module_imports():
    """Test that all CHT module components are importable."""
    from foampilot.cht import (
        ChtSolver, FluidRegion, SolidRegion, CoupledInterface,
        CoupledTemperatureBC, ExternalTemperatureBC, HeatFluxBC,
        FixedTemperatureBC, InletOutletTemperatureBC, SymmetryBC,
        TotalTemperatureBC, RadiationCoupledTemperatureBC,
        calc_region_heat_flux, calc_interface_heat_flux,
        calc_nusselt_number, calc_thermal_boundary_layer_thickness,
        calc_heat_transfer_coefficient, calc_total_heat_balance,
        calc_temperature_contour, calc_thermal_resistance,
    )
    assert ChtSolver is not None
    assert FluidRegion is not None
    assert SolidRegion is not None
    assert CoupledInterface is not None
    print("[OK] All CHT module imports resolve")


def test_cht_syntax_check():
    """Syntax-check all CHT module files."""
    cht_dir = Path(__file__).parent / "foampilot" / "src" / "foampilot" / "cht"
    files = list(cht_dir.glob("*.py"))
    for f in files:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f"[OK] Syntax check: {f.name}")


# ---------------------------------------------------------------------------
# Step 1 — FluidRegion / SolidRegion
# ---------------------------------------------------------------------------

def test_fluid_region_fields():
    """Test FluidRegion generates valid T and U field content."""
    region = FluidRegion(name="fluid", temperature=350.0, velocity=(5.0, 0.0, 0.0))

    t_content = region.get_T_field_content()
    assert "volScalarField" in t_content
    assert "object      T" in t_content
    assert "internalField   uniform 350" in t_content

    u_content = region.get_U_field_content()
    assert "volVectorField" in u_content
    assert "object      U" in u_content
    assert "internalField   uniform (5.0 0.0 0.0)" in u_content
    print("[OK] FluidRegion field content generated correctly")


def test_solid_region_fields():
    """Test SolidRegion generates valid T field content and properties."""
    region = SolidRegion(
        name="solid", temperature=400.0,
        thermal_conductivity=45.0, density=7800.0, specific_heat=460.0
    )

    t_content = region.get_T_field_content()
    assert "volScalarField" in t_content
    assert "object      T" in t_content
    assert "internalField   uniform 400" in t_content

    tp = region.get_thermophysical_properties()
    assert "heSolidThermo" in tp
    assert "specificHeat" in tp
    assert "thermalConductivity" in tp

    tp_trans = region.get_transport_properties()
    assert "transportModel" in tp_trans
    assert "mu" in tp_trans
    print("[OK] SolidRegion field content and properties generated correctly")


def test_fluid_region_thermophysical_properties():
    """Test FluidRegion generates non-empty thermophysical properties."""
    region = FluidRegion(name="fluid")
    tp = region.get_thermophysical_properties()
    assert len(tp) > 0
    assert "heRhoThermo" in tp
    print("[OK] FluidRegion thermophysical properties generated")


# ---------------------------------------------------------------------------
# Step 4 — SOLVER_MODULES
# ---------------------------------------------------------------------------

def test_cht_solvers_in_modules():
    """Test that both CHT solvers are registered in SOLVER_MODULES."""
    from foampilot.solver.base_solver import BaseSolver

    assert "chtMultiRegionFoam" in BaseSolver.SOLVER_MODULES
    assert "chtMultiRegionSimpleFoam" in BaseSolver.SOLVER_MODULES
    print("[OK] Both CHT solvers registered in SOLVER_MODULES")


def test_cht_solver_creation():
    """Test that ChtSolver can be created with valid regions."""
    with tempfile.TemporaryDirectory() as tmp:
        case_path = Path(tmp) / "test_cht_case"
        case_path.mkdir()

        fluid = FluidRegion(name="fluid", temperature=350.0, velocity=(10.0, 0.0, 0.0))
        solid = SolidRegion(name="solid", temperature=400.0)
        interface = CoupledInterface(name="fluid_to_solid", fluid_region="fluid", solid_region="solid")

        solver = ChtSolver(
            case_path=case_path,
            solver_name="chtMultiRegionSimpleFoam",
            regions=[fluid, solid],
            interfaces=[interface],
        )

        assert solver.solver_name == "chtMultiRegionSimpleFoam"
        assert solver.energy_activated is True
        assert len(solver.regions) == 2
        assert len(solver.interfaces) == 1
        print("[OK] ChtSolver created successfully")


def test_cht_solver_invalid_name_raises():
    """Test that ChtSolver raises ValueError for unsupported solver."""
    from foampilot.cht import ChtSolver, FluidRegion

    raised = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ChtSolver(
                case_path=tmp,
                solver_name="notARealSolver",
                regions=[FluidRegion(name="fluid")],
            )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for invalid solver name"
    print("[OK] ChtSolver rejects invalid solver name")


def test_cht_solver_region_solvers_auto():
    """Test that ChtSolver auto-generates region_solvers for fluid/solid."""
    from foampilot.cht import ChtSolver, FluidRegion, SolidRegion

    with tempfile.TemporaryDirectory() as tmp:
        fluid = FluidRegion(name="fluid")
        solid = SolidRegion(name="solid")
        solver = ChtSolver(
            case_path=tmp,
            regions=[fluid, solid],
        )
        assert solver._region_solvers == {"fluid": "fluid", "solid": "solid"}
        print("[OK] ChtSolver auto-generates correct region_solvers")


# ---------------------------------------------------------------------------
# Step 2 — ControlDictFile regionSolvers round-trip
# ---------------------------------------------------------------------------

def test_controlDict_region_solvers():
    """Test regionSolvers is set, written, and serialized in to_dict."""
    from foampilot.system.controlDictFile import ControlDictFile

    c = ControlDictFile()
    c.set_region_solvers({"fluid": "fluid", "solid": "solid"})

    assert c.region_solvers == {"fluid": "fluid", "solid": "solid"}

    d = c.to_dict()
    assert "regionSolvers" in d
    assert d["regionSolvers"] == {"fluid": "fluid", "solid": "solid"}
    print("[OK] ControlDictFile regionSolvers in to_dict")


def test_controlDict_region_solvers_roundtrip():
    """Test that regionSolvers survives to_dict → from_dict round-trip."""
    from foampilot.system.controlDictFile import ControlDictFile

    original = ControlDictFile()
    original.set_region_solvers({"fluid": "fluid", "metal": "solid"})
    original_dict = original.to_dict()

    restored = ControlDictFile.from_dict(original_dict)
    assert restored.region_solvers == {"fluid": "fluid", "metal": "solid"}
    print("[OK] ControlDictFile regionSolvers round-trip")


def test_controlDict_region_solvers_write():
    """Test that regionSolvers is written correctly to file."""
    from foampilot.system.controlDictFile import ControlDictFile

    with tempfile.TemporaryDirectory() as tmp:
        c = ControlDictFile()
        c.set_region_solvers({"fluid": "fluid", "solid": "solid"})
        filepath = Path(tmp) / "controlDict"
        c.write(filepath)

        content = filepath.read_text()
        assert "regionSolvers" in content
        assert "fluid" in content
        assert "solid" in content
        print("[OK] ControlDictFile writes regionSolvers to file")


# ---------------------------------------------------------------------------
# Step 3 — CaseFieldsManager multi-region
# ---------------------------------------------------------------------------

def test_case_fields_manager_multi_region():
    """Test that CaseFieldsManager generates correct per-region fields."""
    from foampilot.base.cases_variables import CaseFieldsManager
    from foampilot.cht import FluidRegion, SolidRegion

    fluid = FluidRegion(name="fluid", turbulence_model="kOmegaSST")
    solid = SolidRegion(name="solid")

    fm = CaseFieldsManager(
        energy_activated=True,
        turbulence_model="kOmegaSST",
        regions=[fluid, solid],
    )

    # Fluid region should have T and U and turbulence fields
    fluid_fields = fm.get_region_field_names("fluid")
    assert "T" in fluid_fields
    assert "U" in fluid_fields

    # Solid region should only have T
    solid_fields = fm.get_region_field_names("solid")
    assert "T" in solid_fields
    assert "U" not in solid_fields
    print("[OK] CaseFieldsManager multi-region field generation")


def test_case_fields_manager_no_regions():
    """Test backward compatibility: CaseFieldsManager without regions."""
    from foampilot.base.cases_variables import CaseFieldsManager

    fm = CaseFieldsManager(energy_activated=True)
    assert "T" in fm.get_field_names()
    assert len(fm.region_fields) == 0
    print("[OK] CaseFieldsManager without regions (backward compat)")


# ---------------------------------------------------------------------------
# Step 5 — Boundary conditions
# ---------------------------------------------------------------------------

def test_coupled_temperature_bc_class():
    """Test CoupledTemperatureBC class."""
    bc = CoupledTemperatureBC(patch_name="interface", T_init=350.0, T_neighbor=355.0)
    result = bc.to_of()
    assert result["type"] == "coupledTemperature"
    assert "350" in result["value"]
    assert "355" in result["Tnbr"]
    print("[OK] CoupledTemperatureBC class")


def test_external_temperature_bc_class():
    """Test ExternalTemperatureBC class."""
    bc = ExternalTemperatureBC(patch_name="wall", ambient_temperature=300.0, heat_transfer_coefficient=25.0)
    result = bc.to_of()
    assert result["type"] == "externalTemperature"
    assert "300" in result["Ta"]
    assert "25" in result["h"]
    print("[OK] ExternalTemperatureBC class")


def test_heat_flux_bc_class():
    """Test HeatFluxBC class."""
    bc = HeatFluxBC(patch_name="wall", heat_flux=500.0)
    result = bc.to_of()
    assert result["type"] == "externalWallHeatFluxTemperature"
    assert "500" in result["q"]
    assert result["flux"] == "q"
    print("[OK] HeatFluxBC class")


def test_fixed_temperature_bc_class():
    """Test FixedTemperatureBC class."""
    bc = FixedTemperatureBC(patch_name="wall", temperature=320.0)
    result = bc.to_of()
    assert result["type"] == "fixedValue"
    assert "320" in result["value"]
    print("[OK] FixedTemperatureBC class")


def test_inlet_outlet_temperature_bc_class():
    """Test InletOutletTemperatureBC class."""
    bc = InletOutletTemperatureBC(
        inlet_name="inlet", outlet_name="outlet",
        inlet_temperature=300.0, outlet_temperature=310.0
    )
    result = bc.to_of()
    assert "inlet" in result
    assert "outlet" in result
    assert result["inlet"]["type"] == "fixedValue"
    assert result["outlet"]["type"] == "inletOutlet"
    print("[OK] InletOutletTemperatureBC class")


def test_symmetry_bc_class():
    """Test SymmetryBC class."""
    bc = SymmetryBC(patch_name="symmetry")
    result = bc.to_of()
    assert result["type"] == "symmetry"
    print("[OK] SymmetryBC class")


def test_total_temperature_bc_class():
    """Test TotalTemperatureBC class."""
    bc = TotalTemperatureBC(patch_name="inlet", total_temperature=500.0)
    result = bc.to_of()
    assert result["type"] == "totalTemperature"
    assert "500" in result["T0"]
    print("[OK] TotalTemperatureBC class")


def test_radiation_coupled_temperature_bc_class():
    """Test RadiationCoupledTemperatureBC class."""
    bc = RadiationCoupledTemperatureBC(patch_name="wall", T_init=400.0, T_neighbor=380.0)
    result = bc.to_of()
    assert result["type"] == "radiationCoupledTemperature"
    assert "400" in result["value"]
    assert "380" in result["Tnbr"]
    print("[OK] RadiationCoupledTemperatureBC class")


# ---------------------------------------------------------------------------
# Step 7 — Post-processing functions
# ---------------------------------------------------------------------------

def test_calc_region_heat_flux():
    """Test calc_region_heat_flux with a simple 1D gradient."""
    T = np.array([300.0, 310.0, 320.0, 330.0])
    q = calc_region_heat_flux(T, thermal_conductivity=50.0, dx=0.1)
    assert q.shape == T.shape
    assert np.all(q >= 0)
    print("[OK] calc_region_heat_flux")


def test_calc_interface_heat_flux():
    """Test calc_interface_heat_flux returns all expected keys."""
    T_f = np.array([300.0, 310.0, 320.0])
    T_s = np.array([350.0, 340.0, 330.0])
    result = calc_interface_heat_flux(T_f, T_s, h=50.0, area=2.0)
    assert "q_total" in result
    assert "q_conv" in result
    assert "q_cond_fluid" in result
    assert "q_cond_solid" in result
    assert "T_interface" in result
    assert isinstance(result["q_total"], float)
    print("[OK] calc_interface_heat_flux")


def test_calc_nusselt_number():
    """Test calc_nusselt_number."""
    nu = calc_nusselt_number(q_wall=1000.0, L=0.5, k_fluid=0.6, T_wall=350.0, T_bulk=300.0)
    expected = abs(1000.0) * 0.5 / (0.6 * abs(50.0))
    assert abs(nu - expected) < 1e-6
    print("[OK] calc_nusselt_number")


def test_calc_nusselt_zero_delta():
    """Test calc_nusselt_number with zero temperature difference returns 0."""
    nu = calc_nusselt_number(q_wall=100.0, L=1.0, k_fluid=0.6, T_wall=300.0, T_bulk=300.0)
    assert nu == 0.0
    print("[OK] calc_nusselt_number zero delta_T")


def test_calc_thermal_boundary_layer_thickness():
    """Test calc_thermal_boundary_layer_thickness."""
    T_field = np.array([300.0, 305.0, 310.0, 315.0, 320.0, 325.0])
    x_positions = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    delta = calc_thermal_boundary_layer_thickness(
        T_wall=325.0, T_bulk=300.0, T_field=T_field, x_positions=x_positions
    )
    assert delta >= 0.0
    print("[OK] calc_thermal_boundary_layer_thickness")


def test_calc_heat_transfer_coefficient():
    """Test calc_heat_transfer_coefficient."""
    h = calc_heat_transfer_coefficient(q_wall=500.0, T_wall=350.0, T_bulk=300.0)
    expected = 500.0 / 50.0
    assert abs(h - expected) < 1e-6
    print("[OK] calc_heat_transfer_coefficient")


def test_calc_heat_transfer_coefficient_zero():
    """Test calc_heat_transfer_coefficient with zero delta_T."""
    h = calc_heat_transfer_coefficient(q_wall=500.0, T_wall=300.0, T_bulk=300.0)
    assert h == 0.0
    print("[OK] calc_heat_transfer_coefficient zero delta_T")


def test_calc_total_heat_balance():
    """Test calc_total_heat_balance energy conservation check."""
    result = calc_total_heat_balance(Q_in=1000.0, Q_out=500.0, Q_stored=500.0)
    assert result["balance"] == 0.0
    assert result["is_conserved"] is True

    result_fail = calc_total_heat_balance(Q_in=1000.0, Q_out=400.0, Q_stored=100.0)
    assert result_fail["is_conserved"] is False
    print("[OK] calc_total_heat_balance")


def test_calc_temperature_contour():
    """Test calc_temperature_contour returns levels."""
    T = np.random.uniform(300, 400, size=(10, 10))
    result = calc_temperature_contour(T, levels=5)
    assert len(result["levels"]) == 5
    assert result["T_min"] >= 300
    assert result["T_max"] <= 400
    assert result["delta_T"] > 0
    print("[OK] calc_temperature_contour")


def test_calc_thermal_resistance():
    """Test calc_thermal_resistance."""
    R = calc_thermal_resistance(T_hot=400.0, T_cold=300.0, Q_total=100.0)
    assert abs(R - 1.0) < 1e-6
    print("[OK] calc_thermal_resistance")


def test_calc_thermal_resistance_zero_flux():
    """Test calc_thermal_resistance with zero heat flux."""
    R = calc_thermal_resistance(T_hot=400.0, T_cold=300.0, Q_total=0.0)
    assert R == 0.0
    print("[OK] calc_thermal_resistance zero flux")


# ---------------------------------------------------------------------------
# Step 1 — CoupledInterface
# ---------------------------------------------------------------------------

def test_coupled_interface():
    """Test CoupledInterface generates correct BC content."""
    interface = CoupledInterface(
        name="fluid_solid",
        fluid_region="fluid",
        solid_region="solid",
        heat_transfer_coefficient=50.0,
        thickness_layers=[0.001],
        kappa_layers=[50.0],
    )

    assert interface.name == "fluid_solid"
    assert interface.fluid_region == "fluid"
    assert interface.solid_region == "solid"
    assert interface.heat_transfer_coefficient == 50.0
    assert len(interface.thickness_layers) == 1

    fluid_bc = interface.get_fluid_bc_content()
    assert "coupledTemperature" in fluid_bc

    solid_bc = interface.get_solid_bc_content()
    assert "coupledTemperature" in solid_bc

    content = interface.get_content()
    assert "fluid_solid" in content
    print("[OK] CoupledInterface BC content")


# ---------------------------------------------------------------------------
# ChtSolver.setup_case integration test
# ---------------------------------------------------------------------------

def test_cht_solver_setup_case():
    """Test that ChtSolver.setup_case() writes all expected files."""
    from foampilot.cht import ChtSolver, FluidRegion, SolidRegion, CoupledInterface

    with tempfile.TemporaryDirectory() as tmp:
        case_path = Path(tmp) / "test_cht_setup"
        case_path.mkdir()

        fluid = FluidRegion(name="fluid", temperature=350.0, velocity=(10.0, 0.0, 0.0))
        solid = SolidRegion(name="solid", temperature=400.0, thermal_conductivity=50.0)
        interface = CoupledInterface(
            name="fluid_solid", fluid_region="fluid", solid_region="solid"
        )

        solver = ChtSolver(
            case_path=case_path,
            solver_name="chtMultiRegionFoam",
            regions=[fluid, solid],
            interfaces=[interface],
        )

        solver.setup_case()
        solver.write_case()

        # Check directories exist
        assert (case_path / "0" / "fluid").exists()
        assert (case_path / "0" / "solid").exists()
        assert (case_path / "constant" / "solid").exists()
        assert (case_path / "constant" / "fluid").exists()
        assert (case_path / "constant" / "regionInterfaces").exists()

        # Check field files
        assert (case_path / "0" / "fluid" / "T").exists()
        assert (case_path / "0" / "fluid" / "U").exists()
        assert (case_path / "0" / "solid" / "T").exists()
        # Solid should NOT have U
        assert not (case_path / "0" / "solid" / "U").exists()

        # Check solid properties
        assert (case_path / "constant" / "solid" / "thermophysicalProperties").exists()
        assert (case_path / "constant" / "solid" / "transportProperties").exists()
        assert (case_path / "constant" / "fluid" / "thermophysicalProperties").exists()

        # Check controlDict has regionSolvers
        control_dict = (case_path / "system" / "controlDict").read_text()
        assert "regionSolvers" in control_dict

        # Check interface file
        interface_file = (case_path / "constant" / "regionInterfaces" / "fluid_solid.dict")
        assert interface_file.exists()

        print("[OK] ChtSolver.setup_case() writes all expected files")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all CHT validation tests and report results."""
    tests = [
        test_cht_module_imports,
        test_cht_syntax_check,
        test_fluid_region_fields,
        test_solid_region_fields,
        test_fluid_region_thermophysical_properties,
        test_cht_solvers_in_modules,
        test_cht_solver_creation,
        test_cht_solver_invalid_name_raises,
        test_cht_solver_region_solvers_auto,
        test_controlDict_region_solvers,
        test_controlDict_region_solvers_roundtrip,
        test_controlDict_region_solvers_write,
        test_case_fields_manager_multi_region,
        test_case_fields_manager_no_regions,
        test_coupled_temperature_bc_class,
        test_external_temperature_bc_class,
        test_heat_flux_bc_class,
        test_fixed_temperature_bc_class,
        test_inlet_outlet_temperature_bc_class,
        test_symmetry_bc_class,
        test_total_temperature_bc_class,
        test_radiation_coupled_temperature_bc_class,
        test_calc_region_heat_flux,
        test_calc_interface_heat_flux,
        test_calc_nusselt_number,
        test_calc_nusselt_zero_delta,
        test_calc_thermal_boundary_layer_thickness,
        test_calc_heat_transfer_coefficient,
        test_calc_heat_transfer_coefficient_zero,
        test_calc_total_heat_balance,
        test_calc_temperature_contour,
        test_calc_thermal_resistance,
        test_calc_thermal_resistance_zero_flux,
        test_coupled_interface,
        test_cht_solver_setup_case,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"[FAIL] {test.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"CHT Tests: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
