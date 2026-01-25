import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import shutil
import re
import warnings

# Importation des mocks et de la classe Boundary (simulée)
# Dans un environnement de test réel, on importerait directement les modules
# Ici, nous allons simuler l'importation de la classe Boundary et utiliser les mocks
# définis dans mock_config.py.

# --- Mocks pour les dépendances externes ---
# Nous allons réutiliser le contenu de mock_config.py directement ici pour simplifier
# l'exécution dans le sandbox, en supposant que les dépendances sont installées.

class ValueWithUnit:
    """Mock de la classe ValueWithUnit pour simuler les valeurs avec unités."""
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit
    
    def __str__(self):
        # Pour le formatage dans _format_config
        if isinstance(self.value, tuple):
            return f"({self.value[0]} {self.value[1]} {self.value[2]})"
        return str(self.value)

# Mock pour BOUNDARY_CONDITIONS_CONFIG
MOCK_BOUNDARY_CONDITIONS_CONFIG = {
    "kEpsilon": {
        "velocityInlet": {
            "U": {
                "default": {"type": "fixedValue", "value": "uniform (${velocity})"},
                "withTurbulence": {"type": "fixedValue", "value": "uniform (${velocity})"},
            },
            "k": {
                "default": {"type": "fixedValue", "value": "uniform ${k_value}"},
                "withTurbulence": {"type": "fixedValue", "value": "uniform ${k_value}"},
            },
            "epsilon": {
                "default": {"type": "fixedValue", "value": "uniform ${epsilon_value}"},
                "withTurbulence": {"type": "fixedValue", "value": "uniform ${epsilon_value}"},
            },
            "p": {
                "default": {"type": "zeroGradient"},
            },
            "T": {
                "default": {"type": "fixedValue", "value": "uniform 300"},
            },
        },
        "pressureOutlet": {
            "U": {
                "default": {"type": "zeroGradient"},
            },
            "k": {
                "default": {"type": "zeroGradient"},
            },
            "epsilon": {
                "default": {"type": "zeroGradient"},
            },
            "p": {
                "default": {"type": "fixedValue", "value": "uniform 0"},
            },
            "T": {
                "default": {"type": "zeroGradient"},
            },
        },
        "wall": {
            "U": {
                "noSlip": {"type": "fixedValue", "value": "uniform (0 0 0)"},
                "slip": {"type": "slip"},
            },
            "k": {
                "noSlip": {"type": "wallFunction", "function": "kqRWallFunction"},
                "slip": {"type": "zeroGradient"},
            },
            "epsilon": {
                "noSlip": {"type": "wallFunction", "function": "epsilonWallFunction"},
                "slip": {"type": "zeroGradient"},
            },
            "p": {
                "default": {"type": "zeroGradient"},
            },
            "T": {
                "default": {"type": "fixedValue", "value": "uniform 300"},
            },
        },
        "symmetry": {
            "U": {"default": {"type": "symmetry"}},
            "k": {"default": {"type": "symmetry"}},
            "epsilon": {"default": {"type": "symmetry"}},
            "p": {"default": {"type": "symmetry"}},
            "T": {"default": {"type": "symmetry"}},
        },
    },
    "kOmegaSST": {
        "velocityInlet": {
            "U": {
                "default": {"type": "fixedValue", "value": "uniform (${velocity})"},
            },
            "k": {
                "default": {"type": "fixedValue", "value": "uniform ${k_value}"},
            },
            "omega": {
                "default": {"type": "fixedValue", "value": "uniform ${omega_value}"},
            },
            "p": {
                "default": {"type": "zeroGradient"},
            },
        },
        "wall": {
            "U": {
                "noSlip": {"type": "fixedValue", "value": "uniform (0 0 0)"},
            },
            "k": {
                "noSlip": {"type": "wallFunction", "function": "kqRWallFunction"},
            },
            "omega": {
                "noSlip": {"type": "wallFunction", "function": "omegaWallFunction"},
            },
            "p": {
                "default": {"type": "zeroGradient"},
            },
        },
    },
}

# Mock pour WALL_FUNCTIONS
MOCK_WALL_FUNCTIONS = {
    "kEpsilon": {
        "kqRWallFunction": {
            "fixedValue": {"type": "fixedValue", "value": "uniform 0"},
            "default": {"type": "kqRWallFunction", "value": "uniform 0"},
        },
        "epsilonWallFunction": {
            "fixedValue": {"type": "fixedValue", "value": "uniform 0"},
            "default": {"type": "epsilonWallFunction", "value": "uniform 0"},
        },
    },
    "kOmegaSST": {
        "kqRWallFunction": {
            "fixedValue": {"type": "fixedValue", "value": "uniform 0"},
            "default": {"type": "kqRWallFunction", "value": "uniform 0"},
        },
        "omegaWallFunction": {
            "fixedValue": {"type": "fixedValue", "value": "uniform 0"},
            "default": {"type": "omegaWallFunction", "value": "uniform 0"},
        },
    },
}

# Mock pour CONDITION_CALCULATORS
def mock_validate_inlet(turbulence_intensity, **kwargs):
    return 0.01 <= turbulence_intensity <= 0.1

def mock_calculate_inlet(turbulence_intensity, velocity, **kwargs):
    U_mag = (velocity[0].value**2 + velocity[1].value**2 + velocity[2].value**2)**0.5
    k_value = 1.5 * (U_mag * turbulence_intensity)**2
    epsilon_value = k_value**1.5 / 0.07
    omega_value = k_value**0.5 / 0.07
    
    return {
        "k_value": k_value,
        "epsilon_value": epsilon_value,
        "omega_value": omega_value,
    }

MOCK_CONDITION_CALCULATORS = {
    "velocityInlet": {
        "validate": mock_validate_inlet,
        "calculate": mock_calculate_inlet,
        "error_message": "Turbulence intensity must be between 0.01 and 0.1 for velocityInlet.",
    }
}

# Mock pour CaseFieldsManager
class MockCaseFieldsManager:
    def __init__(self, turbulence_model, energy_activated=False):
        self.turbulence_model = turbulence_model
        self.energy_activated = energy_activated

    def get_field_names(self):
        fields = ["U", "p"]
        if self.turbulence_model == "kEpsilon":
            fields.extend(["k", "epsilon"])
        elif self.turbulence_model == "kOmegaSST":
            fields.extend(["k", "omega"])
        
        if self.energy_activated:
            fields.append("T")
            
        return fields

# Mock pour OpenFOAMFile
class MockOpenFOAMFile:
    def __init__(self, field):
        self.field = field
        self.written_data = {}

    def write_boundary_file(self, field, boundaries, case_path):
        # Simuler l'écriture du fichier en stockant les données
        self.written_data[field] = {
            "boundaries": boundaries,
            "case_path": case_path
        }
        # Créer un fichier de sortie mock pour vérification
        output_path = Path(case_path) / "0" / field
        with open(output_path, "w") as f:
            f.write(f"// Mock file for {field}\n")
            for patch, config in boundaries.items():
                f.write(f"{patch}: {config}\n")

# Mock pour MockParent
class MockParent:
    def __init__(self, case_path):
        self.case_path = Path(case_path)
        (self.case_path / 'constant' / 'polyMesh').mkdir(parents=True, exist_ok=True)
        (self.case_path / '0').mkdir(exist_ok=True)

    def create_boundary_file(self, content):
        with open(self.case_path / 'constant' / 'polyMesh' / 'boundary', "w") as f:
            f.write(content)

# --- Fixtures Pytest ---

@pytest.fixture
def boundary_content():
    """Contenu de base pour le fichier polyMesh/boundary."""
    return """
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       polyBoundaryMesh;
        location    "constant/polyMesh";
        object      boundary;
    }
    (
        inlet
        {
            type            patch;
            nFaces          20;
            startFace       760;
        }
        outlet
        {
            type            patch;
            nFaces          20;
            startFace       780;
        }
        walls
        {
            type            wall;
            nFaces          400;
            startFace       800;
        }
        empty_patch
        {
            type            empty;
            nFaces          0;
            startFace       1200;
        }
        internal_patch
        {
            type            internal;
            nFaces          0;
            startFace       1200;
        }
    )
    """

@pytest.fixture
def mock_boundary_class():
    """Simule l'importation de la classe Boundary avec les dépendances patchées."""
    # Nous allons patcher les dépendances au niveau du module où Boundary est défini
    # Pour le sandbox, nous allons simuler l'importation et le patch
    
    # Importation dynamique de la classe Boundary
    import sys
    sys.path.append(str(Path("foampilot/src")))
    from foampilot.boundaries.boundaries_dict import Boundary
    
    # Patch des dépendances
    with patch('foampilot.boundaries.boundaries_dict.BOUNDARY_CONDITIONS_CONFIG', MOCK_BOUNDARY_CONDITIONS_CONFIG), \
         patch('foampilot.boundaries.boundaries_dict.WALL_FUNCTIONS', MOCK_WALL_FUNCTIONS), \
         patch('foampilot.boundaries.boundaries_dict.CONDITION_CALCULATORS', MOCK_CONDITION_CALCULATORS), \
         patch('foampilot.boundaries.boundaries_dict.OpenFOAMFile', MockOpenFOAMFile), \
         patch('foampilot.boundaries.boundaries_dict.ValueWithUnit', ValueWithUnit):
        
        yield Boundary
        
    sys.path.pop()

@pytest.fixture
def setup_boundary_manager(tmp_path, mock_boundary_class, boundary_content):
    """Fixture pour initialiser un Boundary Manager avec un cas de test."""
    test_dir = tmp_path / "test_case"
    test_dir.mkdir()
    parent = MockParent(test_dir)
    parent.create_boundary_file(boundary_content)
    
    fields_manager_ke = MockCaseFieldsManager(turbulence_model="kEpsilon")
    boundary_manager = mock_boundary_class(parent, fields_manager=fields_manager_ke, turbulence_model="kEpsilon")
    
    return boundary_manager

# --- Tests de la classe Boundary ---

class TestBoundary:

    # --- Cas de test pour __init__ ---
    def test_init_supported_model(self, mock_boundary_class, tmp_path):
        """Test l'initialisation avec un modèle de turbulence supporté (kEpsilon)."""
        parent = MockParent(tmp_path / "case1")
        fields_manager_ke = MockCaseFieldsManager(turbulence_model="kEpsilon")
        boundary = mock_boundary_class(parent, fields_manager=fields_manager_ke, turbulence_model="kEpsilon")
        assert boundary.turbulence_model == "kEpsilon"
        assert "U" in boundary.fields
        assert "k" in boundary.fields

    def test_init_unsupported_model(self, mock_boundary_class, tmp_path):
        """Test l'initialisation avec un modèle de turbulence non supporté (devrait lever ValueError)."""
        parent = MockParent(tmp_path / "case2")
        fields_manager_mock = MockCaseFieldsManager(turbulence_model="unsupported")
        with pytest.raises(ValueError, match="Turbulence model 'unsupported' is not supported."):
            mock_boundary_class(parent, fields_manager=fields_manager_mock, turbulence_model="unsupported")

    # --- Cas de test pour load_boundary_names ---
    def test_load_boundary_names_success(self, setup_boundary_manager, tmp_path):
        """Test le chargement réussi des noms et types de patchs."""
        boundary = setup_boundary_manager
        patches = boundary.load_boundary_names(tmp_path / "test_case")
        assert "inlet" in patches
        assert patches["inlet"] == "patch"
        assert "walls" in patches
        assert patches["walls"] == "wall"
        assert "empty_patch" in patches
        assert patches["empty_patch"] == "empty"
        assert "internal_patch" not in patches # Exclu car type 'internal'

    def test_load_boundary_names_file_not_found(self, setup_boundary_manager, tmp_path):
        """Test le cas où le fichier boundary est manquant (devrait lever FileNotFoundError)."""
        boundary = setup_boundary_manager
        # Supprimer le fichier boundary
        shutil.rmtree(tmp_path / "test_case" / 'constant' / 'polyMesh')
        (tmp_path / "test_case" / 'constant' / 'polyMesh').mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            boundary.load_boundary_names(tmp_path / "test_case")

    # --- Cas de test pour initialize_boundary ---
    def test_initialize_boundary_defaults(self, setup_boundary_manager):
        """Test l'application des conditions par défaut (wall, empty) et l'initialisation des champs."""
        boundary = setup_boundary_manager
        
        # Capturer les warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            boundary.initialize_boundary()
            
            # Vérifier le warning pour les patchs de type 'patch'
            assert len(w) == 1
            assert "inlet" in str(w[-1].message)
            assert "outlet" in str(w[-1].message)

        # Vérifier l'initialisation des champs
        expected_patches = {"inlet", "outlet", "walls", "empty_patch"}
        for field in boundary.fields_manager.get_field_names():
            assert set(boundary.fields[field].keys()) == expected_patches

        # Vérifier les conditions par défaut appliquées
        # 'walls' (type wall) -> set_condition("walls", "wall", friction=True)
        assert boundary.fields["U"]["walls"]["type"] == "fixedValue" # noSlip par défaut
        assert boundary.fields["k"]["walls"]["type"] == "wallFunction"
        
        # 'empty_patch' (type empty) -> set_condition("empty_patch", "symmetry")
        assert boundary.fields["U"]["empty_patch"]["type"] == "symmetry"
        
        # 'inlet' et 'outlet' (type patch) -> doivent être vides
        assert boundary.fields["U"]["inlet"] == {}
        assert boundary.fields["U"]["outlet"] == {}

    # --- Cas de test pour apply_condition_with_wildcard ---
    def test_apply_condition_with_wildcard(self, setup_boundary_manager):
        """Test l'application d'une condition à tous les patchs correspondant à un motif."""
        boundary = setup_boundary_manager
        boundary.initialize_boundary()
        
        velocity_in = (ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))
        boundary.apply_condition_with_wildcard(".*let", "velocityInlet", turbulence_intensity=0.05, velocity=velocity_in)
        
        # Vérifier que 'inlet' a la condition
        assert boundary.fields["U"]["inlet"]["type"] == "fixedValue"
        
        # Vérifier que 'outlet' a la condition
        assert boundary.fields["U"]["outlet"]["type"] == "fixedValue"

        # Vérifier que 'walls' n'a pas été modifié par le wildcard
        assert boundary.fields["U"]["walls"]["type"] == "fixedValue" # noSlip

    # --- Cas de test pour set_condition ---
    def test_set_condition_unsupported_type(self, setup_boundary_manager):
        """Test set_condition avec un type de condition non supporté (devrait lever ValueError)."""
        boundary = setup_boundary_manager
        boundary.initialize_boundary()
        with pytest.raises(ValueError, match="Condition type 'unsupportedCondition' is not defined for turbulence model 'kEpsilon'."):
            boundary.set_condition("inlet", "unsupportedCondition")

    def test_set_condition_validation_fail(self, setup_boundary_manager):
        """Test set_condition avec une validation qui échoue (devrait lever ValueError)."""
        boundary = setup_boundary_manager
        boundary.initialize_boundary()
        # turbulence_intensity=0.005 est en dehors de la plage [0.01, 0.1]
        velocity_mock = (ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))
        with pytest.raises(ValueError, match="Turbulence intensity must be between 0.01 and 0.1 for velocityInlet."):
            boundary.set_condition("inlet", "velocityInlet", turbulence_intensity=0.005, velocity=velocity_mock)

    def test_set_condition_success_kEpsilon(self, setup_boundary_manager):
        """Test set_condition réussi pour kEpsilon avec turbulence."""
        boundary = setup_boundary_manager
        boundary.initialize_boundary()
        
        velocity_in = (ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))
        boundary.set_condition("inlet", "velocityInlet", turbulence_intensity=0.05, velocity=velocity_in)
        
        # Vérifier l'application de la condition 'withTurbulence'
        assert boundary.fields["k"]["inlet"]["type"] == "fixedValue"
        
        # Vérifier la substitution de placeholder (valeur calculée)
        k_value_pattern = re.compile(r"uniform \d+\.\d+")
        assert k_value_pattern.match(boundary.fields["k"]["inlet"]["value"])
        
        # Vérifier que le champ T (température) est ignoré car non activé
        assert "T" not in boundary.fields

    def test_set_condition_success_kOmegaSST(self, mock_boundary_class, tmp_path):
        """Test set_condition réussi pour kOmegaSST."""
        test_dir = tmp_path / "test_case_sst"
        test_dir.mkdir()
        parent = MockParent(test_dir)
        parent.create_boundary_file(boundary_content())
        fields_manager_sst = MockCaseFieldsManager(turbulence_model="kOmegaSST")
        boundary = mock_boundary_class(parent, fields_manager=fields_manager_sst, turbulence_model="kOmegaSST")
        boundary.initialize_boundary()
        
        velocity_in = (ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))
        boundary.set_condition("inlet", "velocityInlet", turbulence_intensity=0.05, velocity=velocity_in)
        
        # Vérifier l'application de la condition pour omega
        assert boundary.fields["omega"]["inlet"]["type"] == "fixedValue"

    def test_set_condition_default_velocity(self, setup_boundary_manager):
        """Test set_condition s'assure que 'velocity' est toujours défini même si non fourni."""
        boundary = setup_boundary_manager
        boundary.initialize_boundary()
        
        # Appeler set_condition sans fournir 'velocity'
        boundary.set_condition("inlet", "pressureOutlet")
        
        # Vérifier que la condition a été appliquée (p)
        assert boundary.fields["p"]["inlet"]["type"] == "fixedValue"

    # --- Cas de test pour _resolve_field_config ---
    def test_resolve_field_config_wall_function_fixed_value(self, setup_boundary_manager):
        """Test _resolve_field_config pour une fonction de paroi avec 'velocity' fourni (fixedValue)."""
        boundary = setup_boundary_manager
        
        field_config = MOCK_BOUNDARY_CONDITIONS_CONFIG["kEpsilon"]["wall"]["k"]
        kwargs = {"velocity": (ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))}
        
        resolved = boundary._resolve_field_config(field_config, kwargs)
        
        assert resolved["type"] == "fixedValue"

    def test_resolve_field_config_wall_function_no_slip(self, mock_boundary_class):
        """Test _resolve_field_config pour une fonction de paroi sans 'velocity' (noSlip/default)."""
        # Nécessite une instance de Boundary pour accéder à _resolve_field_config
        parent = MockParent(Path("./tmp_resolve"))
        fields_manager_ke = MockCaseFieldsManager(turbulence_model="kEpsilon")
        boundary = mock_boundary_class(parent, fields_manager=fields_manager_ke, turbulence_model="kEpsilon")
        
        field_config = MOCK_BOUNDARY_CONDITIONS_CONFIG["kEpsilon"]["wall"]["k"]
        kwargs = {} # Pas de vitesse fournie
        
        resolved = boundary._resolve_field_config(field_config, kwargs)
        
        assert resolved["type"] == "kqRWallFunction"

    def test_resolve_field_config_slip(self, setup_boundary_manager):
        """Test _resolve_field_config pour le cas 'slip' (friction=False)."""
        boundary = setup_boundary_manager
        
        field_config = MOCK_BOUNDARY_CONDITIONS_CONFIG["kEpsilon"]["wall"]["U"]
        kwargs = {"friction": False}
        
        resolved = boundary._resolve_field_config(field_config, kwargs)
        
        assert resolved["type"] == "slip"

    # --- Cas de test pour _format_config ---
    def test_format_config_substitution(self, setup_boundary_manager):
        """Test _format_config avec substitution de placeholder."""
        boundary = setup_boundary_manager
        config = {"type": "fixedValue", "value": "uniform (${velocity})", "other": 123}
        # Simuler le formatage de ValueWithUnit
        params = {"velocity": ValueWithUnit((10, 0, 0), "m/s"), "k_value": 0.5}
        
        formatted = boundary._format_config(config, params)
        
        assert formatted["value"] == "uniform ((10 0 0))"
        assert formatted["type"] == "fixedValue"

    # --- Cas de test pour write_boundary_conditions ---
    def test_write_boundary_conditions_success(self, mock_boundary_class, tmp_path, boundary_content):
        """Test l'écriture des fichiers de conditions aux limites."""
        test_dir = tmp_path / "test_write"
        test_dir.mkdir()
        parent = MockParent(test_dir)
        parent.create_boundary_file(boundary_content)
        
        fields_manager_thermal = MockCaseFieldsManager(turbulence_model="kEpsilon", energy_activated=True)
        boundary = mock_boundary_class(parent, fields_manager=fields_manager_thermal, turbulence_model="kEpsilon")
        boundary.initialize_boundary()
        
        velocity_in = (ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"))
        boundary.set_condition("inlet", "velocityInlet", turbulence_intensity=0.05, velocity=velocity_in)
        
        boundary.write_boundary_conditions()
        
        # Vérifier que les fichiers ont été créés
        expected_fields = fields_manager_thermal.get_field_names()
        for field in expected_fields:
            file_path = test_dir / "0" / field
            assert file_path.exists()
            
            # Vérifier le contenu du fichier T (température)
            if field == "T":
                with open(file_path, "r") as f:
                    content = f.read()
                    # Vérifier la condition par défaut sur 'walls'
                    assert "walls: {'type': 'fixedValue', 'value': 'uniform 300'}" in content
                    # Vérifier la condition par défaut sur 'inlet'
                    assert "inlet: {'type': 'fixedValue', 'value': 'uniform 300'}" in content