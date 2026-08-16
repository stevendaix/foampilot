# TBAD → OpenFOAM Pipeline

Pipeline complet pour générer des cas OpenFOAM à partir d'images médicales NIfTI de TBAD.

## Vue d'ensemble

```
NIfTI images → STL surfaces → CAD reconstruction → Volume mesh → OpenFOAM case
     (1)           (2)              (3)                (4)           (5)
```

## Installation

```bash
pip install numpy scipy trimesh vtk gmsh geomdl nibabel
```

## Utilisation rapide

```bash
# Pipeline complet pour patient 58
python3 run_tbad_case.py --patient 58

# Mesh seulement
python3 run_tbad_case.py --patient 58 --mesh-only

# OpenFOAM seulement (utilise mesh existant)
python3 run_tbad_case.py --patient 58 --of-only
```

## Structure du projet

```
examples/coa/
├── imageTBAD/                    # Images NIfTI originales
│   ├── 58_image.nii.gz
│   └── 58_label.nii.gz
├── data_preproc/
│   ├── extract_tbad_full.py      # Étape 1: NIfTI → STL
│   └── batch_extract.py          # Extraction batch
├── cad_reconstruction/
│   ├── vmtk_local/               # Implémentation VMTK locale
│   │   ├── vmtkcenterlines.py
│   │   ├── vmtkcenterlinesections.py
│   │   ├── vmtkdistancetocenterlines.py
│   │   ├── vmtkmeshgenerator.py
│   │   └── ...
│   ├── centerline_extractor.py
│   ├── section_extractor.py
│   ├── bspline_fitter.py
│   ├── occ_builder.py
│   └── cad_reconstruction.py     # Étape 2: CAD
├── run_full_pipeline.py          # Pipeline complet (étapes 1-4)
├── run_tbad_case.py              # Point d'entrée simple
├── openfoam_case.py              # Étape 5: Cas OpenFOAM
└── tbad_pipeline_config.example.json
```

## Étapes du pipeline

### Étape 1: Extraction STL
- **Entrée**: `imageTBAD/{patient_id}_image.nii.gz`, `{patient_id}_label.nii.gz`
- **Sortie**: `tbad_TL_walls.stl`, `tbad_FL_walls.stl`, `wall.stl`
- **Script**: `data_preproc/extract_tbad_full.py`

### Étape 2: Reconstruction CAD
- **Entrée**: `tbad_TL_walls.stl`
- **Sortie**: Centerlines, sections, B-splines, volume OCC
- **Script**: `cad_reconstruction/cad_reconstruction.py`

### Étape 3: Maillage volumique
- **Entrée**: STL + CAD
- **Sortie**: `mesh.msh` avec boundary layers
- **Outils**: Gmsh

### Étape 4: Cas OpenFOAM
- **Entrée**: Maillage
- **Sortie**: Cas OpenFOAM complet (`0/`, `constant/`, `system/`)
- **Outils**: foampilot

## Configuration

Copier `tbad_pipeline_config.example.json` vers `tbad_pipeline_config.json` :

```json
{
  "patient_id": 58,
  "data_dir": "imageTBAD",
  "output_dir": "pipeline_output",
  "centerline_spacing_mm": 2.0,
  "mesh": {
    "lc_min": 1.0,
    "lc_max": 4.0,
    "boundary_layers": 3,
    "boundary_layer_factor": 0.5
  },
  "fluid": {
    "name": "Blood",
    "rho": 1060,
    "nu": 3.77e-6
  },
  "solver": {
    "turbulence": "laminar",
    "transient": false,
    "endTime": 1
  }
}
```

## Commandes utiles

```bash
# Pipeline complet
python3 run_tbad_case.py --patient 58

# Mesh plus fin
python3 run_tbad_case.py --patient 58 --lc-min 0.5 --lc-max 2.0

# Plus de couches limites
python3 run_tbad_case.py --patient 58 --layers 5

# Lancer un cas OpenFOAM
cd pipeline_output/patient58/openfoam
foamRun

# Post-traitement
foamToVTK
paraFoam
```

## Tests

```bash
# Tests du module CAD
cd cad_reconstruction && PYTHONPATH=. pytest test_validation.py -v

# Test pipeline complet
python3 test_pipeline_simple.py
```

## Entrées/Sorties détaillées

### Entrées
| Type | Chemin | Description |
|------|--------|-------------|
| NIfTI | `imageTBAD/{id}_image.nii.gz` | Image CT |
| NIfTI | `imageTBAD/{id}_label.nii.gz` | Masque TL/FL |
| Config | `tbad_pipeline_config.json` | Configuration |

### Sorties
| Type | Chemin | Description |
|------|--------|-------------|
| STL | `pipeline_output/patient{id}/tbad_TL_walls.stl` | True Lumen |
| STL | `pipeline_output/patient{id}/tbad_FL_walls.stl` | False Lumen |
| STL | `pipeline_output/patient{id}/wall.stl` | Paroi externe |
| Mesh | `pipeline_output/patient{id}/mesh/mesh.msh` | Maillage volumique |
| OpenFOAM | `pipeline_output/patient{id}/openfoam/` | Cas OpenFOAM complet |

## Développement

Le pipeline s'inspire de :
- VMTK : centerlines, sections, mesh generator
- `building_aero/generate_wind_cases.py` : génération de cas Gmsh → OpenFOAM
- `building_aero/run_all_cases.py` : exécution et post-traitement

## TODO

- [ ] Support multi-région (TL + FL)
- [ ] Boundary layers adaptatives
- [ ] Export direct polyMesh depuis CAD
- [ ] Tests sur cohorte complète
- [ ] Validation CFD (résultats, convergence)
