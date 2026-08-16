# Suivi CAD Reconstruction TBAD

## Objectif
Pipeline complet VMTK-like pour TBAD : NIfTI → STL → CAD → Mesh → OpenFOAM.

## Pipeline en 4 étapes

| Étape | Entrée | Sortie | Script |
|-------|--------|--------|--------|
| 1. Extraction STL | `imageTBAD/{id}_image.nii.gz`, `{id}_label.nii.gz` | `tbad_TL_walls.stl`, `tbad_FL_walls.stl`, `wall.stl` | `data_preproc/extract_tbad_full.py` |
| 2. CAD Reconstruction | `tbad_TL_walls.stl` | Centerlines, sections, loft/sweep OCC | `cad_reconstruction/cad_reconstruction.py` |
| 3. Maillage volumique | STL + centerline | `mesh.msh`, `constant/polyMesh/` | `run_full_pipeline.py` (step 3) |
| 4. Cas OpenFOAM | Maillage | Cas complet prêt pour `foamRun` | `run_full_pipeline.py` (step 4) |

## Nouveautés (2026-08-13)

### 1. Correction bug critique step 3 — `surf_entities`
- Bug: variable `surf_entities` non définie, cassant complètement l'étape 3
- Fix: classification STL via `gmsh.model.mesh.classifySurfaces()` + `createGeometry()`
- Le STL est maintenant importé comme géométrie OCC propriétaire

### 2. Maillage adaptatif (Distance Field)
- Intégration du champ de distance centerline → paroi dans Gmsh
- Utilise Gmsh `Distance` + `Threshold` fields pour un maillage fin près de la paroi
- Le paramètre `centerline` est passé de l'étape 2 à l'étape 3
- Sauvegarde de la centerline en `centerline.npy` pour réutilisation

### 3. Correction du ThruSection / Loft
- Suppression du filtre `if i % 2 != 0: continue` qui sautait les sections impaires
- Ajout de l'approche **sweep** (`addPipe`) le long de la centerline comme fallback
- Meilleure qualité des courbes B-spline (résampling, tri, fermeture)
- Correction du typo d'import: `direct_openflow_exporter` → `direct_openfoam_exporter`

### 4. Validation checkMesh
- Nouvelle fonction `run_checkmesh()` dans `mesh_utils.py`
- Analyse automatique de la qualité du maillage (non-orthogonalité, skewness)
- Intégrée dans le pipeline après la génération du maillage

### 5. Rhéologie non-newtonienne (Carreau-Yasuda)
- Support du modèle Carreau-Yasuda pour le sang
- Paramètres physiologiques: nu0=13.96e-6, nuInf=3.77e-6, lambda=12.3, n=0.216, a=0.6
- Activation via `--non-newtonian` en ligne de commande
- Ajout de `CARREAU_YASUDA` dans `NonNewtonianModels` (foampilot library)
- Correction du bug `_process_coeffs` (KeyError sur `"crossPowerLawCoeffs"`)

### 6. Décimation STL (pyfqmr)
- Intégration de `pyfqmr` pour une décimation rapide et robuste
- Option `--decimate` et `--target-faces` en ligne de commande
- Fallback trimesh si pyfqmr non installé
- Ajout de `pyfqmr` dans `pyproject.toml` et `requirements.txt`

## Configuration

```json
{
  "patient_id": 58,
  "data_dir": "imageTBAD",
  "output_dir": "pipeline_output",
  "centerline_spacing_mm": 2.0,
  "mesh": {
    "lc_min": 0.5,
    "lc_max": 4.0,
    "boundary_layers": 3,
    "boundary_layer_factor": 0.5,
    "decimate": false,
    "target_faces": 50000
  },
  "fluid": {
    "name": "Blood",
    "rho": 1060,
    "nu": 3.77e-6,
    "non_newtonian": false
  },
  "rheology": {
    "model": "Newtonian",
    "carreau_yasuda": {
      "nu0": 13.96e-6,
      "nuInf": 3.77e-6,
      "lambda": 12.3,
      "a": 0.6,
      "n": 0.216
    }
  },
  "solver": {
    "turbulence": "laminar",
    "transient": false,
    "endTime": 1
  }
}
```

## Usage

```bash
# Pipeline complet
python3 run_tbad_case.py --patient 58

# Mesh seulement
python3 run_tbad_case.py --patient 58 --mesh-only

# OpenFOAM seulement
python3 run_tbad_case.py --patient 58 --of-only

# Options mesh
python3 run_tbad_case.py --patient 58 --lc-min 0.5 --lc-max 2.0 --layers 5

# Décimation STL + maillage adaptatif
python3 run_tbad_case.py --patient 58 --decimate --target-faces 50000

# Modèle Carreau-Yasuda (sang non-newtonien)
python3 run_tbad_case.py --patient 58 --non-newtonian
```

## Tests

```bash
# Tests de validation (13 tests)
cd cad_reconstruction && PYTHONPATH=. pytest test_validation.py -v
python3 test_pipeline_simple.py
```

## Statut

✅ Pipeline complet fonctionnel
✅ Extraction STL depuis NIfTI
✅ CAD reconstruction (centerlines + sections + loft/sweep)
✅ Maillage volumique Gmsh avec maillage adaptatif
✅ Validation checkMesh automatique
✅ Rhéologie non-newtonienne (Carreau-Yasuda)
✅ Décimation STL (pyfqmr)
✅ Export OpenFOAM direct
✅ Cas OpenFOAM avec BC
✅ Tests validation (13/13 passés)
✅ Configuration JSON
✅ CLI complète
