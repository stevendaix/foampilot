# Neighborhood CFD Demo

Exemple complet d'un quartier réaliste avec VoxCity, Gmsh, OpenFOAM et foampilot.

## Pipeline
1. **VoxCity** : téléchargement des bâtiments + DEM pour une AOI Paris 15e
2. **Gmsh** : maillage vectoriel mono-fluide avec patches
3. **OpenFOAM** : export direct `polyMesh`, BCs avec profil log-wind
4. **foampilot** : solver `incompressibleFluid`, kEpsilon
5. **Post-traitement** : slices, Cp, statistiques

## Fichiers
- `config.json` — paramètres AOI VoxCity, domaine, maillage, solver
- `generate.py` — pipeline complet VoxCity → simulation
- `generate_synthetic.py` — fallback synthétique si VoxCity/EE échoue
- `postprocess.py` — post-traitement standalone
- `voxcity_postprocess.py` — utilitaires de post-traitement
- `verify_geometry.py` — vérification visuelle des footprints bruts vs traités
- `plot_voxcity_h5.py` — visualisation des bâtiments bruts depuis HDF5

## Usage

### Full pipeline
```bash
cd /home/steven/foampilot/examples/building_geo/neighborhood_demo
PYTHONPATH=../../../src python3 generate.py
```

### Use cached VoxCity data (skip download)
```bash
PYTHONPATH=../../../src python3 generate.py --use-cache
```

### Use a local VoxCity HDF5 file (no download at all)
```bash
PYTHONPATH=../../../src python3 generate.py --voxcity-h5 output/voxcity.h5
```

### Skip simulation
```bash
PYTHONPATH=../../../src python3 generate.py --skip-run
```

### Post-process only
```bash
PYTHONPATH=../../../src python3 postprocess.py --case neighborhood_case
```

### Verify geometry (raw vs processed footprints)
```bash
PYTHONPATH=../../../src:. python3 verify_geometry.py \
    --hdf5 output/voxcity.h5 \
    --output geometry_verification.png
```

### Fallback synthétique
```bash
PYTHONPATH=../../../src python3 generate.py --fallback-synthetic
```

## Prérequis
- Python 3.12+ pour VoxCity
- Earth Engine authentifié (`earthengine authenticate`)
- OpenFOAM 13
- Gmsh
