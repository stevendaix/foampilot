# VoxCity Export Work

Sous-dossier de travail pour implémenter et comparer les voies d’export VoxCity vers OpenFOAM.

## Structure

```text
voxcity_export_work/
├── src/                    # Scripts de conversion
│   ├── read_voxcity.py     # Chargement VoxCity + extraction données
│   ├── vector_builder.py   # Construction Gmsh depuis building_gdf + DEM
│   ├── obj_builder.py      # Conversion OBJ → Gmsh/OpenFOAM
│   └── voxel_builder.py    # Conversion voxel brut → maillage
├── tests/                  # Tests de validation
│   ├── test_vector_path.py
│   ├── test_obj_path.py
│   └── test_voxel_path.py
└── output/                 # Résultats de comparaison
    ├── vector/
    ├── obj/
    ├── voxel/
    └── stl_snappy/
```

## Voies à comparer

1. **Vectorielle Gmsh** (`building_gdf` + DEM → `.msh` → `gmshToFoam`)
2. **OBJ remaillé** (`export_obj` → Gmsh → `.msh`)
3. **Voxel brut** (`voxels.classes` → maillage volumique)
4. **STL/snappyHexMesh** (référence existante)

## Métriques

- Précision géométrique
- Temps de maillage
- Qualité `checkMesh`
- Nombre de cellules
- Mémoire RAM

## Lancer les tests

```bash
PYTHONPATH=../../../foampilot/src python3 -m pytest tests/ -v
```
