# Tutoriel 9 : Transfert de chaleur conjugué (chtMultiRegionFoam)

## Objectif
Apprendre à configurer et exécuter un cas CHT (Conjugate Heat Transfer) multi-régions
avec foampilot et OpenFOAM 13.

## Cas de référence
OpenFOAM-13 : `tutorials/multiRegion/CHT/` (coolingSphere, heatedDuct)

## Physique
- Écoulement laminaire compressible d'air dans un conduit
- Mur solide en cuivre (380 W/(m·K)) chauffant le fluide
- Couplage fluide-solide : température continue à l'interface,
  flux de chaleur égal des deux côtés
- Température d'entrée fluide : 300 K
- Température mur chauffé : 350 K

## Workflow OpenFOAM 13 CHT

```
blockMesh → createZones → splitMeshRegions -cellZones
  → foamSetupCHT → foamDictionary (set T init)
  → chtMultiRegionFoam
  → foamToVTK → pyvista post-processing
```

> **Note OF-13** : `chtMultiRegionFoam` est un binaire autonome (pas un module
> `foamRun`). `foamSetupCHT` génère automatiquement les fichiers de champ
> et les propriétés matériaux. Le répertoire `constant/materialProperties`
> est utilisé au lieu de `constant/thermophysicalProperties` par région.

## Fichiers sources
- `run.py` — script principal utilisant l'API foampilot CHT
- `run_post.py` — post-traitement pyvista + calculs
- `block_mesh.json` — configuration géométrique (entrée pour `BlockMesher`)

## Fichiers générés (à l'exécution)
- `system/blockMeshDict` — maillage généré par `BlockMesher`
- `system/createZonesDict` — zones de cellules (via `OpenFOAMDictAddFile`)
- `system/controlDict` — contrôle temporel et `regionSolvers`
- `system/fvSchemes`, `system/fvSolution` — schèmes et solveurs
- `0/fluid/T`, `0/fluid/U`, `0/solid/T` — champs initiaux
- `constant/fluid/thermophysicalProperties`, `constant/solid/thermophysicalProperties` — propriétés matériaux

## Résultats attendus
- Température continue à l'interface (350 K)
- Profil de température dans le solide (quasi-isotherme)
- Numéro de Nusselt laminaire (Nu < 1)
- Coefficient de transfert de chaleur h ≈ 3-4 W/(m²·K)
- Balance énergétique conservée

## Exécution
```bash
cd foampilot/tutorials/09_CHT_heatedDuct
python run.py         # génération, simulation, conversion VTK
python run_post.py    # post-traitement et graphes
```

## Post-traitement
- `postProcessing/temperature_statistics.csv` — statistiques par région
- `postProcessing/temperature_profile.csv` — profil y-T fluide
- `postProcessing/temperature_profile_combined.csv` — profil combiné
- `postProcessing/fluid_temperature_contour.png` — isolignes fluide
- `postProcessing/solid_temperature_contour.png` — isolignes solide
- `postProcessing/cht_temperature_contour.png` — superpositon
- `postProcessing/CHT_Report.md` — rapport détaillé