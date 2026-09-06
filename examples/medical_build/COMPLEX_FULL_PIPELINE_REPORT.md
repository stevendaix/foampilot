# Audit complet de la pipeline complexe

Ce rapport correspond à l’exécution réelle sur `analysis_sections.json` de l’aorte complexe : 8 branches et 800 sections, avec 32 points de profil utilisés pour la reconstruction.

## Résumé

| Étape | Résultat | Temps total |
|---|---:|---:|
| Chargement du contrat complexe | 8 branches, 800 sections | environ 0,55 s cumulé |
| STL manuel par sections | 8/8 générés | 3,139 s |
| Contrôle STL manuel par branche | 8/8 fermés après fusion des sommets | contrôlé |
| Build123d branche par branche | 8/8 `is_valid=true` | 53,291 s |
| Export STEP Build123d | 8/8 présents | inclus dans le temps Build123d |
| Export STL manuel | 8/8 présents | inclus dans le temps STL |
| Union anatomique globale | Non exécutée | nécessite un junction core global |

## Contrôle STL manuel

Chaque branche est reconstruite directement depuis les points ordonnés des sections. Les contours sont nettoyés, resamplés à 32 points, phase-alignés, triangulés entre stations et fermés aux deux extrémités.

Les huit branches ont obtenu `boundary_edges=0` et `nonmanifold_edges=0` dans le contrôle indexé. La validation indépendante Trimesh/VTK doit lire le STL avec fusion des sommets ; la lecture brute STL duplique les sommets des facettes et peut signaler à tort `watertight=false`.

Les fichiers sont dans `exports_complex/manual_stl/branch_XX.stl`. Ils font chacun 320 084 octets et constituent la sortie STL propre actuellement recommandée.

## Contrôle Build123d

Les huit lofts ont été construits branche par branche avec projection des profils sur leur plan tangent et 32 points maximum par section. Les huit solides OCC sont valides et tous les volumes signés sont positifs.

| Branche | Volume | STEP | Validité OCC |
|---:|---:|---:|---:|
| 0 | 47616,78 | Présent | true |
| 1 | 75602,73 | Présent | true |
| 2 | 49029,86 | Présent | true |
| 3 | 19502,01 | Présent | true |
| 4 | 3070,18 | Présent | true |
| 5 | 52787,94 | Présent | true |
| 6 | 11285,46 | Présent | true |
| 7 | 25970,50 | Présent | true |

Les STEP sont dans `exports_complex/build123d_step/`. Les STL natifs Build123d ont été produits avec succès pour les branches 0 à 3, mais leur tessellation OCC est très volumineuse, environ 90 à 119 Mo par branche. Pour éviter d’intégrer des artefacts lourds et de provoquer une pression mémoire excessive, la sortie STL recommandée pour les huit branches est le STL manuel contrôlé. Les branches 4 à 7 disposent de leur STEP validé et de leur STL manuel correspondant.

## Limite géométrique globale

Les huit branches ne sont pas fusionnées anatomiquement dans ce benchmark. Une `Compound` ou une concaténation de STL produirait huit composantes séparées et ne serait pas un volume fluide unique. L’union globale exige un noyau de bifurcation multi-blocs ou un raccord OCC avec recouvrement réel aux ports. Le validateur `GlobalBlockMesh` rejette volontairement cette situation tant que le carrefour n’est pas construit.

## Reproduction

```bash
cd foampilot/examples/medical_build
PYTHONPATH=../../foampilot/src:. python3 run_complex_full_pipeline.py \
  /path/to/analysis_sections.json \
  --output output/branch_00 \
  --branch 0 --cad-only

python3 section_stl_reconstruction.py \
  /path/to/analysis_sections.json \
  --output output/manual_stl --points 32
```

La validation OpenFOAM finale reste distincte : `surfaceCheck`, `blockMesh`, `snappyHexMesh -overwrite` et `checkMesh` doivent être exécutés dans une installation OpenFOAM réelle.
