# Exemple complet `medical_build`

Le script `medical_build_end_to_end.py` montre le passage du contrat d’analyse vers les artefacts portables. Il est conçu pour fonctionner en environnement minimal et active les étapes CAD ou OpenFOAM lorsqu’elles sont disponibles.

## Matrice des formats

| Étape | Format | Fichier produit | Dépendance | Statut |
|---|---|---|---|---|
| Contrat d’analyse | JSON | `analysis_contract.json` | Python standard | Toujours |
| Branches numériques | NPZ compressé | `npz/branch_XX.npz` | NumPy | Toujours |
| Centerlines VTK XML | VTP | `centerlines.vtp` | VTK | Si VTK installé |
| Centerlines VTK legacy | VTK | `centerlines.vtk` | Python standard | Toujours |
| Sections détaillées | VTP/VTK | généré par l’exporteur de sections | VTK | Étape section dédiée |
| CAD | STEP/BREP/STL | généré par `Build123dReconstruction` | Build123d/OCC | Option `--cad` |
| Surface CFD | STL séparés | `inlet.stl`, `outlet_*.stl`, `wall.stl` | VTK ou pipeline surface | Cas complexe/exporteur |
| STL direct par sections | STL binaire | `branch_XX.stl`, `aorta_manual_sections.stl` | NumPy | `section_stl_reconstruction.py` |
| Maillage OpenFOAM | blockMesh | `system/blockMeshDict` | OpenFOAM pour validation | Cas complexe |
| Maillage OpenFOAM | snappyHexMesh | `system/snappyHexMeshDict` | OpenFOAM | Cas complexe |
| Validation | JSON/Markdown | `export_manifest.json`, `export_report.md` | Python standard | Toujours |

## Exécution sur le cas complexe

À partir d’un contrat JSON complet contenant les sections :

```bash
cd foampilot
PYTHONPATH=src python examples/medical_build/medical_build_end_to_end.py \
  examples/medical_build/case_complex/analysis/analysis_sections.json \
  --output /tmp/medical_build_export \
  --cad --openfoam
```

Le fichier `analysis_sections.json` complet n’est pas commité dans cet exemple car il contient les contours de toutes les sections. Il est produit par `export_structured_sections.py` à partir des sorties VMTK/MedicalBuild. Le cas contient en revanche les centerlines, les diagnostics, les inventaires et les résultats de benchmark.

## Validation

Après export, vérifier `export_manifest.json`, relire les NPZ avec NumPy et visualiser les VTP/VTK dans ParaView ou PyVista. Lorsque les points ordonnés des sections sont disponibles, le STL peut être reconstruit directement sans OCC : `section_stl_reconstruction.py` resample les contours, verrouille leur phase, triangule les bandes et ferme les extrémités. Le résultat doit ensuite être contrôlé par `verify_manual_stl.py` avec fusion des sommets avant de conclure à la fermeture. Pour le CAD, vérifier `is_valid`, l’orientation et le volume signé de chaque branche avant une union globale. Pour OpenFOAM, exécuter `blockMesh`, `surfaceFeatureExtract`, `snappyHexMesh -overwrite` puis `checkMesh`.

Les STL commités dans le cas complexe sont des artefacts de campagne et doivent être contrôlés dans une installation OpenFOAM réelle. La reconstruction propre des caps est fournie par `validation_scripts/rebuild_clean_cfd_stl.py` lorsque la surface non cappée source est disponible.
