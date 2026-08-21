# `medical_build`

Ce sous-package fournit les contrats sérialisables et les backends de reconstruction pour les géométries vasculaires. Il sépare l’analyse géométrique compatible VMTK de la reconstruction CAD ou du maillage OpenFOAM.

## Architecture

`GeometryAnalysisData` contient les branches, les sections, les repères locaux, les contours phase-lockés, les mesures et les diagnostics. `Build123dReconstruction` consomme ces données pour créer des lofts projetés sur les plans tangents. `GlobalBlockMesh` fournit les primitives d’un assemblage direct `blockMesh` avec registre global de sommets, blocs, patches, faces internes et contrôle de connexité.

> Une série de branches reconstruites séparément n’est pas automatiquement un domaine CFD global. Le carrefour doit partager de vraies interfaces topologiques et le validateur doit refuser les composantes déconnectées.

## Validation

Les tests autonomes sont dans `tests/geometry/medical_build`. Ils vérifient le contrat des sections, la sérialisation JSON, les blocs dégénérés, les faces non-manifold et la connexité globale.

```bash
cd foampilot
PYTHONPATH=src pytest -q tests/geometry/medical_build
```

Les données de diagnostic VMTK sont conservées dans `tests/geometry/medical_build/medical_build_vmtk_diagnostics.json`. Les données lourdes telles que les surfaces VTP, les STEP complets et les STL générés ne sont pas intégrées dans Git ; elles doivent être récupérées depuis les artefacts de campagne ou Git LFS.

## Reconstruction Build123d

Les profils doivent être nettoyés, fermés et projetés sur leur plan local avant la création des wires OCC. Le loft lisse est compact et rapide, tandis que le loft réglé est plus conservateur. Chaque branche doit être validée avant une éventuelle union globale. Une orientation de volume négative doit être corrigée avant fusion.

## Sortie CFD

La surface finale destinée à OpenFOAM doit être contrôlée avec au moins un lecteur STL indépendant et, dans une installation OpenFOAM, avec `surfaceCheck`, `snappyHexMesh` et `checkMesh`. Il faut vérifier l’absence d’arêtes frontières non désirées, les faces non-manifold, les normales, la fermeture du volume et la présence des patches `inlet`, `outlet_*` et `wall`.

La procédure complète de préparation STL, d’export multi-régions et de validation OpenFOAM 13 est décrite dans [`STL_OPENFOAM_SNAPPY_GUIDE.md`](STL_OPENFOAM_SNAPPY_GUIDE.md). Le cas reproductible se trouve dans `examples/medical_build/openfoam_case`. Il couvre l’aorte complexe à huit sorties et documente les contrôles `surfaceCheck`, `blockMesh`, `snappyHexMesh` et `checkMesh`.

## Déformation locale optionnelle

Une déformation géométrique locale peut être appliquée entre l’analyse et la reconstruction à l’aide de `LocalDeformationSpec` et `apply_local_deformation`. Cette fonctionnalité s’inspire du paramétrage par sections d’AneuPy, mais conserve les contours réels extraits par notre pipeline au lieu de les remplacer par des cercles.

```python
from foampilot.geometry.medical_build import LocalDeformationSpec, apply_local_deformation

spec = LocalDeformationSpec(
    branch_ids=(2,),
    center_abscissa=12.0,
    sigma=3.0,
    amplitude=0.20,
    junction_protection=2.0,
)
deformed_data = apply_local_deformation(analysis_data, spec)
```

Par défaut, aucune déformation n’est appliquée. `apply_local_deformation(analysis_data, None)` retourne une copie équivalente et ne modifie jamais la référence. Le modèle radial utilise une gaussienne le long de l’abscisse de branche et réduit progressivement l’amplitude près des jonctions. Les données déformées peuvent ensuite être transmises aux backends STL, Build123d et `snappyHexMesh`.

La documentation détaillée et les limites actuelles sont décrites dans [`LOCAL_DEFORMATION.md`](LOCAL_DEFORMATION.md). La validation sur le contrat aortique complexe est reproductible avec `examples/medical_build/validate_local_deformation_real.py`.

## Limites connues

Le noyau multi-branches direct `blockMesh` est un contrat et un validateur de topologie ; il ne prétend pas encore résoudre automatiquement un carrefour anatomique à huit ports. La génération de ce carrefour doit être ajoutée avec des interfaces conformes aux tangentes et aux contours réels. Les benchmarks lourds sont reproduits par scripts externes et leurs rapports sont attachés aux campagnes de validation.
