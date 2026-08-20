# Intégration NetworkX et snappyHexMesh dans `medical_build`

## Périmètre

Le chemin de production de `medical_build` utilise désormais les données Python d’analyse, le graphe NetworkX et l’API `SnappyMesher` de foampilot. VMTK n’est pas importé par la classe d’export. VMTK est conservé uniquement dans l’extra optionnel `vmtk-reference` et dans les scripts de comparaison.

## API publique

```python
from foampilot.geometry.medical_build import (
    MedicalSnappyExporter,
    SnappyExportConfig,
)

config = SnappyExportConfig(
    location_in_mesh=(223.0, 139.0, 24.0),
    n_surface_layers=5,
    first_layer_thickness=0.12,
)
case = MedicalSnappyExporter(config).export(
    patch_dir="analysis/cfd_patches",
    case_dir="case_snappy_aorta",
)
```

Le répertoire `patch_dir` doit contenir `inlet.stl`, au moins un fichier `outlet_*.stl` et `wall.stl`. La classe copie ces surfaces vers `constant/triSurface`, crée `blockMeshDict`, `surfaceFeaturesDict`, `snappyHexMeshDict`, les champs `0/U` et `0/p`, ainsi que `constant/transportProperties`.

## Patches et couches limites

Les patches sont nommés à partir des fichiers STL. Les couches limites sont activées uniquement sur le patch `wall`. Les entrées et sorties ne reçoivent aucune couche prismatique.

| Patch | Rôle par défaut |
|---|---|
| `inlet` | vitesse imposée dans `U`, gradient nul dans `p` |
| `outlet_*` | gradient nul dans `U`, pression imposée dans `p` |
| `wall` | `noSlip` dans `U`, aucune couche sur les caps |

## NetworkX

`VascularGraph` représente les caps et les segments comme un graphe. Il permet de contrôler la connectivité, les composantes, les terminaux, les bifurcations, les branches isolées et les longueurs d’arêtes avant l’export CFD. Le rapport console de la pipeline doit associer ces nœuds aux fichiers `inlet.stl` et `outlet_*.stl`.

## Validation

Les tests `medical_build` couvrent l’analyse, la fusion spatiale NetworkX et l’écriture du cas snappyHexMesh. Dans l’environnement de développement, les cinq tests ciblés passent. La commande `blockMesh` et la commande `snappyHexMesh` doivent encore être exécutées dans un environnement contenant OpenFOAM pour valider le maillage volumique final et la réussite effective de l’extrusion des couches.
