# Préparation STL et validation OpenFOAM 13 pour le cas aortique complexe

## Objet du document

Ce guide décrit la chaîne recommandée pour transformer une surface vasculaire cappée en géométrie exploitable par OpenFOAM. Il distingue la surface anatomique de référence, les exports STL multi-régions et le maillage volumique produit par `snappyHexMesh`. Le cas de validation utilisé est l’aorte complexe comportant une entrée, huit sorties et une paroi.

> Un STL fermé n’est pas nécessairement un bon STL CFD. Il doit aussi être cohérent topologiquement, orienté, correctement partitionné et associé à des noms de patches compatibles avec les conditions limites.

## 1. Cas de validation

Le cas complexe contient neuf caps et une paroi :

| Rôle | Patches |
|---|---|
| Entrée | `inlet` |
| Sorties | `outlet_0`, `outlet_1`, `outlet_2`, `outlet_3`, `outlet_5`, `outlet_6`, `outlet_7`, `outlet_8` |
| Paroi | `wall` |

Les centerlines complexes sont stockées dans `case_complex/analysis/centerlines.vtp`. Elles contiennent huit trajectoires terminales associées au cap d’entrée commun `cap_4` et aux huit caps de sortie.

Le manifeste reproductible est généré avec :

```bash
cd foampilot
python3 examples/medical_build/build_complex_branch_manifest.py
```

Il ne faut pas confondre ce cas avec `test/vmtk_test_data/aorta-centerline-branches.vtp`, qui contient six cellules VTK organisées en deux chaînes principales et ne représente pas le cas anatomique à huit sorties.

## 2. Formats à conserver

Le format VTP doit être conservé comme format interne de travail. Il permet de garder les arrays `PatchId`, `GroupIds`, `CenterlineIds`, `TractIds`, `Blanking` et les identifiants de cellules. Le STL est un format d’échange pour OpenFOAM et ne doit être généré qu’après validation de la surface.

| Format | Usage recommandé | Information conservée |
|---|---|---|
| VTP | analyse et diagnostic | points, cellules, arrays, patches |
| STL multi-régions | entrée `snappyHexMesh` | triangles et noms `solid` |
| VTK/VTU | visualisation du maillage | cellules et patches OpenFOAM après conversion |
| STEP | CAD/Build123d | B-Rep, si la reconstruction CAD est validée |

Il faut éviter de concaténer naïvement dix STL indépendants avec `append`. Cette opération duplique souvent les interfaces wall/cap, perd les noms de régions et peut créer des baffles non subdivisibles.

## 3. Export STL multi-régions

L’export recommandé part directement du VTP qui contient `PatchId`. Chaque groupe de cellules est écrit dans un bloc ASCII STL distinct :

```text
solid outlet_0
  facet ...
endsolid outlet_0
solid inlet
  facet ...
endsolid inlet
solid wall
  facet ...
endsolid wall
```

Dans le cas complexe, le mapping utilisé est :

```python
PATCH_NAMES = {
    0: "outlet_0",
    1: "outlet_1",
    2: "outlet_2",
    3: "outlet_3",
    4: "inlet",
    5: "outlet_5",
    6: "outlet_6",
    7: "outlet_7",
    8: "outlet_8",
    9: "wall",
}
```

Le script de référence est :

```bash
python3 examples/medical_build/export_multiregion_stl.py
```

La surface doit être triangulée, contenir des normales cohérentes et ne comporter aucune paire de triangles identiques. Une paire de triangles ayant les mêmes sommets et des orientations opposées est particulièrement dangereuse : elle crée une face non-manifold et peut provoquer l’erreur `hexRef8::createInternalFaces` dans `snappyHexMesh`.

## 4. Contrôles préalables obligatoires

Les contrôles doivent être exécutés dans l’ordre suivant :

```bash
. /opt/openfoam13/etc/bashrc
surfaceCheck constant/triSurface/aorta_multiregion.stl
```

Le résultat attendu est :

```text
Surface has no illegal triangles.
Surface is closed. All edges connected to two faces.
Number of zones : 1
```

Il faut également vérifier avec un lecteur indépendant, par exemple PyVista :

```python
import pyvista as pv
mesh = pv.read("aorta_multiregion.stl")
print(mesh.n_points, mesh.n_cells)
print(mesh.volume, mesh.area, mesh.n_open_edges)
```

Une surface peut être déclarée fermée par une bibliothèque et être rejetée par OpenFOAM si elle contient des triangles dupliqués ou des orientations contradictoires. `surfaceCheck` est donc le contrôle de référence pour la compatibilité OpenFOAM.

## 5. Configuration minimale snappyHexMesh

Le cas validé utilise un bloc de fond couvrant la surface :

```text
(-145 -35 -10) à (-25 285 60)
```

Le `locationInMesh` doit se trouver dans le volume fluide, et non dans la paroi ou dans un cap. Une configuration minimale est :

```foam
castellatedMesh true;
snap true;
addLayers false;

geometry
{
    aorta_surface
    {
        type triSurfaceMesh;
        file "aorta_multiregion.stl";
    }
}

castellatedMeshControls
{
    maxLocalCells 100000;
    maxGlobalCells 500000;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {
        aorta_surface
        {
            level (2 3);
            regions
            {
                inlet    { level (3 3); patchInfo { type patch; } }
                outlet_0 { level (3 3); patchInfo { type patch; } }
                outlet_1 { level (3 3); patchInfo { type patch; } }
                outlet_2 { level (3 3); patchInfo { type patch; } }
                outlet_3 { level (3 3); patchInfo { type patch; } }
                outlet_5 { level (3 3); patchInfo { type patch; } }
                outlet_6 { level (3 3); patchInfo { type patch; } }
                outlet_7 { level (3 3); patchInfo { type patch; } }
                outlet_8 { level (3 3); patchInfo { type patch; } }
                wall     { level (2 2); patchInfo { type wall; } }
            }
        }
    }
    resolveFeatureAngle 30;
    locationInMesh (-100 100 25);
}
```

Dans OpenFOAM 13, `errorReduction` et `nSmoothScale` doivent être présents dans `meshQualityControls` lorsque la phase de snapping utilise la réduction d’erreur :

```foam
meshQualityControls
{
    maxNonOrtho 70;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-9;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    errorReduction 0.75;
    nSmoothScale 4;
}
```

## 6. Procédure reproductible

Le cas de test est disponible dans :

```text
examples/medical_build/openfoam_case/
```

Pour le reconstruire :

```bash
cd foampilot
python3 examples/medical_build/openfoam_case/create_snappy_case.py
```

Pour lancer la chaîne complète :

```bash
cd examples/medical_build/openfoam_case
. /opt/openfoam13/etc/bashrc
surfaceCheck constant/triSurface/aorta_multiregion.stl
blockMesh
snappyHexMesh -overwrite
checkMesh
```

Le cas validé produit :

| Contrôle | Résultat obtenu |
|---|---:|
| Surface STL | fermée, sans triangles illégaux |
| `blockMesh` | réussi |
| Cellules finales | 231 628 |
| Points finaux | 308 222 |
| Patches | 11, incluant `outer` |
| Non-orthogonalité maximale | 69,43° |
| Non-orthogonalité moyenne | 12,71° |
| Skewness maximale | 2,119 |
| `snappyHexMesh` | `Finished meshing without any errors` |
| `checkMesh` | `Mesh OK` |

Les patches finaux sont préfixés par le nom de la surface :

```text
aorta_surface_inlet
aorta_surface_outlet_0
aorta_surface_outlet_1
aorta_surface_outlet_2
aorta_surface_outlet_3
aorta_surface_outlet_5
aorta_surface_outlet_6
aorta_surface_outlet_7
aorta_surface_outlet_8
aorta_surface_wall
```

Le patch `outer` appartient au bloc de fond et ne représente pas une frontière anatomique. Pour un cas CFD final, il faut définir les conditions limites sur les patches `aorta_surface_*` et décider explicitement si `outer` doit être supprimé, transformé ou conservé comme frontière externe du domaine.

## 7. Diagnostic de l’erreur `hexRef8::createInternalFaces`

L’erreur :

```text
nAnchors:2 facei:...
FOAM FATAL ERROR in hexRef8::createInternalFaces
```

apparaît lorsque la surface produit des baffles ou des intersections impossibles à raffiner. Les causes principales sont :

| Cause | Symptôme | Correction |
|---|---|---|
| triangle dupliqué | `illegal triangles` | supprimer ou reconstruire localement la triangulation |
| triangle opposé superposé | arêtes `>2 faces` | supprimer la double peau, puis recoudre localement |
| append de wall et caps | baffles et perte de régions | exporter directement depuis `PatchId` |
| surface ouverte | arêtes connectées à une seule face | corriger le cap ou la jonction |
| normales incohérentes | plusieurs zones de normales | réorienter par composante cohérente |
| `locationInMesh` incorrect | presque toutes les cellules supprimées | choisir un point dans le volume fluide |
| raffinement excessif | mémoire et cellules trop nombreuses | réduire les niveaux ou le bloc initial |

Il ne faut pas corriger ce problème en désactivant les contrôles de qualité. Un maillage qui passe grâce à des seuils relâchés peut rester impropre au calcul CFD.

## 8. Patches et conditions limites

Après `snappyHexMesh`, vérifier les noms dans `constant/polyMesh/boundary`. Les conditions limites doivent utiliser les noms effectivement présents :

```text
inlet  → aorta_surface_inlet
outlet → aorta_surface_outlet_0 ... aorta_surface_outlet_8
wall   → aorta_surface_wall
```

Les fichiers minimaux de simulation devront ensuite contenir `0/U`, `0/p` et `constant/transportProperties`. La viscosité cinématique `nu` doit être écrite explicitement dans `transportProperties`; l’absence de cette propriété rend le cas incomplet même si le maillage est valide.

## 9. Ce qui est recommandé et ce qui ne l’est pas

La méthode recommandée est :

```text
VTP avec PatchId
→ contrôle des doublons et normales
→ STL ASCII multi-régions
→ surfaceCheck
→ blockMesh
→ snappyHexMesh
→ checkMesh
→ définition des conditions limites
```

Il faut éviter :

```text
STL par branche
→ append naïf
→ fusion implicite de surfaces déjà contaminées
→ fill_holes global
→ validation uniquement par volume
```

Une validation globale par volume ne détecte pas nécessairement les erreurs locales de bifurcation. Les contrôles de surfaces, de patches et de maillage doivent être exécutés ensemble.

## 10. Références

[1]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM 13 pour Ubuntu — documentation officielle"

[2]: https://doc.openfoam.com/2306/tools/pre-processing/mesh/generation/snappyhexmesh/ "OpenFOAM — snappyHexMesh"

[3]: https://openfoam.org/documentation/user-guide/ "OpenFOAM User Guide"

[4]: https://github.com/vmtk/vmtk "VMTK — dépôt officiel"
