# Exemple spray VOF-to-DPM : `sprayCrossFlow`

Cet exemple porte dans **foampilot** le cas `crossFlow` du dépôt [atomizationFoam](https://github.com/imfd-stroemungsmechanik/atomizationFoam), qui est une référence GitHub directement orientée vers l’atomisation de spray. La géométrie représente une buse de liquide débouchant dans un écoulement d’air transversal. Le jet est d’abord résolu par VOF ; les fragments liquides détachés peuvent ensuite être convertis en parcels par `vofFragmentInjection`.

Le cas original ciblait des versions anciennes ou commerciales d’OpenFOAM et utilisait un solveur atomisationFoam monolithique. Cette version conserve sa géométrie STL, son écoulement cross-flow et ses conditions limites liquide/air, mais utilise le solveur `foamRun -solver incompressibleVoF` d’OpenFOAM 13 et le modèle `incompressibleVoFClouds` de foampilot. Le dictionnaire `phaseProperties` ainsi que les deux fichiers `physicalProperties.*` proviennent du cas laminaire OpenFOAM 13 ; le vieux `transportProperties` de la référence n’est volontairement pas utilisé.

## Exécution

Depuis la racine du dépôt :

```sh
cd examples/openfoam13/vof_to_dpm/example/sprayCrossFlow
./Allrun
```

Le script construit une copie temporaire du cas, exécute `blockMesh`, `snappyHexMesh` et le solveur jusqu’à `0.01 s`, puis vérifie la sélection du solveur, du modèle `incompressibleVoFClouds`, du cloud et la progression temporelle. Pour conserver le cas calculé et son journal, utiliser :

```sh
KEEP_CASE=1 ./Allrun
```

Le chemin du cas temporaire est alors imprimé sous la forme `CASE_DIR=...`.

## Conversion VOF-to-DPM

La conversion est activée dans `constant/fvModels` par `consumeAlpha true` et dans `constant/cloudProperties` par le modèle `vofFragmentInjection`. Le modèle détecte les composantes liquides détachées du champ `alpha.water`, construit un parcel équivalent et retire le volume converti du champ VOF. Le seuil `minVolume 1e-8` est volontairement bas pour rendre l’exemple utile à l’exploration de la fragmentation ; pour une étude de production, il doit être calibré avec la résolution de la maille et la taille minimale de goutte acceptable.

Le cas de référence utilise une surface STL et un raffinement local autour de la buse. Le dictionnaire `snappyHexMeshDict` a été adapté à la syntaxe OpenFOAM 13 (`type triSurface` avec une clé `file`) et la précision d’écriture a été augmentée pour être cohérente avec la tolérance de fusion de la maille.

## Validation

La validation minimale attend les marqueurs suivants dans `log.foamRun` :

| Vérification | Attendu |
|---|---|
| Solveur | `Selecting solver incompressibleVoF` |
| Modèle fvModel | `Selecting finite volume model type incompressibleVoFClouds` |
| Cloud | `Selecting parcelCloud collidingCloud` |
| Avancement | `Time = 0.01` |
| Conversion | lignes `VOF fragments detected:` ; des parcels apparaissent lorsque des fragments satisfont les filtres |

La référence GitHub est [imfd-stroemungsmechanik/atomizationFoam](https://github.com/imfd-stroemungsmechanik/atomizationFoam). Sa publication associée est Heinrich et Schwarze, « 3D-coupling of Volume-of-Fluid and Lagrangian particle tracking for spray atomization simulation in OpenFOAM », SoftwareX 11 (2020), [DOI 10.1016/j.softx.2020.100483](https://doi.org/10.1016/j.softx.2020.100483).
