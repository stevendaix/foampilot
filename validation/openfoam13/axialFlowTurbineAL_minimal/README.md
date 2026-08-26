# Cas CFD minimal `floatingTurbinesFoam` / OpenFOAM 13

Ce cas valide le chargement runtime et l’application d’un modèle `axialFlowTurbineALSource` porté vers `fvModel`. Il utilise le solveur `incompressibleFluid`, un maillage cartésien `12 x 24 x 12` de **3 456 cellules**, un modèle de viscosité newtonien et un écoulement laminaire pour isoler le comportement actuator-line.

Le dictionnaire `constant/fvModels` charge trois pales, un moyeu et une tour. La sélection moderne `cellZone all` est accompagnée de `selectionMode cellZone` et `cellSet turbine` pour satisfaire les clés encore lues par la couche de compatibilité. La bibliothèque est chargée depuis `system/controlDict` sous le nom `libturbinesFoam.so`.

## Exécution

```bash
. /opt/openfoam13/etc/bashrc
export FOAM_USER_LIBBIN="$HOME/OpenFOAM/$USER-$WM_PROJECT_VERSION/platforms/$WM_OPTIONS/lib"
./Allrun
```

Le script exige que `libturbinesFoam.so` ait été compilée et soit disponible dans `$FOAM_USER_LIBBIN`.

## Résultat observé

`blockMesh` et `checkMesh` passent. Le calcul de deux pas temporels s’achève avec `foamRun -solver incompressibleFluid` et instancie les six objets `turbine`, `blade1`, `blade2`, `blade3`, `hub` et `tower`.

| Grandeur | t = 0,01 s | t = 0,02 s |
| --- | ---: | ---: |
| Coefficient de puissance `Cp` | 0,380156 | 0,451833 |
| Coefficient de traînée rotor | 0,593972 | 0,667113 |
| Erreur de continuité globale par pas | -4,74e-08 | -4,71e-08 |
| Nombre de cellules | 3 456 | 3 456 |

Ce test est un smoke test de fonctionnement et non une validation scientifique de maillage ou de convergence. Une validation physique devra comparer les forces et `Cp` à un cas de référence, augmenter la résolution et vérifier l’effet de la sélection `cellZone` sur un maillage partitionné.
