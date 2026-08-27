# Cas CFD minimal `floatingTurbinesFoam` / OpenFOAM 13

Ce cas valide le chargement runtime et l’application d’un modèle `axialFlowTurbineALSource` porté vers `fvModel`. Il utilise le solveur `incompressibleFluid`, un maillage cartésien `12 x 24 x 12` de **3 456 cellules**, un modèle de viscosité newtonien et un écoulement laminaire pour isoler le comportement actuator-line.

Le dictionnaire `constant/fvModels` charge trois pales, un moyeu et une tour. La sélection moderne `cellZone all` est accompagnée de `selectionMode cellZone` et `cellSet turbine` pour satisfaire les clés encore lues par la couche de compatibilité. La bibliothèque est chargée depuis `system/controlDict` sous le nom `libturbinesFoam.so`.

## Exécution

```bash
# Définir FOAM_BASHRC vers le bashrc OpenFOAM Foundation 13 de votre installation.
# Exemple : export FOAM_BASHRC="$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc"
: "${FOAM_BASHRC:?Définissez FOAM_BASHRC vers OpenFOAM Foundation 13}"
. "$FOAM_BASHRC"
export FOAM_USER_LIBBIN="${FOAM_USER_LIBBIN:-$HOME/OpenFOAM/$USER-$WM_PROJECT_VERSION/platforms/$WM_OPTIONS/lib}"
./Allrun
```

Le script exige que `libturbinesFoam.so` ait été compilée et soit disponible dans `$FOAM_USER_LIBBIN`.

## Résultat observé

`blockMesh` et `checkMesh` passent. Le calcul de deux pas temporels s’achève avec `foamRun -solver incompressibleFluid` et instancie les six objets `turbine`, `blade1`, `blade2`, `blade3`, `hub` et `tower`. Le même cas a été décomposé et exécuté avec deux rangs MPI via `AllrunMPI`, puis les champs ont été reconstruits sans erreur.

| Grandeur | t = 0,01 s | t = 0,02 s |
| --- | ---: | ---: |
| Coefficient de puissance `Cp` | 0,380156 | 0,451833 |
| Coefficient de traînée rotor | 0,593972 | 0,667113 |
| Erreur de continuité globale par pas | -4,74e-08 | -4,71e-08 |
| Nombre de cellules | 3 456 | 3 456 |

En MPI à deux rangs, `Cp` vaut 0,380156 à `t = 0,01 s` et 0,451833 à `t = 0,02 s`. Les valeurs sont cohérentes avec le calcul séquentiel à mieux que `3e-7` sur `Cp` au second pas. L’erreur de continuité globale reste de l’ordre de `1e-7`, et `reconstructPar -latestTime` se termine correctement.

Ce test est un smoke test de fonctionnement et non une validation scientifique de maillage ou de convergence. Une validation physique devra comparer les forces et `Cp` à un cas de référence, augmenter la résolution et vérifier l’effet de la sélection `cellZone` sur un maillage partitionné.
