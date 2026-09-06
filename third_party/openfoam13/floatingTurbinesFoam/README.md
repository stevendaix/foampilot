# floatingTurbinesFoam pour OpenFOAM 13

Cette copie porte le module actuator-line historique `floatingTurbinesFoam` vers l’API `fvModel` d’OpenFOAM 13. Les classes `turbineALSource`, `axialFlowTurbineALSource`, `crossFlowTurbineALSource` et `actuatorLineSource` sont enregistrées dans la table runtime `fvModel` et leur constructeur suit la signature OpenFOAM 13 `(name, modelType, mesh, dict)`.

Le portage conserve les modèles aérodynamiques du dépôt source et introduit une couche locale de compatibilité `cellSetOption.H`. Cette couche traduit les callbacks historiques avec `fieldI` vers les surcharges field-based de `fvModel`, expose les champs via `addSupFields()` et fournit les callbacks `movePoints`, `topoChange`, `mapMesh` et `distribute` nécessaires à OpenFOAM 13. La sélection de cellules est préparée avec `fvCellZone`; la migration complète des dictionnaires `selectionMode/cellSet` vers la syntaxe native `cellZone` devra être vérifiée sur un cas réel.

## Compilation

```bash
. /opt/openfoam13/etc/bashrc
export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/$USER-$WM_PROJECT_VERSION"
cd third_party/openfoam13/floatingTurbinesFoam
./Allwmake
```

Le build produit `libturbinesFoam.so` dans `$FOAM_USER_LIBBIN` et lie `finiteVolume`, `sampling` et `meshTools`.

## Configuration fvModels

Le dictionnaire cible doit utiliser le type `axialFlowTurbineALSource`, `crossFlowTurbineALSource` ou `turbineALSource` sous `constant/fvModels`. Les coefficients historiques sont placés dans le sous-dictionnaire `<type>Coeffs`, conformément à `fvModel::coeffs()` d’OpenFOAM 13. La bibliothèque doit être chargée par `system/controlDict` :

```foam
libs ("libturbinesFoam.so");
```

## Vérifications

Le portage a été compilé avec OpenFOAM 13 après corrections successives de `fvModel`, `meshSearch`, des constructeurs `(mesh, dict)`, de `Time::timeName` et des conversions vectorisées de listes de profil. Il reste à exécuter un cas CFD minimal avec un vrai maillage, `checkMesh`, puis un solveur transitoire pour vérifier les sources de quantité de mouvement, la génération des fichiers de performance et la conservation du comportement aérodynamique historique.
