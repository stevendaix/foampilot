# Validation YADE–OpenFOAM 13

Cette arborescence contient deux cas CFD–DEM complets basés sur le couplage MPI `FoamCoupling` de YADE : `icoFoamYade` pour le couplage avec interpolation ponctuelle et `pimpleFoamYade` pour le couplage avec interpolation gaussienne et boucle PIMPLE.

## Pré-requis

L’environnement attendu est Ubuntu 24.04 avec OpenFOAM Foundation 13, OpenMPI et YADE compilé avec MPI. Les exécutables `icoFoamYade` et `pimpleFoamYade` doivent être présents dans `$FOAM_USER_APPBIN`. L’exécutable YADE doit être accessible via `$YADE_EXEC`, ou simplement via le `PATH`.

```bash
source /opt/openfoam13/etc/bashrc
export YADE_EXEC=/opt/yade/bin/yade
export FOAM_USER_APPBIN="$HOME/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin"
```

## Construction du couplage

Depuis la racine du dépôt :

```bash
source /opt/openfoam13/etc/bashrc
cd third_party/yade-openfoam-coupling/FoamYade/commYade && wmake
cd ../meshtree && wmake
cd .. && wmake
cd ../../icoFoamYade && wmake
cd ../pimpleFoamYade && wmake
```

Le build produit `libYadeComm.so`, `libMeshTree.so`, `libYadeFoam.so`, `icoFoamYade` et `pimpleFoamYade`. Les fichiers `Make/options` utilisent les bibliothèques et les headers réorganisés d’OpenFOAM 13 ; aucun modèle DEM n’est retiré.

## Cas icoFoamYade

```bash
cd validation/yade-openfoam13/icoFoamYade
./Allclean
./run.sh
```

Le script prépare le maillage, lance le script YADE MPI et le solveur `icoFoamYade`. La validation doit vérifier la production des champs couplés, l’absence d’erreur MPI et la décroissance cohérente de l’erreur de continuité.

## Cas pimpleFoamYade

```bash
cd validation/yade-openfoam13/pimpleFoamYade
./Allclean
./run.sh
```

Ce cas utilise `pimpleFoamYade` et le chemin d’interpolation gaussienne du couplage YADE. Il doit être contrôlé sur la convergence PIMPLE, la conservation du volume solide, l’échange des forces hydrodynamiques et l’absence de particules non localisées.

## Critères d’acceptation

Une validation est considérée réussie uniquement si le solveur OpenFOAM démarre avec le dictionnaire de cas, YADE rejoint le communicateur MPI, les deux codes échangent des données à chaque pas, les champs `alpha`, `uSource` et `uSourceDrag` sont mis à jour lorsque le cas les utilise, et les journaux ne contiennent ni `MPI_ABORT`, ni `was not found in FOAM`, ni erreur de segmentation. Les journaux complets sont conservés par le script de validation.
