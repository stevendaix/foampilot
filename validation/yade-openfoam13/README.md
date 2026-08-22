# Validation YADE–OpenFOAM 13

Cette arborescence contient deux cas CFD–DEM complets basés sur le couplage MPI `FoamCoupling` de YADE : `icoFoamYade` pour le couplage avec interpolation ponctuelle et `pimpleFoamYade` pour le couplage avec interpolation gaussienne et boucle PIMPLE.

## Pré-requis

L’environnement attendu est Ubuntu 24.04 avec OpenFOAM Foundation 13, OpenMPI et YADE avec le module MPI `mpy`. Sur Ubuntu 24.04, le guide YADE recommande le paquet quotidien officiel `yadedaily`, qui fournit `yadedaily-batch` et `yade.mpy`. Les exécutables `icoFoamYade` et `pimpleFoamYade` doivent être présents dans `$FOAM_USER_APPBIN`.

```bash
source /opt/openfoam13/etc/bashrc
export YADE_BATCH_EXEC=yadedaily-batch
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

Le script prépare le maillage, lance `yadedaily-batch`; le module `mpy` crée ensuite le communicateur MPI du couplage et démarre `icoFoamYade` avec deux processus OpenFOAM. La validation doit vérifier la production des champs couplés et la cohérence de l’erreur de continuité.

## Cas pimpleFoamYade

```bash
cd validation/yade-openfoam13/pimpleFoamYade
./Allclean
./run.sh
```

Ce cas utilise `pimpleFoamYade` et le chemin d’interpolation gaussienne du couplage YADE. Il doit être contrôlé sur la convergence PIMPLE, la conservation du volume solide, l’échange des forces hydrodynamiques et l’absence de particules non localisées.

## Critères d’acceptation

Une validation est considérée physiquement réussie si le solveur OpenFOAM démarre avec le dictionnaire de cas, YADE rejoint le communicateur MPI, les deux codes échangent des données à chaque pas, les champs couplés sont mis à jour et les bilans de continuité restent cohérents. Dans la version actuelle de YADE, `killMPI` peut encore produire un `MPI_ABORT` final lors de la fermeture ; ce code retour d’orchestration est distinct de la convergence physique du cas et doit être nettoyé séparément.
