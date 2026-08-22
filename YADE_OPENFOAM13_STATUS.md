# État du portage YADE–OpenFOAM 13

## Réalisé

OpenFOAM Foundation 13 est installé dans `/opt/openfoam13` et son environnement est chargé par `source /opt/openfoam13/etc/bashrc`.

Le couplage YADE–OpenFOAM a été porté depuis la variante YADE la plus récente disponible dans `pkg/openfoam/coupling`. Les bibliothèques suivantes compilent avec OpenFOAM 13 et OpenMPI : `libYadeComm.so`, `libMeshTree.so` et `libYadeFoam.so`. Les adaptations incluent la suppression de `fvCFD.H`, la migration vers les headers séparés OpenFOAM 13, `meshSearch::New(mesh).findCell`, `dimKinematicViscosity`, `mesh.schemes().setFluxRequired`, les nouveaux MomentumTransportModels et la suppression du standard C++ obsolète `gnu++0x`.

Les deux solveurs complets compilent et se lient aux bibliothèques du couplage : `icoFoamYade` et `pimpleFoamYade`. Les commandes `icoFoamYade -help` et `pimpleFoamYade -help` fonctionnent, et `blockMesh` termine sans erreur dans les deux cas de validation.

Les cas `validation/yade-openfoam13/icoFoamYade` et `validation/yade-openfoam13/pimpleFoamYade` proviennent des exemples YADE correspondants. Ils conservent les scripts MPI, les dictionnaires OpenFOAM, les champs initiaux et les critères de validation CFD–DEM.

## Blocage d’exécution

La compilation YADE complet avec MPI a produit sa bibliothèque cœur `libyade.so`, puis a été arrêtée à environ 19 % lors de la compilation de `pkg_common/Grid.cpp` par le signal `Terminated` envoyé à `cc1plus`. Le sandbox dispose d’environ 3,8 Gio de RAM ; l’ajout de 8 Gio de swap ne fournit pas une capacité suffisante et stable pour terminer le build complet de YADE dans ce contexte.

Aucun modèle DEM n’a été retiré et aucune version simplifiée n’a été créée. Les validations CFD–DEM nécessitent l’exécutable YADE installé, notamment le module Python `yade.mpy`; elles ne doivent donc pas être déclarées réussies tant qu’elles n’ont pas été exécutées sur une machine disposant idéalement de 8 à 16 Gio de RAM.

## Commandes de validation à exécuter sur une machine adaptée

```bash
source /opt/openfoam13/etc/bashrc
export FOAM_USER_APPBIN="$HOME/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin"
export PATH="$FOAM_USER_APPBIN:$PATH"
export YADE_EXEC=/opt/yade/bin/yade

cd validation/yade-openfoam13/icoFoamYade
./Allclean && ./run.sh

cd ../pimpleFoamYade
./Allclean && ./run.sh
```

Les critères d’acceptation sont l’échange MPI à chaque pas, l’absence de `MPI_ABORT`, l’absence de `was not found in FOAM`, la mise à jour des champs de couplage et la cohérence des bilans de masse, volume solide, forces hydrodynamiques et quantité de mouvement.
