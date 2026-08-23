# État du portage YADE–OpenFOAM 13

## Réalisé

OpenFOAM Foundation 13 est installé dans `/opt/openfoam13` et son environnement est chargé par `source /opt/openfoam13/etc/bashrc`.

Le couplage YADE–OpenFOAM a été porté depuis la variante YADE la plus récente disponible dans `pkg/openfoam/coupling`. Les bibliothèques suivantes compilent avec OpenFOAM 13 et OpenMPI : `libYadeComm.so`, `libMeshTree.so` et `libYadeFoam.so`. Les adaptations incluent la suppression de `fvCFD.H`, la migration vers les headers séparés OpenFOAM 13, `meshSearch::New(mesh).findCell`, `dimKinematicViscosity`, `mesh.schemes().setFluxRequired`, les nouveaux MomentumTransportModels et la suppression du standard C++ obsolète `gnu++0x`.

Les deux solveurs complets compilent et se lient aux bibliothèques du couplage : `icoFoamYade` et `pimpleFoamYade`. Les commandes `icoFoamYade -help` et `pimpleFoamYade -help` fonctionnent, et `blockMesh` termine sans erreur dans les deux cas de validation.

Les cas `validation/yade-openfoam13/icoFoamYade` et `validation/yade-openfoam13/pimpleFoamYade` proviennent des exemples YADE correspondants. Ils conservent les scripts MPI, les dictionnaires OpenFOAM, les champs initiaux et les critères de validation CFD–DEM.

## Diagnostic corrigé et installation YADE

La compilation locale complète depuis les sources a été interrompue à environ 19 % dans `pkg_common/Grid.cpp` par le signal `Terminated` envoyé à `cc1plus`. Les journaux noyau consultés ne montrent pas de message explicite `OOM` ou `Killed process`; cette compilation locale ne permettait donc pas, à elle seule, d’affirmer une limite de RAM.

Le guide officiel YADE indique qu’Ubuntu 24.04 doit utiliser les paquets quotidiens ou la compilation depuis les sources. Le dépôt officiel a donc été ajouté et `yadedaily` a été installé avec succès. `yadedaily --version`, l’import `from yade import mpy` et la sonde batch officielle retournent un état OK. Le module `yade.mpy` est disponible dans `/usr/lib/x86_64-linux-gnu/yadedaily/py/yade/mpy.py`.

Les contrôles officiels `yadedaily --check` et `yadedaily --test` ont été exécutés. Le test global signale une erreur indépendante liée à l’environnement Python/Numpy (`numpy.core.multiarray failed to import`) dans `testPointInsidePolygon`, mais les tests MPI et le module `mpy` sont présents. Aucun modèle DEM n’a été retiré et aucune version simplifiée n’a été créée.

Les deux validations CFD–DEM ont ensuite été lancées avec `yadedaily-batch`. Elles démarrent OpenFOAM 13, échangent les données avec YADE, produisent les champs temporels et affichent les bilans de continuité. L’arrêt MPI final `MPI_ABORT` est lié au mécanisme de terminaison existant `killMPI`, et non à un manque de mémoire ; il doit être traité séparément si l’on exige un code retour nul strict du script.

## Commandes de validation

```bash
source /opt/openfoam13/etc/bashrc
export FOAM_USER_APPBIN="$HOME/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin"
export PATH="$FOAM_USER_APPBIN:$PATH"
export YADE_BATCH_EXEC=yadedaily-batch

cd validation/yade-openfoam13/icoFoamYade
./Allclean && ./run.sh

cd ../pimpleFoamYade
./Allclean && ./run.sh
```

Les critères physiques vérifiés sont l’échange MPI à chaque pas, l’absence de `FOAM FATAL ERROR`, la mise à jour des champs de couplage, la progression temporelle et la cohérence des bilans de continuité. Le code retour final doit encore être nettoyé pour éviter que `killMPI` transforme une fin de calcul normale en `MPI_ABORT` dans l’orchestrateur batch.
