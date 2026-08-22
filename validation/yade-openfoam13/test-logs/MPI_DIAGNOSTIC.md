# Diagnostic MPI — 22 août 2026

## Symptômes observés

Les cas `icoFoamYade` et `pimpleFoamYade` exécutent correctement les pas CFD–DEM : les journaux contiennent `Time = ...`, les résidus et les erreurs de continuité. À la fin, YADE signale toutefois :

```text
MPI_ABORT was invoked on rank 0 in communicator <Unknown>
with errorcode -100.
```

Le job YADE est alors marqué `FAILED`, code 39936, malgré l’avancement physique.

Un essai incohérent en mode série (`YADE_PARALLEL=false` alors que le timestepper et la décomposition restaient parallèles) produisait un segfault dans `GlobalStiffnessTimeStepper::computeTimeStep`, via `PMPI_Allreduce`. Les scripts ont ensuite été alignés sur le mode choisi.

## Cause confirmée

La méthode binaire réellement utilisée est celle de `/usr/lib/x86_64-linux-gnu/yadedaily/libpkg_openfoam.so`. Sa désassemblage montre que `yade::FoamCoupling::killMPI()` charge le communicateur stocké à l’offset `0xd8`, place `-100` dans le second argument, puis saute directement vers `MPI_Abort@plt` :

```text
00000000001a7780 <yade::FoamCoupling::killMPI()>:
  mov 0xd8(%rdi),%rdi
  mov $0xffffff9c,%esi
  jmp MPI_Abort@plt
```

`0xffffff9c` est la représentation signée de `-100`. L’appel final à `fluidCoupling.killMPI()` dans les deux scripts est donc, par conception de la version YADE installée, un abandon MPI forcé et non une fermeture normale.

Le code FoamPilot/OpenFOAM local possède une routine distincte `Foam::FoamYade::finalizeRun()` qui diffuse une valeur sur `interComm` et appelle `MPI_Finalize()` si cette valeur vaut `10`, mais cette routine n’est pas référencée ailleurs dans le code local. Elle ne peut donc pas être considérée comme le chemin de fermeture effectivement utilisé par `FoamCoupling::killMPI()`.

## Conséquence

Le couplage est fonctionnel pendant l’intégration, mais le critère actuel de fin de job est faux si l’on exige un code retour zéro. Le correctif retenu dans les cas FoamPilot est de ne plus appeler `killMPI()` par défaut après `RUN FINISH`. La sortie normale du processus YADE libère alors le communicateur au niveau du processus, sans abandon MPI. Le chemin historique reste reproductible avec `CFDEM_KILL_MPI=true`, mais il est explicitement marqué comme fermeture forcée et ne doit pas servir de critère de succès.

## Validation du correctif

Avec `CFDEM_NSTEPS=20`, `OPENFOAM_PROCS=2`, `YADE_PARALLEL=false` et le comportement par défaut (`CFDEM_KILL_MPI=false`), les deux cas ont terminé avec `launcher_rc=0`, `Master: RUN FINISH`, `status : 0 (OK)` et sans `MPI_ABORT`. Aucun processus YADE, OpenFOAM, `mpirun` ou `orted` orphelin n’a été observé après les tests.

## Corrections déjà appliquées aux cas

Les scripts de validation acceptent désormais `CFDEM_NSTEPS`, `OPENFOAM_PROCS` et `YADE_PARALLEL`. `Allclean` n’utilise plus l’utilitaire absent `foamCleanPolyMesh`. Les lanceurs vérifient le journal `scriptMPI.py.default.log` afin de ne plus masquer un job YADE marqué `FAILED`.

## Références

[1]: https://yade-dem.org/doc/FoamCoupling.html "YADE CFD-DEM FoamCoupling documentation"
[2]: https://yade-dem.org/doc/yade.wrapper.html "YADE wrapper reference"
[3]: https://gitlab.com/yade-dev/Yade-OpenFOAM-coupling "YADE OpenFOAM Coupling source repository"
[4]: https://openfoam.org/version/13/ "OpenFOAM 13 release page"
