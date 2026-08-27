# Audit OF13 — legacy/lagrangian/dsmcFoam/freeSpacePeriodic

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/freeSpacePeriodic`.

L’Allrun officielle exécute `blockMesh`, `dsmcInitialise`, puis `dsmcFoam`. Le maillage est un domaine cubique avec trois paires de patches périodiques : `xPeriodic_half0`/`xPeriodic_half1`, `yPeriodic_half0`/`yPeriodic_half1` et `zPeriodic_half0`/`zPeriodic_half1`.

`constant/dsmcProperties` conserve `nEquivalentParticles=1e12`, `WallInteractionModel SpecularReflection`, le modèle de collisions `LarsenBorgnakkeVariableHardSphere`, `Tref=273`, le nombre de relaxation `5`, aucune entrée (`InflowBoundaryModel none`) et les deux espèces `N2` et `O2` avec leurs masses, diamètres, degrés de liberté internes et paramètres `omega` officiels. Les champs DSMC (`boundaryT`, `boundaryU`, `dsmcRhoN`, `fD`, `iDof`, `internalE`, `linearKE`, `momentum`, `q`, `rhoM`, `rhoN`) et les fonctions `dsmcFields` sont importés sans modification.

Le contrôle officiel est `endTime=1e-3 s`, `deltaT=1e-6 s` et écriture toutes les `1e-4 s`. Le runner `161_legacy_dsmcFoam_freeSpacePeriodic/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`. La validation atteint `End=1e-3 s`; `dsmcInitialise` et `dsmcFoam` terminent sans erreur fatale, avec environ `64 009` particules DSMC, `6,4009e16` molécules équivalentes et des collisions régulières. Aucune nouvelle fonction d’API n’est nécessaire.
