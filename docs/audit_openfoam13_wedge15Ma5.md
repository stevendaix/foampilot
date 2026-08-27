# Audit OF13 — legacy/lagrangian/dsmcFoam/wedge15Ma5

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/wedge15Ma5`.

L’Allrun officielle exécute `blockMesh`, `decomposePar`, `dsmcInitialise -parallel`, `dsmcFoam -parallel`, puis `reconstructPar -noLagrangian`. Le maillage 2D en wedge utilise `convertToMeters=1`, deux blocs `(20 40 1)` et `(40 40 1)`, une paroi `obstacle`, un patch `flow` et des faces `frontAndBack` de type `empty`. La décomposition officielle est simple avec `numberOfSubdomains=4`.

`dsmcProperties` conserve `nEquivalentParticles=5e12`, l’interaction murale `MaxwellianThermal`, les collisions `LarsenBorgnakkeVariableHardSphere` avec `Tref=273` et le nombre de relaxation `5`, ainsi que `InflowBoundaryModel FreeStream`. Les espèces sont `N2` et `O2`, avec densités de référence dans `FreeStreamCoeffs`; les propriétés moléculaires sont importées intégralement. `dsmcInitialiseDict` et le champ `boundaryU` imposent l’écoulement supersonique de référence `U=(1736 0 0)` à `T=300 K`; la paroi obstacle est à `T=550 K`.

Le contrôle officiel est `endTime=0,02 s`, `deltaT=2e-6 s` et écriture toutes les `1e-3 s`. Le runner `164_legacy_dsmcFoam_wedge15Ma5/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`, avec MPI à quatre domaines. La validation atteint `End=0,02 s`; `blockMesh`, `decomposePar`, les deux applications parallèles et `reconstructPar -noLagrangian` terminent sans `FOAM FATAL`. Le calcul conserve environ 27 500 particules, des collisions régulières et les avertissements FreeStream non fatals de localisation de particules. Aucune nouvelle fonction d’API n’est nécessaire.
