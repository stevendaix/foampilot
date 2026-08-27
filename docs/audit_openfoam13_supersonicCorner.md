# Audit OF13 — legacy/lagrangian/dsmcFoam/supersonicCorner

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/supersonicCorner`.

L’Allrun officielle exécute `blockMesh`, `decomposePar`, `dsmcInitialise -parallel`, `dsmcFoam -parallel`, puis `reconstructPar -noLagrangian`. Le maillage utilise `convertToMeters=0,01`, deux blocs de tailles `(10 36 36)` et `(50 36 36)`, un coin supersonique et une paroi `walls`; la décomposition officielle est simple à `numberOfSubdomains=4`.

`dsmcInitialiseDict` initialise une population d’argon avec `numberDensities { Ar 1.0e20; }`, température `300 K` et vitesse `(1936 0 0)`. `dsmcProperties` conserve `nEquivalentParticles=1.2e12`, `MaxwellianThermal`, les collisions `VariableHardSphere` avec `Tref=273`, le modèle d’entrée `FreeStream` avec `Ar=1.0e20`, et les propriétés moléculaires d’argon de la référence. Les champs DSMC, la condition de température murale `1000 K` et les conditions de vitesse/flux sont importés sans modification.

Le contrôle officiel est `endTime=0,01 s`, `deltaT=1e-6 s` et écriture toutes les `1e-3 s`. Le runner `163_legacy_dsmcFoam_supersonicCorner/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`, avec `mpirun` à quatre domaines comme dans l’Allrun. La validation confirme `blockMesh`, `decomposePar`, `dsmcInitialise -parallel` et le démarrage de `dsmcFoam -parallel` sans `FOAM FATAL`; à `Time≈0,000132/0,01 s`, environ `845 000` particules sont présentes et les collisions sont régulières. Des avertissements FreeStream de localisation de particules restent non fatals. Le calcul est arrêté proprement pour coût disproportionné et classé accepté avec réserve. Aucune nouvelle fonction d’API n’est nécessaire.
