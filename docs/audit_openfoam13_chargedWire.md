# Audit OF13 — legacy/electromagnetics/electrostaticFoam/chargedWire

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/electromagnetics/electrostaticFoam/chargedWire`.

L’Allrun officielle exécute `blockMesh`, puis `electrostaticFoam`. La mise en données comprend les champs `phi` et `rho`, `constant/physicalProperties` avec `epsilon0 = 8.85419e-12`, et les dictionnaires `controlDict`, `fvSchemes` et `fvSolution`. Le contrôle officiel est `endTime 0.02`, `deltaT 5e-05` et `writeInterval 100`.

Le runner `154_legacy_electrostaticFoam_chargedWire/run.py` importe les champs, propriétés et dictionnaires OF13 via les gestionnaires FoamPilot dédiés, puis exécute `blockMesh` et `electrostaticFoam` avec `run_command`. La validation OF13 atteint `End=0,02 s`; les équations de `phi` et `rho` sont résolues sans erreur fatale, avec des résidus finaux observés de l’ordre de `10^-9` ou inférieurs. Aucune nouvelle fonction d’API n’est nécessaire.
