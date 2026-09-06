# Audit OF13 — legacy/incompressible/icoFoam/elbow

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/incompressible/icoFoam/elbow`.

L’Allrun officielle exécute `fluentMeshToFoam elbow.msh`, `icoFoam`, `foamMeshToFluent` et `foamDataToFluent`. Le maillage Fluent fourni est importé sans transformation supplémentaire; la conversion crée le maillage OpenFOAM et conserve les patches `wall-4`, `velocity-inlet-5`, `velocity-inlet-6`, `pressure-outlet-7` et `wall-8`, avec les faces avant/arrière 2D.

La viscosité cinématique officielle est `nu=0.01`. Le champ `U` impose `(1 0 0)` à `velocity-inlet-5` et `(0 3 0)` à `velocity-inlet-6`; la pression est fixée à zéro à `pressure-outlet-7`, les parois sont `noSlip` et les patches 2D sont `empty`. Le contrôle officiel est `endTime 10`, `deltaT 0.05` et écriture toutes les 20 étapes.

Le runner `157_legacy_icoFoam_elbow/run.py` importe les champs, la propriété physique, les dictionnaires et `elbow.msh` avec FoamPilot, puis exécute uniquement les quatre applications de l’Allrun via `run_command`. La validation atteint `End=10 s`, avec Courant maximal ≈`0,50`, erreurs de continuité de l’ordre de `10^-9` à `10^-10`, et les exports Fluent terminent sans `FOAM FATAL`. Aucune nouvelle fonction d’API n’est nécessaire.
