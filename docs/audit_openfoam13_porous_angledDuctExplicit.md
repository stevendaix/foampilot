# Audit OF13 — legacy/incompressible/porousSimpleFoam/angledDuctExplicit

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/incompressible/porousSimpleFoam/angledDuctExplicit`.

L’Allrun officielle exécute `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/angledDuct`, puis `porousSimpleFoam`. La ressource commune crée un canal anglé de 22 000 cellules et les zones `inlet`, `porosity` et `outlet`.

La viscosité cinématique officielle est `nu=1.5e-05`, avec turbulence `kEpsilon`. La zone poreuse utilise `DarcyForchheimer`, la cellule-zone `porosity`, `d=(5e7 -1000 -1000)`, `f=(0 0 0)` et un repère cartésien tourné de 45°. Les champs `U`, `p`, `T`, `k`, `epsilon` et `nut` ainsi que les conditions inlet/outlet/parois sont importés sans modification.

Le contrôle officiel atteint `End=200 s` avec `deltaT=1` et écriture toutes les 10 étapes. Le runner `158_legacy_porousSimpleFoam_angledDuctExplicit/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`. La validation atteint `End=200 s`; les résidus restent finis, les erreurs de continuité restent contrôlées et aucun `FOAM FATAL` n’est observé. Aucune nouvelle fonction d’API n’est nécessaire.
