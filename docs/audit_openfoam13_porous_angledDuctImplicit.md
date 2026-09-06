# Audit OF13 — legacy/incompressible/porousSimpleFoam/angledDuctImplicit

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/incompressible/porousSimpleFoam/angledDuctImplicit`.

L’Allrun officielle utilise la ressource commune `blockMesh/angledDuct`, puis exécute `porousSimpleFoam`. Le maillage et les zones `inlet`, `porosity` et `outlet` sont identiques au cas explicite. La différence de référence est portée par la formulation implicite des termes poreux dans `system/fvSolution`.

La mise en données conserve `nu=1.5e-05`, la turbulence `kEpsilon`, la zone `porosity` avec modèle `DarcyForchheimer`, `d=(5e7 -1000 -1000)`, `f=(0 0 0)` et le repère cartésien tourné de 45°. Les champs `U`, `p`, `T`, `k`, `epsilon` et `nut`, leurs conditions aux limites ainsi que les contrôles numériques sont importés sans modification.

Le runner `159_legacy_porousSimpleFoam_angledDuctImplicit/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`. La validation atteint `End=100 s`; les résidus de pression, turbulence et vitesse restent finis, les erreurs de continuité sont contrôlées et aucun `FOAM FATAL` n’est observé. Aucune nouvelle fonction d’API n’est nécessaire.
