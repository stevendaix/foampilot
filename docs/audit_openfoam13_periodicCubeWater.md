# Audit OF13 — legacy/lagrangian/mdEquilibrationFoam/periodicCubeWater

Source locale : `/opt/openfoam13/tutorials/legacy/lagrangian/mdEquilibrationFoam/periodicCubeWater`.

L’Allrun officielle exécute `blockMesh`, `mdInitialise`, puis `mdEquilibrationFoam`. Le maillage est un cube périodique unique de `12x12x12` cellules, à l’échelle `2,462491658e-9 m`, avec six patches `cyclic` appariés selon les trois directions. Le cas utilise les dictionnaires OF13 `mdInitialiseDict`, `potentialDict` et `moleculeProperties` sans réécriture manuelle.

`mdInitialiseDict` définit une zone `liquid` de densité massique `1220`, température `300 K`, vitesse nulle et réseau `Ar` dans la référence source; `moleculeProperties` décrit les molécules multi-sites `water` et `water2`, avec sites H/H/O/M, masses, charges et positions de référence. `potentialDict` conserve le potentiel Lennard-Jones O–O, le potentiel électrostatique amorti, le tethering par ressort harmonique restreint et la gravité nulle. Le contrôle source est `endTime=5e-11 s`, `deltaT=1e-14 s` et écriture toutes les `5e-12 s`.

Le runner `166_legacy_mdEquilibrationFoam_periodicCubeWater/run.py` importe les champs, fichiers `constant` et dictionnaires `system` via les managers FoamPilot et exécute uniquement `blockMesh`, `mdInitialise` et `mdEquilibrationFoam` via `solver.run_command`. La validation OF13 initialise exactement `2457` molécules et reste stable avec une densité massique `980,014493831`. Elle atteint `Time=3,38e-13 s` après environ 67 secondes; le calcul a été arrêté proprement pour coût disproportionné avant `5e-11 s`. Aucun `FOAM FATAL` n’est observé. Statut : accepté avec réserve; aucune nouvelle API nécessaire.
