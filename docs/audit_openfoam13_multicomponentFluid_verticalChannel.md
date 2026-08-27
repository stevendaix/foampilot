# Audit OF13 — multicomponentFluid/verticalChannel

L’Allrun OpenFOAM 13 exécute `blockMesh`, `potentialFoam`, supprime `0/phi`, lance `foamRun`, puis `particleTracks`. Le cas est un canal vertical 3D `multicomponentFluid` turbulent avec injection de gouttelettes d’eau par le patch `inletCentral`, un cloud réactif/thermique et des fonctions de moyenne de `H2O` et `T` au patch `outlet`. Le contrôle impose `endTime=0.5`, `deltaT=1e-5`, `writeInterval=0.01`, ajustement de pas et `maxDeltaT=1e-3`. Le dictionnaire `particleTracksDict` suit le cloud `cloud` en format raw.

Le runner `206_multicomponentFluid_verticalChannel/run.py` importe par FoamPilot les champs, constantes, `cloudProperties`, positions, dictionnaires de fonctions et `particleTracksDict`, puis reproduit `blockMesh`, `potentialFoam`, la suppression gérée de `0/phi` par `remove_case_asset`, `foamRun` et `particleTracks` sous environnement OF13 explicite. Aucun appel shell direct n’est utilisé pour la suppression ou la création de fichiers de cas.

`blockMesh` et `potentialFoam` terminent correctement; le fichier `0/phi` est supprimé avant le solveur. `foamRun` reste stable jusqu’à `Time≈0.246 s` sur `0.5 s` au plafond de 300 secondes, avec environ 10 150 parcels courants, plus de 23 000 parcels injectés cumulativement et des sorties `surfaceFieldValue` produites à l’outlet. Les erreurs de continuité restent maîtrisées et aucun `FOAM FATAL`, problème de cloud ou erreur de potentiel n’est observé. Le plafond intervient avant la fin de `foamRun`; `particleTracks` n’a donc pas pu être exécuté dans ce budget.

Statut : **accepté avec réserve — préparation et calcul cloud validés, limite de temps avant `End=0,5 s` et `particleTracks`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et `BaseSolver.remove_case_asset(...)` pour la suppression gérée du fichier intermédiaire.
