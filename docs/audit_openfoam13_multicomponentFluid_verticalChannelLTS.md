# Audit OF13 — multicomponentFluid/verticalChannelLTS

L’Allrun OpenFOAM 13 exécute `blockMesh`, `potentialFoam`, supprime `0/phi`, lance `foamRun`, puis `steadyParticleTracks`. Le cas est un canal vertical 3D avec injection de gouttelettes d’eau et cloud `cloudTracks`; il utilise le schéma LTS `localEuler` dans `fvSchemes`, un contrôle jusqu’à `endTime=300`, `deltaT=1`, `writeInterval=10`, `purgeWrite=20` et le dictionnaire `steadyParticleTracksDict` pour suivre les champs `d U T`.

Le runner `207_multicomponentFluid_verticalChannelLTS/run.py` importe par FoamPilot les champs, constantes, `cloudProperties`, positions, fonctions et dictionnaires de suivi, puis reproduit `blockMesh`, `potentialFoam`, la suppression gérée de `0/phi` par `remove_case_asset`, `foamRun` LTS et `steadyParticleTracks` sous environnement OF13 explicite. Aucun fichier de cas n’est supprimé par une commande shell directe.

La validation est complète. `blockMesh` et `potentialFoam` terminent correctement; le fichier intermédiaire `0/phi` est supprimé. `foamRun` atteint `End=300 s` avec le schéma `localEuler`. `steadyParticleTracks` traite les temps de 0 à 300 s, lit environ 6 200 à 6 400 particules selon le temps et écrit les fichiers VTK `particleTracks.vtk` correspondants. Aucun `FOAM FATAL`, problème de cloud, erreur LTS ou erreur de post-traitement n’est observé.

Statut : **validé OF13 — calcul LTS jusqu’à `End=300 s` et suivi particulaire complet**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et `BaseSolver.remove_case_asset(...)` pour la suppression gérée de `0/phi`.
