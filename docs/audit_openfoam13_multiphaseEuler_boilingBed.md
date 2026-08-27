# Audit OF13 — multiphaseEuler/boilingBed

L’Allrun OpenFOAM 13 exécute `blockMesh`, `extrudeMesh`, `createZones`, `setFields`, `decomposePar`, `foamRun -parallel` puis `reconstructPar -latestTime`. Le cas est un lit triphasique `gas/liquid/solid`, avec solide stationnaire, extrusion en coin, zone cellulaire `bed`, changement de phase liquide-gaz limité par transfert thermique et modèle d’ébullition de surface solide. Les dictionnaires contiennent aussi les contraintes de température des trois phases, les flux gaz/liquide aux frontières et les fonctions de continuité, de flux, de température et de flux thermique mural solide. Le calcul est contrôlé jusqu’à `endTime=7`, avec `deltaT=1e-4`, `writeInterval=0.2`, ajustement de pas et `maxDeltaT=0.005`.

Le runner `213_multiphaseEuler_boilingBed/run.py` importe par FoamPilot les champs suffixés, les dictionnaires `constant/system` et la totalité des propriétés d’ébullition et de transfert. Il reproduit `blockMesh`, `extrudeMesh`, `createZones`, `setFields`, `decomposePar` à 4 domaines Scotch, `foamRun -parallel` à 4 processus et `reconstructPar -latestTime`. La génération de cas et les lancements passent par FoamPilot; aucune commande shell directe de gestion de fichiers n’est utilisée dans le runner.

Une première exécution a révélé une omission du helper `run_parallel` dans le runner copié. Le helper a été ajouté localement au runner, sans changement d’API FoamPilot, puis la validation a été relancée. La chaîne complète réussit : `createZones` crée la zone `bed`, `setFields` initialise `alpha.gas`, `alpha.liquid` et `alpha.solid`, le calcul parallèle atteint `Time=7 s` et `End`, puis la reconstruction écrit les champs et sorties de phase, de transfert et de wall boiling au dernier temps. Aucun `FOAM FATAL`, problème MPI, défaut de reconstruction ou erreur de modèle d’ébullition n’est observé.

Statut : **validé OF13 — lit bouillant triphasique avec changement de phase et ébullition de surface jusqu’à `End=7 s`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037) et le helper local `run_parallel`; aucun changement d’API supplémentaire n’a été nécessaire.
