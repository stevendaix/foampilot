# Audit OF13 — multiphaseEuler/injection

La référence OpenFOAM 13 est un cas sériel air/eau d’injection par entrée. L’Allrun exécute `blockMesh`, `setFields`, puis `foamRun`. Le champ initial global est `alpha.air=1`, `alpha.water=0`; la zone `water` est initialisée avec `alpha.air=0` et `alpha.water=1`. Les conditions d’entrée, de sortie, de vitesse et de température reproduisent l’injection d’eau dans l’air. Les propriétés de phase, tension de surface, transfert de quantité de mouvement, correction de traînée et contrôles MULES sont importés sans simplification.

Le runner `226_multiphaseEuler_injection/run.py` importe par FoamPilot tous les champs de `0/`, y compris les variantes `.orig`, ainsi que les dictionnaires `constant/system`. Il reproduit exactement la chaîne `blockMesh → setFields → foamRun` sous environnement OF13 explicite, sans appel shell direct pour la logique de cas.

La validation est complète. `blockMesh` et `setFields` réussissent; la zone `water` est correctement initialisée. `foamRun` atteint `Time=10 s` et `End` en environ 103 secondes. Les fractions air/eau restent bornées et leur somme est égale à 1; les températures restent proches de `300 K`, le Courant maximal observé est proche de `0.249`, les solveurs convergent à chaque pas et aucun `FOAM FATAL`, signal ou défaut de maillage n’apparaît.

Statut : **validé OF13 — injection air/eau jusqu’à `End=10 s`**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
