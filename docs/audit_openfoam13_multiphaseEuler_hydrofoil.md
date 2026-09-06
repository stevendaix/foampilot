# Audit OF13 — multiphaseEuler/hydrofoil

La référence OpenFOAM 13 est un cas `multiphaseEuler` air/eau autour d’un profil hydrofoil NACA modifié à 6 degrés. Le maillage `blockMesh` est construit à partir d’un domaine structuré puis projeté sur trois surfaces STL : `NACAMOD66_6deg_tip.stl`, `NACAMOD66_6deg_lower.stl` et `NACAMOD66_6deg_upper.stl`. Les assets sont fournis compressés sous `constant/geometry/*.stl.gz`; FoamPilot les importe et les décompresse dans la structure de cas attendue. Le dictionnaire de phases conserve un diamètre gazeux constant de `0.0002`, une phase liquide continue et une tension de surface `0.071`.

L’Allrun de référence exécute `blockMesh`, puis `foamRun` en sériel. Le runner `225_multiphaseEuler_hydrofoil/run.py` importe par FoamPilot tous les champs `0/`, les dictionnaires `constant/system`, les STL et les fonctions de post-traitement (`yPlus(phase=liquid)`, pressions sur `hydrofoilLower`/`hydrofoilUpper` et création de graphes). Il reproduit la chaîne exacte sans ajouter `setFields`, décomposition MPI ou appels shell directs de logique de cas, sous environnement OF13 explicite.

La validation est stable mais limitée par le temps d’exécution. `blockMesh` réussit avec les surfaces projetées et `foamRun` progresse jusqu’à `Time≈0.100465 s` sur `0.2 s` avant l’expiration du plafond de 300 secondes. Les fractions `gas/liquid` restent bornées, les températures restent proches de `293 K`, le Courant maximal observé est proche de `0.80`, les solveurs linéaires réduisent les résidus et aucun `FOAM FATAL`, signal ou défaut de maillage n’apparaît.

Statut : **accepté avec réserve — calcul air/eau stable jusqu’à `Time≈0.1005 s` sur `0.2 s` au plafond de validation**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
