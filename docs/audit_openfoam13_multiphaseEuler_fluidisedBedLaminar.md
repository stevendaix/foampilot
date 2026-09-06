# Audit OF13 — multiphaseEuler/fluidisedBedLaminar

L’Allrun OpenFOAM 13 de la variante laminaire exécute `blockMesh`, `setFields`, puis `foamRun` en sériel. Le cas conserve les phases `air` et `particles`, la zone `bed` initialisée avec `alpha.air=0.45` et `alpha.particles=0.55`, les champs thermiques air/particules, la théorie cinétique granulaire et les conditions aux limites Johnson–Jackson. La variante se distingue de `fluidisedBed` par `dragCorrection no` et `maxDeltaT=1e-5`; ces dictionnaires de référence sont importés sans simplification.

Le runner `224_multiphaseEuler_fluidisedBedLaminar/run.py` importe par FoamPilot les champs et les dictionnaires `constant/system`, puis reproduit exactement la chaîne `blockMesh → setFields → foamRun` sous environnement OF13 explicite. Il n’ajoute ni décomposition MPI ni reconstruction, conformément à l’Allrun de référence.

La validation est stable mais limitée par le temps d’exécution. `blockMesh` et `setFields` réussissent. `foamRun` progresse jusqu’à `Time≈1.1944 s` sur les `2 s` demandées avant l’expiration du plafond de 300 secondes. La somme des fractions est `1`, les fractions des phases restent bornées, les températures restent physiques (`particles≈599.61–600.008 K`, `air≈300–600.017 K`), le Courant maximal observé est proche de `0.289`, et aucun `FOAM FATAL`, signal ou défaut de convergence terminale n’apparaît.

Statut : **accepté avec réserve — calcul laminaire stable jusqu’à `Time≈1.1944 s` sur `2 s` au plafond de validation**.

Le runner utilise `BaseSolver.run_command(environment=...)` (API-037); aucun changement d’API supplémentaire n’a été nécessaire.
