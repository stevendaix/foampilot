# Audit OF13 — multiRegion/CHT/misalignedDuct

Le cas de référence OpenFOAM 13 utilise un maillage unique contenant des zones cellulaires `fluid` et `solid`, avec géométrie décalée. Son Allrun exécute `blockMesh`, `splitMeshRegions -cellZonesOnly`, supprime les champs auxiliaires `cellToRegion`, crée les couples non conformes, prépare les fichiers ParaView puis lance `foamMultiRun`.

Le runner `176_multiRegion_CHT_misalignedDuct/run.py` importe les champs et dictionnaires OF13 puis reproduit cette chaîne avec FoamPilot : `blockMesh`, `splitMeshRegions -cellZonesOnly`, suppression gérée de `0/cellToRegion`, `0/fluid/cellToRegion` et `0/solid/cellToRegion`, `createNonConformalCouples`, `/opt/openfoam13/bin/paraFoam -touchAll` et `foamMultiRun`.

La validation a d’abord confirmé la séparation en deux régions et les couples, puis a révélé que l’étape `cellToRegion` de l’Allrun était nécessaire : sans elle, `createNonConformalCouples` échoue sur un champ auxiliaire présent dans `system/createNonConformalCouplesDict/patchFields`. Une méthode générique `BaseSolver.remove_case_asset` a donc été ajoutée, bornée au répertoire du cas, pour transposer ce type de nettoyage sans commande shell directe dans un runner.

Après correction, les couples `fluid1Fluid2`, `solid1Solid2`, `fluid1Solid2` et `fluid2Solid1` sont créés. `foamMultiRun` active les solveurs fluide et solide, atteint `Time=20 s` puis `End`, avec des erreurs de continuité de l’ordre de `1e-9` et aucun `FOAM FATAL`.

Statut : **validé OF13**.
