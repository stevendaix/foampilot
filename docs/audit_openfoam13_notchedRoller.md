# Audit OF13 — multiRegion/CHT/notchedRoller

Le cas `notchedRoller` génère un maillage multi-région unique avec les zones `fluid`, `solid` et `roller`. L’Allrun OF13 exécute `blockMesh`, `createBaffles`, `splitBaffles`, `splitMeshRegions -cellZonesOnly`, supprime les champs auxiliaires `cellToRegion`, crée la zone cylindrique tournante dans la région fluide, puis crée les couples non conformes. La variante parallèle décompose toutes les régions avec `-copyZero -cellProc`, exécute `foamMultiRun -parallel` et reconstruit toutes les régions.

Le runner `178_multiRegion_CHT_notchedRoller/run.py` reproduit explicitement cette chaîne via FoamPilot. Les dictionnaires et champs des trois régions sont importés depuis OF13. Les nettoyages `cellToRegion` utilisent `BaseSolver.remove_case_asset`; `paraFoam -touchAll` passe par le script OF13 `/opt/openfoam13/bin/paraFoam`.

La validation crée les baffles, sépare les régions, crée la zone `rotating` et les interfaces non conformes. Les solveurs `fluid`, `solid` et `roller` démarrent avec quatre domaines. `foamMultiRun` atteint `Time=20 s` puis `End`; la reconstruction de toutes les régions atteint également `Time=20 s` et `End`. Les messages de faceZone pendant la séparation ne sont pas des erreurs fatales; aucun `FOAM FATAL` n’est observé.

Statut : **validé OF13**.
