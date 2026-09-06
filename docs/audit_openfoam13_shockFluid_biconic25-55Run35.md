# Audit OF13 — shockFluid/biconic25-55Run35

La référence OpenFOAM 13 exécute `blockMesh`, `datToFoam constant/grid256.dat`, une fusion manuelle de `points.tmp` dans `constant/polyMesh/points`, `collapseEdges`, la conversion de `wedge1/wedge2` en patches `wedge`, puis `foamRun`. Le cas compressible `shockFluid` utilise un maillage biconique et `endTime=1e-3 s` avec `deltaT=1e-7 s`.

Le runner `239_shockFluid_biconic25-55Run35/run.py` reproduit cette chaîne sans logique shell de transformation : import FoamPilot de `grid256.dat`, appel de `datToFoam`, fusion par `BaseSolver.merge_mesh_points`, `collapseEdges`, conversion par `BaseSolver.update_mesh_patch_types`, puis `foamRun` sériel. Les champs compressibles `T/U/p`, les propriétés physiques, les fonctions et les dictionnaires OF13 sont importés par FoamPilot.

La validation confirme le maillage, l’import de la grille, la fusion ASCII des points et `collapseEdges` jusqu’à `End`. `foamRun` progresse sans erreur fatale jusqu’à `Time≈4,90e-4 s` sur `1e-3 s` au plafond de 300 secondes. Le champ `wallHeatFlux` est actif et la solution compressible évolue. OpenFOAM émet un avertissement de qualité indiquant que le patch `wedge2` peut ne pas être suffisamment planaire; aucun `FOAM FATAL`, NaN ou divergence n’est observé. La fin du calcul n’est pas atteinte dans le budget.

Statut : **accepté avec réserves — préparation et conversion des patches conformes, calcul stable jusqu’à `Time≈4,90e-4/1e-3 s`, réserve sur le temps et la planéité de `wedge2`**.

Évolutions API ajoutées et utilisées : `BaseSolver.merge_mesh_points` (API-039) et `BaseSolver.update_mesh_patch_types` (API-038). Leur comportement est générique et remplace les opérations shell de l’Allrun OF13.
