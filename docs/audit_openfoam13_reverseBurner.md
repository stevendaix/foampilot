# Audit OF13 — multiRegion/CHT/reverseBurner

La référence OpenFOAM 13 utilise deux régions, `gas` et `solid`. Le gaz est traité par `multicomponentFluid` avec chimie et combustion du méthane, espèces `N2`, `O2`, `CH4`, puis produits `H2O` et `CO2`; le solide est traité par le solveur thermique solide. Le contrôle impose `endTime=6`, `deltaT=1e-5`, `maxCo=0.2` et l’ajustement automatique du pas.

Le runner `179_multiRegion_CHT_reverseBurner/run.py` importe les champs et dictionnaires OF13, puis reproduit l’Allrun : `blockMesh`, `splitMeshRegions -cellZonesOnly`, création des fichiers ParaView pour `gas` et `solid`, `decomposePar -allRegions`, `setFields -region gas` en parallèle à quatre domaines, `foamMultiRun -parallel` et `reconstructPar -allRegions -newTimes`.

La validation confirme la génération des deux régions, la décomposition à quatre domaines et l’initialisation parallèle des champs `N2`, `O2` et `CH4`, avec le volume fuel correctement configuré. `foamMultiRun` résout les espèces réactives et les produits `H2O`/`CO2`, ainsi que les équations d’énergie, de quantité de mouvement et de pression, sans `FOAM FATAL` observé.

Le calcul est très coûteux : après environ 278 secondes, il atteint `Time≈0,73 s` sur `6 s` et le plafond global de 300 secondes interrompt la validation avant `End`. Les résultats sont stables dans l’extrait disponible; le cas est donc classé **accepté avec réserve — limite de temps**, et non comme validation complète.
