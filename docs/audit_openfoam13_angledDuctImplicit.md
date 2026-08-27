# Audit OF13 — legacy/compressible/rhoPorousSimpleFoam/angledDuctImplicit

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/legacy/compressible/rhoPorousSimpleFoam/angledDuctImplicit`.

L’Allrun officielle exécute `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/angledDuct`, puis `rhoPorousSimpleFoam`. La ressource externe est donc importée dans `system/angledDuct` par le runner FoamPilot. Le maillage paramétré produit 22 000 cellules et les zones `inlet`, `porosity` et `outlet`.

Le cas utilise `constant/porosityProperties` avec le modèle `DarcyForchheimer`, la cellule-zone `porosity`, `d=(5e7 -1000 -1000)`, `f=(0 0 0)` et un repère cartésien dont l’axe est tourné de 45°. La thermodynamique est `heRhoThermo` avec gaz parfait, `Cp=1005`, masse molaire `28.9`, `mu=1.82e-05` et `Pr=0.71`. Le débit massique d’entrée est `0.1` et les conditions turbulentes sont celles de la référence.

Le contrôle officiel est `endTime 100`, `deltaT 1`, écriture toutes les 10 étapes. La formulation implicite est conservée par `fvSolution` de la référence. Le runner `153_legacy_rhoPorousSimpleFoam_angledDuctImplicit/run.py` utilise uniquement les gestionnaires FoamPilot `fields_manager`, `constant`, `system` et `run_command`. La validation atteint `End=100 s`, les résidus restent finis et aucun `FOAM FATAL` n’est observé. Aucune nouvelle fonction d’API n’est nécessaire.
