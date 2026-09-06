# Audit OF13 — multiRegion/CHT/coolingCylinder2D

Source locale : `/opt/openfoam13/tutorials/multiRegion/CHT/coolingCylinder2D`.

L’Allrun OF13 exécute `blockMesh`, `splitMeshRegions -cellZones`, puis `foamMultiRun`. Le maillage blockMesh contient les zones `fluid` et `solid` autour d’un cylindre, avec une interface couplée `fluid_to_solid/solid_to_fluid`. Les champs initiaux sont régionaux sous `0/fluid` et `0/solid`; les propriétés thermiques et les dictionnaires `fvSchemes`, `fvSolution`, `functions` sont importés sans perte.

Le contrôle source définit `regionSolvers { fluid fluid; solid solid; }`, `endTime=20`, `deltaT=0,01`, écriture toutes les `0,1` et pas non ajustable. Le runner `171_multiRegion_CHT_coolingCylinder2D/run.py` utilise uniquement les managers FoamPilot et `solver.run_command` pour importer les données puis exécuter les trois étapes de l’Allrun.

La validation OF13 crée correctement les deux régions, puis `foamMultiRun` atteint `Time=20 s` et `End`. Les solveurs fluide et solide traitent leurs équations, avec un nombre de Courant maximal observé autour de `3,67`, et aucun `FOAM FATAL`. Aucune nouvelle API nécessaire; statut validé OF13.
