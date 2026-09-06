# Audit OF13 — multicomponentFluid/nc7h16

L’Allrun OpenFOAM 13 exécute `zeroDimensionalMesh`, convertit le mécanisme Chemkin NC7H16 avec `chemkinToFoam`, lance `foamPostProcess -func massFractions`, puis `foamRun`. Le mécanisme de référence est situé dans `/opt/openfoam13/test/chemistry/nc7h16/chemkin` et contient `chem.inp`, `therm.dat` et `transportProperties`. Le cas est zéro-dimensionnel, avec chimie ODE `Seulex`, `absTol=1e-12`, `relTol=0.1`, `initialChemicalTimeStep=1e-7`, `endTime=0.001`, `deltaT=1e-7`, `maxDeltaT=1e-3` et `writeInterval=5e-5`.

Le runner `201_multicomponentFluid_nc7h16/run.py` importe par FoamPilot les champs, propriétés et dictionnaires du tutoriel, puis importe explicitement les trois assets Chemkin OF13 depuis `test/chemistry/nc7h16/chemkin` dans un répertoire de cas géré par FoamPilot. Il reproduit ensuite `zeroDimensionalMesh`, `chemkinToFoam`, le post-traitement `massFractions` et `foamRun`. Le tracé gnuplot facultatif de l’Allrun n’est pas exécuté, conformément à la contrainte d’utiliser uniquement des commandes FoamPilot; les sorties numériques sont produites par les étapes OpenFOAM reproduites.

La validation est complète. `zeroDimensionalMesh`, `chemkinToFoam` et le post-traitement des fractions massiques terminent correctement. `foamRun` atteint `Time=0.001 s` et `End` en environ 18 secondes. La vitesse est nulle comme attendu pour le cas zéro-dimensionnel, les erreurs de continuité sont de l’ordre de `10^-15` et aucun `FOAM FATAL`, problème Chemkin ou erreur de champ n’est observé.

Statut : **validé OF13 — conversion NC7H16, post-traitement et calcul zéro-dimensionnel réussis**.

Le runner utilise l’extension générique `BaseSolver.run_command(environment=...)` documentée sous API-037; aucun changement d’API supplémentaire n’a été nécessaire.
