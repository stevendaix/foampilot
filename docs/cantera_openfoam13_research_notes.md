# Notes de recherche — Cantera/OpenFOAM 13

## Références consultées

- Dépôt Cantera : https://github.com/Cantera/cantera
- Article : https://link.springer.com/article/10.1007/s10494-023-00449-8
- Article fourni par l’utilisateur : https://www.researchgate.net/publication/371952800_Assessment_of_Numerical_Accuracy_and_Parallel_Performance_of_OpenFOAM_and_its_Reacting_Flow_Extension_EBIdnsFoam

## Points établis

Le dépôt Cantera est une suite de thermodynamique, cinétique chimique et transport. La page GitHub consultée indique une branche principale active et une organisation comprenant notamment `include/cantera`, `src`, `interfaces`, `samples` et `data`.

L’article de Zirwes et al. (2023), publié dans *Flow, Turbulence and Combustion*, présente une suite de benchmarks pour évaluer la précision numérique et les performances parallèles d’OpenFOAM et d’EBIdnsFoam sur des écoulements réactifs. Le résumé mentionne des écoulements incompressibles, la conduction thermique, la diffusion multi-espèces et un cas de flamme hydrogène couplée à un vortex de Taylor–Green. Les cas OpenFOAM sont annoncés comme publics.

FoamPilot contient déjà une infrastructure OpenFOAM 13 et des cas de couplage YADE, mais aucune intégration Cantera détectée dans les recherches initiales. Le dépôt suit une philosophie de cas générés et reproductibles et contient des tests OpenFOAM 13 sous `foampilot/test/openfoam13` ainsi que des guides d’installation.

## Décision de conception à confirmer dans le code

L’intégration doit rester compatible avec OpenFOAM Foundation 13. Une première cible robuste est un adaptateur de propriétés thermochimiques Cantera sous forme de bibliothèque/utilitaire OpenFOAM, accompagné d’un cas 0D/1D de validation qui compare une évolution homogène (auto-inflammation ou réacteur parfaitement agité) à une référence Cantera Python. Le cas doit éviter de prétendre reproduire EBIdnsFoam sans porter ce solveur et ses modèles spécifiques.
