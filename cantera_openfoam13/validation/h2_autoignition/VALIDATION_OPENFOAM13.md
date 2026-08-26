# Validation OpenFOAM Foundation 13

**Date :** 26 août 2026  
**OpenFOAM :** Foundation 13 (`13-18870c24d21c`)  
**Cantera C++ :** 4.0.0a2, installé sous `$HOME/.local/cantera`  
**Cas :** `h2_autoignition/openfoam_case`

## Résultat

Le portage C++ `canteraFoam` compile avec `wmake` et s’exécute dans OpenFOAM 13. Le cas produit un maillage de **1000 cellules** et écrit **1000 états thermochimiques** dans `openfoam_case/canteraThermo.csv`.

Le lanceur exécute ensuite `icoFoam`. Le solveur démarre, avance jusqu’à `0.02 s`, et termine sans `FOAM FATAL ERROR`. Les erreurs de continuité observées sont nulles dans ce cas uniforme.

La référence Cantera H₂/air est également générée. Le pic OH est atteint vers `0.00032 s`, avec une température d’environ `2564.44 K`.

| Étape | Résultat |
|---|---|
| `wmake` du portage `canteraFoam` | PASS |
| `blockMesh` | PASS, 1000 cellules |
| `canteraFoam` | PASS, 1000 lignes produites |
| Référence Cantera | PASS |
| `icoFoam` OpenFOAM 13 | PASS |
| Erreur fatale OpenFOAM | Absente |

Le cas est un test d’intégration logicielle et thermochimique. Il ne constitue pas une reproduction quantitative complète du benchmark EBIdnsFoam/Taylor–Green de Zirwes et al.
