# DTCMoving Overset — Foundation 13

Ce répertoire est le portage de travail du cas `DTCMoving_Overset` vers OpenFOAM Foundation 13. Il conserve séparément les sous-cas `background` et `hull`, leurs géométries et leurs champs initiaux, tout en remplaçant les dictionnaires OpenCFD incompatibles par la syntaxe Foundation 13.

## État de validation

| Élément | État |
|---|---|
| `background/system/blockMeshDict` | Validé avec `blockMesh`, 134064 cellules |
| `hull/system/blockMeshDict` | Validé avec `blockMesh`, 147200 cellules |
| `background/system/controlDict` | Porté vers `marineFoam` + `incompressibleVoF` |
| `background/constant/dynamicMeshDict` | Porté vers le mover Foundation 13 `motionSolver` + six-DoF rigide |
| `background/system/fvSolution` | Pression configurée avec `PBiCGStab`/`DILU` |
| `background/system/fvConstraints` | Déclare `marineOversetConstraint` pour les champs transportés |
| Interpolation overset | Disponible via `constant/marineOversetStencils` et la bibliothèque runtime |
| Couplage réellement multi-maillage | À implémenter dans l’étape suivante |

## Commandes de base

Depuis chaque sous-cas, exécuter `blockMesh`. La génération snappy doit ensuite être appliquée avec les surfaces placées dans `constant/triSurface`. La bibliothèque runtime doit être accessible via `FOAM_USER_LIBBIN` ou par son chemin dans `system/controlDict`.

Le dictionnaire `fvConstraints` est volontairement séparé du mover. Le mover Foundation 13 est responsable du déplacement six-DoF, tandis que `marineOversetConstraint` est appliquée au moment où le solveur assemble chaque équation. Cette séparation est nécessaire puisque le cycle `fvMeshMover` ne reçoit pas directement les objets `fvMatrix`.

## Limite scientifique actuelle

Le cas possède encore deux bases de maillage indépendantes, mais le runtime personnalisé ne réalise pas encore la fusion topologique ou le transfert inter-maillages pendant le mouvement. Avant une comparaison quantitative DTC, il faut construire les stencils à partir des centres de cellules des deux maillages, produire `zoneID` et `oversetCellStatus` dans le maillage de calcul retenu, puis vérifier la conservation de masse et la stabilité des corrections de pression.

## Mapping inter-maillages généré

Le script `../build_dtc_intermesh_stencils.py` lit les centres de cellules Foundation 13 produits par `foamPostProcess -func writeCellCentres`, puis associe chaque cellule hull à quatre donneurs background par recherche locale et distance inverse. Le résultat est conservé dans `marineInterMeshStencils.json` comme contrat de données intermédiaire.

Le mapping contient **147 200 stencils**, chacun avec quatre donneurs distincts. L’erreur maximale sur la somme des poids est de **2,22 × 10⁻¹⁶**. Ce fichier n’est pas encore consommé par `MarineOversetCellState`, car le runtime actuel attend un dictionnaire OpenFOAM associé à un seul maillage. La prochaine modification doit donc définir le format C++ multi-maillage et produire `zoneID`, `oversetCellStatus` et les stencils dans le maillage de calcul final.

## Smoke test du solveur

Le démarrage `marineFoam -solver incompressibleVoF` atteint la création du mover six-DoF, la sélection du modèle RAS `kOmegaSST`, la création de `fvConstraints` et l’instanciation de `marineOversetConstraint` avec 134064 cellules calculées. Après correction du fichier `points0`, du modèle `momentumTransport`, du patch overset Foundation et de la condition limite de sortie, le cas franchit le premier pas et termine le smoke test sans erreur. Les journaux indiquent toutefois que la contrainte est déclarée mais non appelée pour certains champs, ce qui est attendu puisque `oversetCellStatus` ne contient encore aucune cellule interpolée ou trou dans ce background. Le fichier `fvOptions` OpenCFD contenant `velocityDampingConstraint` a été conservé sous `fvOptions.opencfd-reference` et désactivé dans le squelette Foundation 13.

## Prochain harness multi-région

Le runner devra enregistrer le maillage donneur sous un nom de région, par exemple `background`, avec le chemin `constant/background/polyMesh`, puis charger les champs donneurs dans le même `Time`. La contrainte pourra alors résoudre `mesh.time().lookupObject<fvMesh>(donorRegion)` et appliquer `MarineInterMeshMatrix` à la matrice de la région receveuse. Tant que cette étape n’est pas réalisée, les deux sous-cas restent volontairement indépendants.

## Runner multi-région

`marineFoam` accepte désormais `-donor-region <name>`. Cette option charge le maillage nommé et les champs `U`, `p_rgh`, `alpha.water`, `k`, `omega`, `epsilon` et `nut` dans le même registre `Time` avant la construction du solver. La contrainte peut alors détecter la région et utiliser `MarineInterMeshMatrix`. Le runner et la bibliothèque compilent avec Foundation 13 ; l’exécution de cette voie exige encore une casse unifiée contenant réellement `constant/<name>/polyMesh` et les champs correspondants.

## Validation du couplage effectif

Le harness `openfoam13/marineInterMeshCouplingTest` charge simultanément le maillage hull receveur et le maillage background donneur dans un même objet `Time`. Il construit un champ donneur uniforme, applique les 147200 stencils à la matrice receveuse et vérifie analytiquement la reconstruction de la valeur 2. Le test se termine avec `inter-mesh matrix coupling passed` et le code retour 0. Cette validation démontre le chemin `fvMesh` donneur → stencils → `MarineInterMeshMatrix` → `fvMatrix`, sans encore constituer un calcul DTC physique complet à deux régions.
