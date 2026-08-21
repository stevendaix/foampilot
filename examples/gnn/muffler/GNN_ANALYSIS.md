# Étude approfondie du GNN FoamPilot — cas muffler

## Conclusion exécutive

Le code contient une architecture GNN fonctionnelle au sens logiciel, mais il ne constitue pas encore un surrogate CFD physiquement fiable. Le principal problème ne vient pas du nombre de couches ou de la taille cachée. Il vient de la représentation du problème : le graphe est reconstruit par k-NN sur les centroïdes, alors que le solveur OpenFOAM fournit une connectivité exacte entre cellules et faces ; plusieurs informations nécessaires aux pertes physiques ne sont ensuite pas transmises au modèle ou à la fonction de coût.

La priorité doit donc être de corriger la **sémantique du graphe et des unités** avant d’optimiser les hyperparamètres. Une architecture plus complexe ne compensera pas une connectivité approximative, des arêtes orientées dans un seul sens et une évaluation réalisée avec seulement quelques cas.

## 1. Architecture actuelle

Le pipeline suit la chaîne suivante : lecture des résultats VTK, conversion des cellules en nœuds, extraction de features locales, construction d’un graphe k-NN, prédiction de `p`, `Ux`, `Uy`, `Uz`, puis calcul d’une loss supervisée complétée par des termes physiques.

| Élément | Implémentation actuelle | Évaluation |
|---|---|---|
| Nœuds | Centroïdes des cellules VTK | Correct comme approximation initiale |
| Arêtes | k-NN géométrique, `k=6` | Insuffisant pour représenter les faces CFD |
| Features nœuds | Position, distances aux frontières, type de frontière, volume, paramètres globaux selon la version | Utile mais incomplet |
| Features arêtes | Longueur, direction, distance k-NN | Incomplet : manque aire de face, normale et distance de centres exacte |
| Encodeur | `Linear → BatchNorm → ReLU → Dropout` | Standard |
| Couches GNN | 6 couches `UniversalGraphConv` | Acceptable, mais la propagation est problématique |
| Sortie | `p`, `Ux`, `Uy`, `Uz` pour le muffler | Cohérente avec les champs disponibles |
| Incertitude | Tête produisant des paramètres, sans vraisemblance ni calibration effective | Déclarative, non validée |
| Batching | Un seul graphe par batch | Très inefficace et fortement limitant |

## 2. Problème critique : le graphe n’est pas le graphe CFD

Dans `_extract_edge_features`, les arêtes sont produites par un `cKDTree` sur les centroïdes. Cette construction ne garantit pas qu’une arête corresponde à une face commune entre deux cellules. Deux cellules peuvent être proches sans être voisines dans le maillage, tandis que des cellules partageant une face peuvent être exclues du top-k.

Plus grave, l’ensemble d’arêtes est dédoublonné avec `(min(i,j), max(i,j))`. Cela crée une seule arête orientée du plus petit indice vers le plus grand. Dans le fallback sans `torch_scatter`, l’agrégation utilise `src → dst` ; l’information ne circule donc pas symétriquement. Le modèle ne réalise pas une convolution sur un graphe non orienté, mais une propagation dépendante de l’ordre arbitraire des indices de cellules.

La correction prioritaire est de lire la connectivité exacte du maillage OpenFOAM depuis `polyMesh/faces`, `owner` et `neighbour`, puis de créer deux arêtes par face interne : `owner → neighbour` et `neighbour → owner`. Chaque arête doit porter au minimum le vecteur centre-à-centre, sa longueur, la normale orientée et l’aire de face.

## 3. Les informations physiques sont perdues entre l’extracteur et la loss

L’extracteur retourne les features, les arêtes, les volumes et les cibles, mais ne retourne pas explicitement `node_positions` ni `boundary_type`. Pourtant, `_momentum_conservation_loss` cherche `graph.get("node_positions")`, et `_boundary_condition_loss` cherche `graph.get("boundary_type")`. Ces clés ne sont pas intégrées de manière fiable au graphe retourné.

Par conséquent, le terme de conditions limites est généralement nul, car `boundary_type` est absent. Les termes de conservation utilisent alors des approximations ou ne s’activent pas. Pour le cas incompressible, `rho` n’est pas une sortie du modèle ; le terme de conservation de masse est donc également inactif. Le label « physics-informed » est ainsi beaucoup plus ambitieux que l’implémentation réellement exécutée.

| Terme | Condition d’activation | État probable sur muffler |
|---|---|---|
| Données | `p` et `U` présents | Actif |
| Masse | `rho`, `U`, connectivité et volumes | Inactif sans `rho` |
| Quantité de mouvement | `p`, `U`, connectivité | Approximation très faible |
| Énergie | Fonction renvoyant zéro | Inactif |
| Conditions limites | `boundary_type` dans le batch | Généralement inactif |
| Turbulence | `k` et `omega` prédits | Inactif avec sorties `p,Ux,Uy,Uz` |

## 4. La loss supervisée est mal adaptée aux champs incompressibles

La loss de données utilise une erreur relative point par point :

```text
(pred - target) / (abs(target) + eps)
```

Cette forme est instable lorsque la cible est proche de zéro, ce qui est fréquent pour les composantes transverses de la vitesse et pour les fluctuations de pression. Elle donne aussi un poids excessif aux petites valeurs et peut privilégier la réduction d’erreur relative locale plutôt que la structure spatiale du champ.

Une formulation plus robuste doit normaliser chaque variable avec des statistiques calculées uniquement sur le train, puis appliquer une MSE ou Huber loss dans l’espace normalisé. Les métriques finales doivent être reconverties dans les unités OpenFOAM. Pour `p`, il faut conserver explicitement la convention OpenFOAM incompressible : `p` est une pression cinématique en `[m²/s²]`, pas une pression absolue en pascals.

## 5. L’architecture d’attention est coûteuse et mal justifiée

Lorsque `use_attention=true`, `nn.MultiheadAttention` reçoit une séquence contenant tous les nœuds et un masque dérivé des arêtes. La mémoire et le coût sont quadratiques en nombre de cellules. Pour des graphes de plusieurs milliers de cellules, cette approche devient rapidement prohibitive.

Lorsque `use_attention=false`, la couche utilise une agrégation locale, mais les features d’arêtes ne modulent pas réellement le message de manière physique : elles sont passées à la couche puis essentiellement ignorées par l’agrégation. Une architecture de type message passing avec MLP d’arête est préférable :

```text
m_ij = MLP([h_i, h_j, e_ij])
h_i' = h_i + MLP_agg(sum_j m_ij)
```

Cette formulation exploite effectivement les normales, aires et distances de faces et évite l’attention globale quadratique.

## 6. Le protocole d’évaluation est trop faible

Le split actuel est effectué par cas, ce qui est une bonne direction, mais le nombre de cas de test reste trop petit pour conclure. Avec deux cas de test, un seul cas atypique peut modifier fortement le R². En outre, la validation interne choisit aléatoirement un sous-ensemble à chaque époque, ce qui rend la courbe de validation bruitée et non reproductible.

Le protocole recommandé est le suivant : 60 à 80 % des cas pour l’entraînement, 10 à 20 % pour la validation fixe et 20 % pour le test final, avec trois répétitions de seed ou une validation croisée par cas. Les cas de test doivent être choisis dans des régions de l’espace paramétrique non triviales, et les métriques doivent être rapportées par variable, par patch et par cas.

## 7. Diagnostic des résultats observés

Les derniers résultats disponibles sur seize cas et deux cas de test sont :

| Variable | MAE | RMSE | R² |
|---|---:|---:|---:|
| `p` cinématique | 20,39 | 29,03 | -0,657 |
| `U` | 1,35 | 2,43 | -0,232 |

Les R² négatifs indiquent que le modèle testé est moins bon qu’un prédicteur constant égal à la moyenne du jeu de test. Cela ne signifie pas que le réseau ne peut pas apprendre ; cela indique que le problème d’apprentissage est dominé par la représentation et le protocole, et non par un manque de profondeur.

Le premier diagnostic à exécuter après chaque modification est un test de surapprentissage sur deux ou trois cas. Si le modèle ne peut pas obtenir une erreur quasi nulle sur ces cas, il existe un bug dans les unités, l’alignement nœud-cible, la connectivité ou la loss. Si le surapprentissage fonctionne mais que le test reste mauvais, le problème est alors le manque de couverture paramétrique ou la mauvaise extrapolation.

## 8. Plan d’amélioration prioritaire

| Priorité | Action | Critère de réussite |
|---:|---|---|
| 1 | Remplacer le k-NN par la connectivité exacte `owner/neighbour` | Deux arêtes orientées par face interne, sans dépendance à l’ordre des cellules |
| 2 | Retourner explicitement positions, volumes, boundary masks, normales et aires | Tous les termes de loss reçoivent leurs entrées réelles |
| 3 | Ajouter un test d’overfit sur 2–3 cas | RMSE train très faible sur `p` et `U` |
| 4 | Normaliser séparément `p` et `U` sur le train | Loss stable, métriques physiques correctement reconstruites |
| 5 | Remplacer la loss physique approximée par une loss de flux face-based | Conservation basée sur les faces OpenFOAM réelles |
| 6 | Utiliser un message passing local avec features d’arêtes | Coût linéaire en nombre d’arêtes, meilleure invariance géométrique |
| 7 | Passer à au moins 50–200 simulations | Test indépendant avec plusieurs dizaines de cas |
| 8 | Ajouter une baseline constante et une interpolation paramétrique | Le GNN doit battre les baselines sur chaque variable |

## Corrections actuellement implémentées

Une première réparation a été appliquée au code : la construction des arêtes privilégie désormais les voisins de cellules VTK partageant une face, crée systématiquement les deux orientations de chaque arête et ne conserve le k-NN qu’en fallback. Le graphe retourne également `node_positions` et `boundary_type`, et le batch transmet ces informations aux pertes physiques. Le sous-échantillonnage conserve ces mêmes attributs.

La convolution locale utilise maintenant les features d’arêtes dans un petit MLP de message passing. Les vecteurs de longueur et de direction ne sont plus ignorés. La loss supervisée a été rendue plus stable avec une erreur Huber normalisée par variable, au lieu d’une erreur relative singulière lorsque les champs traversent zéro.

Deux tests automatisés passent : `test_exact_connectivity.py` vérifie que deux cellules VTK adjacentes donnent exactement les arêtes `(0,1)` et `(1,0)`, et `test_repaired_gnn.py` vérifie le forward, la rétropropagation et les sorties `p`/`U` du modèle.

Ces tests sont des tests de structure et de différentiabilité. Une nouvelle validation CFD complète devra être relancée dans un environnement contenant OpenFOAM 13 et les résultats VTK ; les anciens artefacts CFD ne sont pas présents dans l’environnement restauré actuel.

## Verdict

Le GNN est un prototype intéressant, mais son nom « universel » et « physics-informed » doit être interprété avec prudence. La partie la plus solide est le câblage général : extraction VTK, features de cellules, encodeur/décodeur et sauvegarde des expériences. Les parties les plus faibles sont la connectivité, les pertes physiques et l’évaluation.

La meilleure prochaine étape n’est pas d’ajouter des couches. Il faut construire un graphe CFD exact à partir de `owner/neighbour`, rendre les features physiques explicitement accessibles à la loss, normaliser les champs par variable et démontrer un overfit contrôlé avant d’augmenter le dataset. Une fois ces tests réussis, une architecture locale de message passing pourra produire un surrogate réellement exploitable.

## Références

[1]: https://github.com/stevendaix/foampilot/tree/main/examples%2Fgnn "FoamPilot — exemple GNN fourni"

[2]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM 13 — installation Ubuntu"
