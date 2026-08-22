# Audit technique de l’implémentation VOF–DPM

## 1. Conclusion exécutive

L’implémentation actuelle est solide comme **prototype d’extraction offline** et comme **pont solver–cloud OpenFOAM 13**, mais elle ne réalise pas encore une conversion VOF-to-DPM temps réel complète. Le convertisseur Python et l’utilitaire C++ identifient des cellules liquides connectées, calculent un volume liquide physique `alpha V`, un centre et une vitesse moyennés par ce volume, puis écrivent des descriptions compatibles avec une injection contrôlée. Les bibliothèques `incompressibleVoFClouds` et `compressibleVoFClouds` instancient et font évoluer un `parcelCloudList` et ajoutent le retour de quantité de mouvement dans l’équation `U`.

La lacune principale est volontairement visible dans l’architecture : **aucun parcel n’est créé dynamiquement à partir d’un fragment VOF dans le `fvModel`, et le champ `alpha` n’est pas décrémenté du volume converti**. Les cas damBreak validés utilisent `manualInjection`. Ils prouvent l’initialisation du cloud, son évolution et le chemin de force, mais pas la conservation globale lors d’une conversion automatique.

## 2. Cartographie des composants

| Composant | Responsabilité | État audité |
|---|---|---|
| `VofToDpmConverter` | Extraction Python de composantes et calcul des propriétés | Fonctionnel sur tableaux et cas ASCII |
| `OpenFoamAsciiReader` | Lecture de `alpha`, `U`, `C`, `V`, `owner`, `neighbour` | Fonctionnel, ASCII seulement |
| `vofToDpm` | Version native offline de l’extraction | Fonctionnelle en série |
| `incompressibleVoFClouds` | Évolution cloud et source momentum dans `incompressibleVoF` | Compilé et validé avec injection manuelle |
| `compressibleVoFClouds` | Évolution cloud et source momentum dans `compressibleVoF` | Compilé et validé avec injection manuelle |
| `foampilot/examples/course_vof_to_dpm.py` | Support pédagogique et rapports de bilan | Ajouté dans ce cours |

## 3. Audit du convertisseur Python

### 3.1 Lecture et validation des données

`OpenFoamAsciiReader._tokens` retire les commentaires et refuse explicitement les fichiers déclarés binaires. Cette décision est correcte pour un prototype pédagogique : elle évite de produire silencieusement des résultats faux à partir d’un format non décodé. En revanche, une campagne de production devra ajouter un lecteur collé aux API OpenFOAM ou une conversion préalable fiable des champs binaires.

`OpenFoamCaseReader.read` lit `alpha`, les centres `C`, les volumes de cellules, puis la connectivité interne en associant les premières entrées de `owner` aux entrées de `neighbour`. Le commentaire et l’intention sont corrects pour le format polyMesh standard. Le contrôle des dimensions, des indices et des volumes positifs constitue une bonne barrière d’intégrité.

### 3.2 Détection par composantes connexes

La méthode `extract` construit un masque `eligible = alpha >= alpha_threshold`, puis utilise un parcours en profondeur. La connectivité est face-à-face, ce qui est la bonne topologie pour éviter de relier artificiellement des cellules seulement par un sommet ou une arête.

Le poids physique est `w_i = alpha_i V_i`. Le volume, le centroïde et la vitesse sont calculés par sommes pondérées. L’absence de renormalisation d’`alpha` est correcte : un seuil de sélection ne doit pas transformer une cellule d’interface à `alpha=0.5` en cellule pleine.

La méthode trie les cellules d’un fragment et les fragments par premier indice. Ce choix améliore la reproductibilité des tests et des rapports. Les filtres `min_cells` et `min_volume` sont explicites, mais ils peuvent éliminer du liquide. Le rapport doit donc toujours être interprété avec le volume rejeté, et une future conversion temps réel devra soit conserver ce liquide en VOF, soit expliquer physiquement son traitement.

### 3.3 Sorties et traçabilité

Les trois sorties sont bien séparées : positions, dictionnaire de propriétés et rapport JSON. Le JSON contient les paramètres, les indices de cellules, les volumes, les centres, les vitesses et les diamètres équivalents. Cette traçabilité est suffisante pour une validation offline et doit être conservée dans la future version temps réel sous forme de journal de conversion.

## 4. Audit de l’utilitaire C++

L’utilitaire `vofToDpm` reprend l’algorithme Python avec `mesh.cellCells()`, `DynamicList`, un parcours des composantes, et les mêmes poids `alpha*V`. Il calcule également la masse `rhoLiquid*volume`, le diamètre équivalent et les volumes sélectionné, converti et rejeté.

Le garde-fou `Pstream::parRun()` arrête explicitement l’exécution parallèle. C’est préférable à une conversion incorrecte de fragments séparés par une frontière MPI. Pour rendre l’utilitaire parallèle, il faudra étiqueter les composantes localement, échanger les labels aux frontières processor, effectuer une fusion distribuée et réduire les propriétés avec des sommes globales ; le simple `gather` des résultats locaux ne suffit pas.

L’option `rhoLiquid` est adaptée à une première version incompressible, mais elle est insuffisante pour une variante compressible générale. Dans ce cas, la masse d’un fragment doit être intégrée comme `sum(alpha_i rho_liquid,i V_i)` ou être construite à partir de la thermodynamique locale, avec cohérence énergétique.

## 5. Audit du `fvModel` incompressible

Le constructeur recherche `incompressibleTwoPhaseVoFMixture`, lit `g`, calcule un champ de viscosité dynamique `mu = rho*nu`, puis construit `parcelCloudList`. `addSupFields()` déclare le champ `U`, et `correct()` utilise un garde sur `timeIndex` afin de ne pas faire évoluer le cloud plusieurs fois pendant un même pas logique.

Le hook vectoriel `addSup` ajoute `clouds_.SU(eqn.psi())` à l’équation de quantité de mouvement. Le test montre que le modèle est effectivement sélectionné et que la quantité de mouvement Lagrangienne évolue. C’est le chemin correct pour le retour mécanique du cloud vers le porteur.

Le hook scalaire lié à `alpha` n’effectue pas de consommation de volume liquide. Cette absence empêche de qualifier le système de conversion conservative : le cloud peut contenir la masse injectée tandis que le VOF conserve encore le liquide correspondant. Le correctif de production devra créer un état de conversion, calculer un delta `alpha`, le borner dans `[0,1]`, et ne valider la création du parcel qu’après succès de la mise à jour eulérienne.

## 6. Audit du `fvModel` compressible

Le modèle compressible recherche `compressibleTwoPhaseVoFMixture`, récupère la densité et la viscosité cinématique du mélange, puis construit `mu_ = rho*nu`. Cette adaptation est pragmatique pour `collidingCloud`, qui demande un champ de viscosité dynamique. Elle permet au smoke test compressible d’atteindre la boucle temporelle et de faire évoluer un parcel.

La variante ne transfère toutefois pas l’énergie, la masse totale ou l’enthalpie du cloud vers les équations thermodynamiques. Elle ne réalise donc qu’un couplage mécanique. Une version compressible physiquement complète devra définir le traitement de la masse, de l’énergie interne/enthalpie, de la pression et de la fermeture thermodynamique après retrait d’une fraction liquide.

## 7. Invariants et tests actuels

Les tests unitaires vérifient la connectivité, les erreurs d’indices, la validité d’`alpha`, les volumes positifs, la pondération du centroïde et de la vitesse, les filtres et l’écriture des sorties. Les tests OpenFOAM vérifient la sélection du modèle et l’activité du cloud.

Il manque les assertions suivantes pour une qualification scientifique :

| Invariant à ajouter | Test recommandé |
|---|---|
| Conservation du volume | `sum(converted alpha*V) == sum(parcel volumes)` |
| Conservation de la masse | `sum(rho_l alpha V)` avant/après, incluant les parcels |
| Conservation de la quantité de mouvement | `P_VOF + P_DPM` avant/après conversion |
| Absence de double conversion | même fragment identifié deux fois dans deux pas |
| Robustesse au maillage | raffinement et changement de non-uniformité des volumes |
| Robustesse MPI | fragment coupé par une frontière processor |
| Compressible | masse, enthalpie et pression après conversion |
| Rejet contrôlé | volume rejeté restitué à VOF ou comptabilisé explicitement |

## 8. Architecture recommandée pour la prochaine étape

Il faut introduire un service `VofFragmentTransition` côté C++ avec quatre phases transactionnelles : détection, décision, création et consommation. La détection construit les composantes candidates et leurs moments. La décision applique les critères de taille, résolution, sphéricité, distance à l’interface et hystérésis temporelle. La création prépare un parcel avec `m`, `x`, `U` et, pour le compressible, les variables thermodynamiques. La consommation modifie `alpha` et les champs associés ; elle ne doit être confirmée que si les bornes et les bilans passent.

Chaque fragment doit recevoir un identifiant stable ou un empreinte géométrique pour empêcher une reconversion immédiate. En parallèle, la composante doit être fusionnée entre partitions avant le calcul de ses propriétés. Le transfert de quantité de mouvement doit utiliser la même discrétisation temporelle que la création du parcel, avec un signe opposé pour la réaction eulérienne.

## 9. Verdict

Le code est **adéquat pour un prototype de recherche et un cours**, car les responsabilités sont séparées, les calculs de volume sont explicites, les entrées invalides sont rejetées et les limites sont documentées. Il est **insuffisant pour une revendication de conversion VOF-to-DPM automatique conservative en production**, car l’étape fondamentale de retrait du liquide VOF et d’insertion dynamique de parcels n’est pas encore intégrée au `fvModel`. Cette distinction doit rester centrale dans toute publication, PR ou présentation du projet.
