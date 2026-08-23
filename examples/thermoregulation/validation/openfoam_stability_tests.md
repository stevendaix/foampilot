# Essais de stabilisation OpenFOAM 13 du cas humain

## Résultat retenu

La configuration qui fait aboutir le calcul humain ouvert et couplé est la suivante : maillage MakeHuman `body-only`, modèle laminaire, équation d’état Boussinesq, gravité active, `externalCoupledTemperature` sur le patch `human`, `deltaT = 0,05 s`, `endTime = 0,20 s` et relaxation du retour JOS-3 `alpha = 0,1`.

Le calcul atteint `0,20 s` avec `CFD_status=0` et `JOS3_status=0`. Quatre échanges portent sur 9 418 faces. Le HTC OpenFOAM reste compris entre 1,351 et 13,44 W m⁻² K⁻¹ dans ce cas body-only. Le pilote JOS-3 retourne une température de peau comprise approximativement entre 33,94 et 34,01 °C au dernier échange.

## Matrice des essais

| Variante | Résultat | Interprétation |
|---|---|---|
| Perfect gas, cavité fermée, peau fixe | Échec vers 0,20 s | Divergence pression/thermodynamique malgré cavité fermée |
| Perfect gas, frontière ouverte, peau fixe | Échec vers 0,15–0,20 s | Modèle compressible trop instable pour la configuration actuelle |
| Perfect gas, frontière ouverte, `deltaT=0,01 s` | Échec vers 0,04–0,05 s | La réduction du pas ne supprime pas l’instabilité |
| Boussinesq, cavité fermée, peau fixe | Aboutit à 0,20 s | Formulation thermique stable |
| Boussinesq, frontière ouverte, peau fixe | Aboutit à 0,20 s | Frontières ouvertes compatibles sur cette durée courte |
| Boussinesq, frontière ouverte, couplage JOS-3 | Aboutit à 0,20 s | Échange bidirectionnel validé |
| Boussinesq, `deltaT=0,01 s`, Euler sans pilote | Non concluant | Arrêt par timeout normal du protocole externe, car aucun processus Python ne répondait |
| Boussinesq body-only raffiné niveau `(3 3)`, couplé | Échec vers 0,15 s | Maillage `Mesh OK`, mais gradients/HTC plus raides et instabilité CFD |

## Diagnostic physique et numérique

Le passage de `perfectGas` à `Boussinesq` est la correction déterminante. Le modèle Boussinesq représente ici la poussée d’Archimède par une densité de référence et une dilatation linéarisée, ce qui évite la boucle fortement non linéaire entre température, densité, pression et vitesse du modèle parfait-gaz. La gravité reste active dans `constant/g` avec `g = (0 -9,81 0)` puisque l’axe vertical MakeHuman est `y`.

L’ouverture `inlet/outlet` utilisée dans le test est : `inlet` à vitesse nulle imposée, `outlet` en `zeroGradient`, et `fixedFluxPressure` pour `p_rgh`. Cette ouverture est suffisante pour faire aboutir le calcul court, mais elle ne constitue pas encore une chambre climatique physiquement complète. Une validation longue devra contrôler le débit net, les profils de pression, le nombre de Rayleigh et la conservation énergétique.

Le raffinement global `(3 3)` n’est pas retenu comme défaut. Il transforme le patch humain en surface fermée dans `checkMesh`, mais augmente le nombre de cellules de 66 379 à 259 196 et le HTC maximal du couplage jusqu’à environ 146 W m⁻² K⁻¹. Pour la suite, il est préférable d’utiliser le maillage body-only niveau `(2 2)` et un raffinement local ciblé des mains, des bras et des zones de faible qualité.

## Modifications permanentes appliquées

Le générateur MakeHuman sélectionne désormais le groupe `body` (`group=0`) au lieu d’exporter tous les groupes `joint-*` et `helper-*`. Le cas d’exemple versionné utilise désormais l’équation d’état Boussinesq dans `constant/physicalProperties`. Le lanceur `Allrun` ne recharge `bashrc` que si l’environnement OpenFOAM n’est pas déjà initialisé.

## Limite du test Euler

Le test Euler a été correctement reconfiguré avec les entrées `rho`, `rhoFinal`, `p_rghFinal` et les solveurs finaux PISO. Son arrêt à `0,01 s` ne correspond pas à un échec CFD : le champ `T` conservait la condition `externalCoupledTemperature` et aucun pilote JOS-3 n’était lancé pour fournir `data.in`. Le test transitoire Euler devra donc être rejoué avec le processus de couplage actif.

## Fichiers de preuve

| Fichier | Contenu |
|---|---|
| `openfoam_stability_tests/boussinesq_body_only_open/fluid_coupled.log` | Couplage Boussinesq ouvert à 0,20 s |
| `openfoam_stability_tests/boussinesq_body_only_open/jos3_driver.log` | Traces des quatre échanges JOS-3 |
| `openfoam_stability_tests/boussinesq_body_only/result.log` | Ligne de base Boussinesq fermée |
| `openfoam_stability_tests/boussinesq_body_only_open/result.log` | Boussinesq ouvert avec peau fixe |
| `examples/thermoregulation/makehuman/openfoam_cube_case/versioned_body_only_Allrun.log` | Regénération body-only du cas d’exemple |

## Essais longue durée et séparation de responsabilité

Un essai OpenFOAM seul avec température humaine fixe a été lancé jusqu’à `5 s`. Il s’arrête à `0,30 s` par erreur flottante, après une dégradation déjà visible à `0,25 s` : la somme locale des erreurs de continuité atteint `5,78 × 10⁻²` et l’erreur globale `−1,07 × 10⁻³`. Le cas ne contient alors aucun calcul JOS-3 ; cette divergence est donc intrinsèque à la résolution CFD actuelle.

Le même cas avec le couplage JOS-3 actif s’arrête également à `0,30 s`. Le pilote JOS-3 termine normalement après six échanges, sans exception Python ni valeur non finie. Les séries de continuité sont quasiment identiques au cas à température fixe :

| Temps | Fixe : somme locale | Couplé : somme locale | Fixe : globale | Couplé : globale |
|---:|---:|---:|---:|---:|
| 0,05 s | 1,04585e-02 | 1,04585e-02 | 5,23e-18 | 1,32e-18 |
| 0,10 s | 1,24952e-02 | 1,24970e-02 | 3,15e-06 | 3,15e-06 |
| 0,15 s | 8,36686e-03 | 8,36646e-03 | −5,74e-05 | −5,74e-05 |
| 0,20 s | 4,49189e-02 | 4,49386e-02 | −2,90e-04 | −2,90e-04 |
| 0,25 s | 5,78012e-02 | 5,77267e-02 | −1,07e-03 | −1,07e-03 |

Cette expérience est le contrôle le plus important : **JOS-3 et le protocole de fichiers ne sont pas la cause dominante de la divergence à long terme**. Le couplage modifie légèrement la température de peau, mais la trajectoire CFD reste la même et l’arrêt survient au même instant. La cause résiduelle est à rechercher dans la stabilité hydrodynamique/pression de la formulation Boussinesq ouverte, avec un déclenchement au voisinage de `0,25–0,30 s`.

Le cas final validé est donc limité à `0,20 s` pour l’instant. Pour dépasser cette durée, la prochaine correction doit porter sur la stabilisation OpenFOAM : davantage de corrections PISO/PIMPLE, une limitation de la vitesse ou du nombre de Courant, une relaxation de pression plus stricte et un raffinement local autour des bras/mains plutôt qu’un raffinement global. Le retour JOS-3 devra ensuite être retesté exactement sur la même configuration.

## Essai de stabilisation longue durée par dissipation

Un cas fixe avec convection d’enthalpie `upwind`, relaxation `p_rgh = 0,1`, relaxation `h = 0,03` et limiteur `270–340 K` a été testé jusqu’à `5 s`. Il progresse jusqu’à `1,40 s`, soit nettement plus loin que la configuration standard, mais finit par diverger à `1,45 s`. L’erreur globale de continuité augmente progressivement de `1,8 × 10⁻³` à `2,0 × 10⁻¹` entre `0,50` et `1,40 s`.

La trace finale montre que l’inversion énergie-température oscille entre environ `240 K` et `360 K`, puis atteint plus de `1000 K` malgré le limiteur. OpenFOAM arrête alors la résolution thermodynamique après 100 itérations. Cela confirme que le mécanisme terminal est une enthalpie devenue non physique, consécutive à une dérive du champ de vitesse/pression ; ce n’est pas une exception du pilote JOS-3.

L’upwind améliore donc la durée de calcul mais ne constitue pas une correction physique suffisante. La prochaine stabilisation sérieuse doit contrôler la vitesse et la pression (PISO/PIMPLE, relaxation, conditions de débit aux ouvertures), puis vérifier le bilan d’énergie avant de relancer JOS-3. Un limiteur de température peut rester un garde-fou de diagnostic, mais ne doit pas être utilisé comme correction finale de la physique.

## Test de couplage contrôlé sans JOS-3

Pour isoler le protocole lui-même, un coupler minimal a été écrit. Il lit `data.out`, attend que le fichier soit complètement écrit, vérifie exactement 9 418 faces, puis renvoie une température constante de 307,75 K dans `data.in` sans appeler JOS-3. Le coupler effectue six échanges correctement et termine sans erreur.

OpenFOAM diverge toutefois au même instant, `0,30 s`, avec la même signature thermodynamique que le cas température fixe. Cette expérience exclut un défaut de calcul physiologique JOS-3 et un défaut de parsing du protocole comme cause immédiate de la divergence longue durée. Elle a aussi révélé et corrigé un défaut réel du coupler de test : la lecture pouvait intervenir avant la fin d’écriture de `data.out`. Le fournisseur de production attendait déjà une taille stable ; le coupler de test a été aligné sur ce comportement.

Un second défaut réel a été corrigé dans le pilote principal : `DistributedSurfaceNetwork.step()` recevait auparavant `dtime=1.0` à chaque échange alors que `controlDict` utilisait `deltaT=0.05`. Le pilote lit désormais `deltaT` dans `controlDict` et transmet `dtime=0.05`. Après correction, JOS-3 avance au rythme CFD correct et retourne une température pratiquement stationnaire autour de 34 °C, mais OpenFOAM diverge toujours à 0,30 s. Le couplage reste donc un point à sécuriser, mais la preuve contrôlée indique que le défaut long terme persiste même lorsque sa sortie est constante.

## Isolation de la flottabilité

Le même cas OpenFOAM découplé avec température de peau fixe a été relancé avec `g = (0 0 0)`. Il converge jusqu’à `5 s`, avec des erreurs de continuité nulles sur la fin du calcul et sans erreur thermodynamique. À l’inverse, avec la gravité active, le cas diverge entre `0,20` et `0,30 s` dans la configuration standard.

Ce résultat isole la boucle déstabilisante : la thermique seule et la condition externe peuvent être intégrées sur une durée longue ; c’est l’interaction entre la poussée Boussinesq, la pression `p_rgh` et les conditions ouvertes qui génère la vitesse non physique puis l’explosion de l’énergie. Le prochain réglage doit donc préserver le cas sans gravité comme référence et réintroduire la gravité avec une rampe de température, un pas contrôlé par Courant et une condition d’ouverture débit/pression correctement équilibrée.

## Correction de la géométrie réellement ouverte

L’audit des patches a révélé une incohérence importante. Le cas précédemment appelé « plafond ouvert » avait en réalité : `inlet` sur la face latérale `x=-0,75`, `outlet` sur la face latérale `x=+0,75`, tandis que `ceiling` restait une paroi. Il ne s’agissait donc pas d’un volume ouvert par le dessus.

Un cas corrigé a été construit avec `inlet` et `outlet` convertis en murs et `ceiling` comme unique patch ouvert. Avec température humaine fixe, Boussinesq et gravité active, il converge jusqu’à `1 s`, puis jusqu’à `5 s` sans erreur fatale. Les erreurs de continuité diminuent jusqu’à environ `1,7 × 10⁻⁴` en somme locale à `5 s`, avec une erreur globale d’environ `1,06 × 10⁻⁴` par pas et une continuité cumulée qui revient progressivement vers zéro.

Cette expérience explique le comportement jugé anormal : le problème venait en grande partie de la définition du scénario ouvert, pas de la complexité intrinsèque du volume. Le cas de référence à retenir est désormais celui où le plafond est réellement ouvert.

## Couplage sur le vrai plafond ouvert

Le couplage bidirectionnel a ensuite été activé sur la configuration corrigée : `ceiling` est l’unique ouverture, `human` utilise `externalCoupledTemperature`, JOS-3 reçoit les champs OpenFOAM face par face et le pilote utilise désormais `dtime = deltaT = 0,05 s`.

Le calcul atteint `5 s` avec `100` échanges, sans erreur CFD ni erreur Python. Le mapping comporte 9 418 faces humaines et les températures retournées restent dans une plage stable d’environ `33,91–34,02 °C` en fin de calcul. Le HTC reste entre `1,351` et `13,44 W m⁻² K⁻¹` dans la trace du pilote. Le plafond ouvert réellement défini est donc stable sans couplage et avec le couplage JOS-3 sur la fenêtre de 5 secondes.
