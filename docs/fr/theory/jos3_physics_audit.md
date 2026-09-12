# Audit physique de JOS-3 et du couplage FoamPilot

Cette page vérifie la physique effectivement appliquée par le code, et pas seulement la présence de fonctions portant les mêmes noms que les équations de l’article de référence [1]. L’audit couvre les bilans d’énergie, les signes, les unités, la conduction, la perfusion, les pertes environnementales et l’extension de surface CFD.

## 1. Résumé des conclusions

Le modèle JOS-3 original est un modèle thermique concentré à 85 états, organisé sur 17 segments corporels. Il conserve l’énergie interne par résolution d’un système implicite ; les échanges internes par conduction et perfusion sont assemblés comme des transferts entre nœuds.

| Point vérifié | Résultat | Statut |
|---|---|---|
| Nombre d’états | 85 dans le code | Conforme au code de référence |
| Capacités thermiques | Positives, unités finales J/K | Conforme |
| Conduction et perfusion internes | Opérateur assemblé à somme de ligne nulle | Conservation vérifiée |
| Pertes sèches JOS-3 | `h_t (Tsk-To) BSA` | Conforme en mode natif |
| Évaporation | Pertes latentes calculées séparément | Conforme en mode natif |
| Respiration | Perte sensible et latente appliquée au core thorax | Conforme |
| Surface CFD distribuée | Température indépendante par point | Physiquement cohérent pour la partie sèche |
| Rayonnement distribué | Doit utiliser `Tr` distinct de `Ta` | Corrigé dans FoamPilot |
| Flux retourné à OpenFOAM | Doit être en W/m², non en W | Corrigé dans FoamPilot |
| Évaporation dans la couche CFD | Pas encore distribuée par point | Limitation restante importante |

## 2. Conservation des échanges internes JOS-3

La conduction entre deux couches suit :

$$
D_{ij}=G_{ij}(T_i-T_j),
$$

avec `Gij` en W/K. Pour un réseau de conduction, la contribution à l’opérateur possède une diagonale égale à la somme des conductances sortantes et des termes hors diagonale négatifs. Une température uniforme doit donc produire un échange interne nul.

La perfusion est construite de manière analogue avec le facteur de capacité thermique du sang :

$$
K_b=1.067\,\dot V_b,
$$

où `Vb` est en L/h et `Kb` en W/K. Le code assemble les matrices locales et de circulation sanguine, puis ajoute la diagonale de fermeture dans `arrA`.

Le test numérique réalisé donne :

```text
Nombre d’états : 85
Matrice de conduction : (85, 85)
Asymétrie de la matrice interne : 0 W/K
Somme de ligne de l’opérateur assemblé : 3.55e-15 W/K
```

La somme de ligne de la matrice brute des coefficients n’est pas nulle avant l’assemblage de la diagonale ; c’est normal. Après l’assemblage implicite utilisé par JOS-3, elle est nulle à la précision machine. Cela confirme la conservation des échanges internes, indépendamment des capacités individuelles.

## 3. Résolution temporelle

JOS-3 utilise une discrétisation implicite backward difference. La forme générale est :

$$
A T^{n+1}=T^n+\Delta t\,C^{-1}Q+\Delta t\,C^{-1}B T_o,
$$

où `C` est la matrice diagonale des capacités, `Q` la production nette de chaleur et `B` le coefficient de perte sèche.

L’utilisation d’une résolution implicite est adaptée aux grandes disparités de capacités et de conductances du réseau. Elle ne garantit toutefois pas une précision physique indépendante du pas : le pas `dtime` doit être suffisamment petit pour les transitions rapides de ventilation, de sudation et de flux CFD.

## 4. Vérification des signes et des unités

Dans le mode natif, JOS-3 calcule une perte sèche positive lorsque `Tsk > To` :

$$
Q_{dry}=h_t(T_{sk}-T_o)BSA.
$$

Cette puissance est soustraite de l’équation de la peau. Une évaporation positive `Esk` est également soustraite de la peau. Une puissance externe `ex_q > 0` est au contraire ajoutée au nœud physiologique.

Dans le réseau distribué FoamPilot :

$$
Q_{body,i}=G_i(T_{surf,i}-T_{skin,zone(i)}),
$$

$$
Q_{env,i}=h_iA_i(T_{surf,i}-T_{a,i})
+h_{r,i}A_i(T_{surf,i}-T_{r,i}).
$$

Le signe positif de `Qenv` signifie que la surface perd de la chaleur vers l’environnement. Le signe de `Qbody` est positif lorsque la surface reçoit de la chaleur depuis le corps ; le réseau utilise alors `-Qbody` dans son bilan local et transmet `Qbody` au nœud cutané JOS-3 comme puissance externe.

Une correction importante a été appliquée : le réseau stocke `Qenv` en watts pour le bilan nodal, mais OpenFOAM attend un flux surfacique :

$$
q_{OpenFOAM,i}=\frac{Q_{env,i}}{A_i}\quad [W/m^2].
$$

Le provider écrit désormais `Qenv/A` et non `Qenv`. Cette distinction est essentielle dès que les aires des faces ne sont pas toutes identiques.

Les températures OpenFOAM sont converties de kelvins vers degrés Celsius avant leur utilisation dans JOS-3. Les coefficients doivent rester en W/m²/K, les aires en m², les puissances en W et les flux retournés en W/m².

## 5. Convection et rayonnement

Le code JOS-3 original utilise des coefficients locaux de convection et de rayonnement calibrés pour les postures debout, assise et couchée. Dans un couplage CFD, il est préférable d’utiliser `h` local fourni par OpenFOAM plutôt que de recalculer `hc` à partir d’une vitesse moyenne.

La température de l’air `Ta` et la température radiative `Tr` ne sont pas interchangeables :

$$
Q_{conv}=h(T_{surf}-T_a)A,
$$

$$
Q_{rad}=h_r(T_{surf}-T_r)A.
$$

Le réseau distribué accepte maintenant explicitement `radiative_temperature`. Si elle n’est pas fournie, `Tr=Ta` est utilisé comme hypothèse simplificatrice. Cette hypothèse est acceptable uniquement pour un environnement radiativement uniforme ou lorsque le rayonnement est volontairement négligé.

## 6. Capacités et conductances de la surface distribuée

Pour chaque zone JOS-3, la capacité cutanée est répartie proportionnellement aux aires duales :

$$
C_i=C_{sk,zone}\frac{A_i}{\sum_{k\in zone}A_k}.
$$

La conductance d’ancrage est répartie de la même manière :

$$
G_i=G_{sk\rightarrow profond,zone}\frac{A_i}{\sum_{k\in zone}A_k}.
$$

Le test donne une erreur maximale de conservation des capacités de :

```text
2.27e-13 J/K
```

Cette extension conserve donc les propriétés globales de JOS-3 par zone. Elle ajoute cependant une hypothèse spatiale : JOS-3 ne fournit pas de distribution intra-zone, de sorte que la répartition par aire n’est pas une donnée expérimentale native du modèle.

## 7. Limitation physique restante : évaporation distribuée

Le mode `external_surface` retire la perte sèche interne `C+R` de JOS-3 afin d’éviter son double comptage. En revanche, l’évaporation `E` et la sudation restent calculées par JOS-3 au niveau des 17 zones.

La conséquence est importante : la température de surface CFD reçoit actuellement les pertes convectives et radiatives locales dans `Qenv`, mais pas une perte latente locale calculée à partir de l’humidité CFD. L’énergie latente est retirée du compartiment physiologique JOS-3, mais elle n’est pas encore explicitement retirée de chaque état thermique `Tsurf,i`.

Cette situation est acceptable comme première extension sèche, mais elle ne constitue pas encore un couplage complet chaleur–humidité. Pour un couplage physique complet, il faut ajouter par point :

$$
Q_{lat,i}=w_i h_{e,i}(p_{sat}(T_{surf,i})-p_{a,i})A_i,
$$

puis utiliser :

$$
Q_{env,i}=Q_{conv,i}+Q_{rad,i}+Q_{lat,i}.
$$

Il faudra également éviter de compter deux fois la même sudation : soit JOS-3 fournit le taux de mouillabilité par zone et la CFD calcule la puissance latente locale, soit JOS-3 conserve le calcul latent global et la couche distribuée ne fait que redistribuer cette puissance par aire.

## 8. Bilan énergétique recommandé pour le couplage complet

Pour chaque point CFD, le bilan recommandé est :

$$
C_i\frac{dT_{surf,i}}{dt}
=-Q_{body,i}-Q_{conv,i}-Q_{rad,i}-Q_{lat,i}+Q_{solar,i}.
$$

La puissance retournée au fluide doit être :

$$
q_i=\frac{Q_{conv,i}+Q_{rad,i}+Q_{lat,i}-Q_{solar,i}}{A_i},
$$

avec une convention de signe fixée dans la condition limite OpenFOAM. Cette équation doit être documentée avec la convention exacte de `externalWallHeatFluxTemperature`, car certaines conditions limites expriment le flux positif vers le domaine et d’autres vers la paroi.

## 9. Régulation physiologique

Les régulations de vasodilatation, vasoconstriction, AVA, sudation, frisson et thermogenèse non frissonnante sont des lois empiriques de contrôle, et non des lois locales CFD. Elles utilisent des erreurs de température par segment, des pondérations `SKINR`, des facteurs d’âge et des seuils expérimentaux.

Cela implique que même avec un champ CFD très détaillé, les commandes physiologiques restent principalement à l’échelle des 17 zones JOS-3. La température par point ajoutée par FoamPilot influence le modèle via une puissance intégrée par zone, mais ne crée pas encore 34 ou 10 000 récepteurs thermiques physiologiques indépendants.

## 10. Résultats numériques de l’audit

| Test | Résultat |
|---|---:|
| Nombre d’états JOS-3 | 85 |
| Capacité minimale | 15.903 J/K |
| Capacité maximale | 37071 J/K |
| Asymétrie de l’opérateur interne | 0 W/K |
| Somme de ligne après assemblage | 3.55e-15 W/K |
| Erreur de conservation des capacités distribuées | 2.27e-13 J/K |
| Flux environnemental positif pour une surface chaude | Oui |
| Températures de surface indépendantes avec champ non uniforme | Oui |
| Tests distribués et externalCoupled | Réussis |

## 11. Recommandations

La première correction indispensable était la conversion des puissances nodales en flux surfaciques avant l’écriture vers OpenFOAM ; elle est maintenant appliquée. La seconde correction consiste à distinguer `Ta` et `Tr`, désormais supportés séparément.

La prochaine correction physique prioritaire est l’ajout d’un échange latent distribué avec humidité, pression de vapeur, humidité relative locale et stratégie explicite de non-double-comptage de la sudation. Ensuite, il faudra réaliser un bilan global à chaque pas : énergie stockée dans JOS-3 et la surface, chaleur métabolique, convection, rayonnement, évaporation, respiration et flux OpenFOAM.

## Références

[1]: https://doi.org/10.1016/j.enbuild.2020.110575 "Takahashi et al., Thermoregulation model JOS-3 with new open source code"
[2]: https://github.com/TanabeLab/JOS-3 "Dépôt officiel JOS-3"
[3]: https://doc.openfoam.com/2306/tools/processing/boundary-conditions/rtm/derived/thermal/externalWallHeatFluxTemperature/ "OpenFOAM externalWallHeatFluxTemperature"
[4]: https://github.com/stevendaix/foampilot "Dépôt FoamPilot"
