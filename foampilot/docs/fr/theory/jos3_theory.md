# Théorie et audit du modèle JOS-3

Cette page présente une réécriture structurée de la théorie du modèle de thermorégulation **JOS-3**, puis vérifie sa correspondance avec le code Python original et la copie embarquée dans FoamPilot. La référence scientifique principale est l’article de Takahashi et al. [1]. La source logicielle de référence est le dépôt public JOS-3 [2].

## 1. Positionnement du modèle

JOS-3 est un modèle thermophysiologique multi-nœuds dérivé de la famille Stolwijk–65MN–JOS. Il calcule les températures corporelles, les flux sanguins, la sudation, la thermogenèse frissonnante et non frissonnante ainsi que les pertes respiratoires dans des environnements uniformes, non uniformes et transitoires [1] [2].

L’article comporte une incohérence de comptage qu’il faut conserver explicitement dans toute documentation : son résumé mentionne **83 nœuds**, tandis que la section de construction et le code Python utilisent **85 états**. La composition effectivement utilisée par le code est un nœud de sang central, 17 artères, 17 veines, 12 veines superficielles, 17 cœurs, 2 muscles, 2 graisses et 17 peaux, soit 85 états.

| Composant | Nombre dans le code | Organisation |
|---|---:|---|
| Sang central | 1 | Nœud global |
| Artères | 17 | Un par segment |
| Veines | 17 | Un par segment |
| Veines superficielles | 12 | Membres uniquement |
| Cœur/core | 17 | Un par segment |
| Muscle | 2 | Tête et pelvis |
| Graisse | 2 | Tête et pelvis |
| Peau | 17 | Un par segment |
| **Total** | **85** | — |

Les 17 segments sont `Head`, `Neck`, `Chest`, `Back`, `Pelvis`, `LShoulder`, `LArm`, `LHand`, `RShoulder`, `RArm`, `RHand`, `LThigh`, `LLeg`, `LFoot`, `RThigh`, `RLeg` et `RFoot`.

## 2. Réseau thermique

Pour chaque état thermique `j` d’un segment `i`, le bilan général s’écrit :

$$
Cap_{j,i}\frac{dT_{j,i}}{dt}=Q_{j,i}+B_{j,i}+D_{j,i}-L_{j,i}+S_{j,i},
$$

où `Cap` est la capacité thermique [J/K], `T` la température [°C], `Q` la production de chaleur [W], `B` l’échange par le sang [W], `D` la conduction entre couches [W], `L` les pertes environnementales [W] et `S` le gain radiatif solaire court [W].

Le sang central échange de la chaleur uniquement par perfusion :

$$
Cap_{cb}\frac{dT_{cb}}{dt}=B_{cb}.
$$

Pour une artère et une veine du segment `i` :

$$
Cap_{ar,i}\frac{dT_{ar,i}}{dt}=B_{ar,i}-D_{ar-cr,i}-D_{ar-ve,i},
$$

$$
Cap_{ve,i}\frac{dT_{ve,i}}{dt}=B_{ve,i}-D_{ve-cr,i}+D_{ar-ve,i}.
$$

Pour les membres, la veine superficielle vérifie :

$$
Cap_{sve,i}\frac{dT_{sve,i}}{dt}=B_{sve,i}-D_{sve-sk,i}.
$$

Le compartiment core dépend de la topologie du segment :

$$
Cap_{cr,i}\frac{dT_{cr,i}}{dt}
=Q_{cr,i}+B_{cr,i}-D_{cr-ms,i}
$$

pour la tête et le pelvis ;

$$
Cap_{cr,i}\frac{dT_{cr,i}}{dt}
=Q_{cr,i}+B_{cr,i}-D_{cr-sk,i}-RES
$$

pour le thorax ; et

$$
Cap_{cr,i}\frac{dT_{cr,i}}{dt}
=Q_{cr,i}+B_{cr,i}+D_{ar-cr,i}+D_{ve-cr,i}-D_{cr-sk,i}
$$

pour les autres segments concernés.

Pour la tête et le pelvis, les couches muscle et graisse sont résolues par :

$$
Cap_{ms,i}\frac{dT_{ms,i}}{dt}=Q_{ms,i}+B_{ms,i}+D_{cr-ms,i}-D_{ms-fat,i},
$$

$$
Cap_{fat,i}\frac{dT_{fat,i}}{dt}=Q_{fat,i}+B_{fat,i}+D_{ms-fat,i}-D_{fat-sk,i}.
$$

Enfin, la peau reçoit la chaleur des tissus profonds, perd de la chaleur par convection, rayonnement et évaporation, et peut recevoir le rayonnement solaire court :

$$
Cap_{sk,i}\frac{dT_{sk,i}}{dt}
=Q_{sk,i}+B_{sk,i}+D_{deep-sk,i}
-(C_i+R_i)-E_i+SW_{sk,i}.
$$

Ces équations correspondent aux équations (1)–(8) de l’article [1] et à la construction de la matrice `arrA` dans `jos3.py` et `construction.py`.

## 3. Conditions aux limites thermiques

La perte sèche de la peau est calculée avec la température opérative :

$$
C_i+R_i=h_{t,i}(T_{sk,i}-T_{o,i})BSA_i,
$$

$$
\frac{1}{h_{t,i}}=0.155Icl_i+\frac{1}{f_{cl,i}(h_{c,i}+h_{r,i})}.
$$

Le facteur de surface du vêtement est :

$$
 f_{cl}=\begin{cases}
1+0.2Icl,&Icl<0.5,\\
1.05+0.1Icl,&Icl\geq0.5.
\end{cases}
$$

La température opérative est :

$$
T_o=\frac{h_cT_a+h_rT_r}{h_c+h_r}.
$$

La perte latente est :

$$
E_i=w_i h_{e,i}(P_{sk,s,i}-P_{a,i})BSA_i,
$$

avec :

$$
\frac{1}{h_{e,i}}=\frac{1}{LR\,i_{cl,i}}+\frac{0.155Icl_i}{f_{cl,i}LRh_{c,i}}.
$$

Le code utilise `LR=16.5 K/kPa`, l’équation d’Antoine pour la pression de vapeur saturante de la peau et de l’air, puis limite la mouillabilité à `w<=1`. Lorsque la capacité évaporative est nulle ou négative, la version durcie conserve une valeur de base et évite les divisions non définies.

La respiration est appliquée au core thoracique :

$$
RES_{sh}=0.0014\,Met\,(34-T_a),
$$

$$
RES_{lh}=0.0173\,Met\,(5.87-p_a),
$$

avec une perte totale `RES=RESsh+RESlh`.

## 4. Surface corporelle, capacités et conductances

La surface corporelle totale par défaut est celle de DuBois :

$$
BSA_{all}=0.2025\,Height^{0.725}Weight^{0.425}.
$$

Le code permet aussi Fujimoto, Takahira et Kurazumi. La surface locale est :

$$
BSA_i=BSA_{st,i}\frac{BSA_{all}}{1.87}.
$$

Les capacités des couches sont distribuées à partir des valeurs du corps standard. Les couches et tissus sont mises à l’échelle par le rapport de masse ; les pools sanguins sont mis à l’échelle par le rapport du débit sanguin basal. Le code convertit ensuite les capacités de Wh/K en J/K par multiplication par 3600.

Les conductances sont données en W/K et le flux de conduction entre deux nœuds est :

$$
D_{j-j',i}=Cdt_{j-j',i}(T_{j,i}-T_{j',i}).
$$

Pour la tête et le cou, la correction géométrique est proportionnelle à `Weightra/BSAra`. Pour les autres segments, elle est proportionnelle à `BSAra²/Weightra`, conformément à l’équation (23) de l’article [1].

## 5. Signaux de thermorégulation

L’erreur locale est :

$$
Err_{j,i}=T_{j,i}-T_{setpt,j,i}.
$$

Les composantes de signal sont :

$$
Wrm_i=\max(Err_{sk,i},0),
$$

$$
Cld_i=\max(-Err_{sk,i},0).
$$

Les signaux intégrés sont pondérés par `SKINR` :

$$
WRMS=\sum_i SKINR_iWrm_i,
$$

$$
CLDS=\sum_i SKINR_iCld_i.
$$

Ces signaux pilotent la vasodilatation, la vasoconstriction, la sudation, le frisson et la thermogenèse non frissonnante.

## 6. Production de chaleur

La production par compartiment est :

$$
Q_{j,i}=Mbase_{j,i}+Mwork_{j,i}+Mshiv_{j,i}+Mnst_{j,i}.
$$

La version Harris–Benedict du métabolisme basal est :

$$
Mbase_{all}=\begin{cases}
(88.362+500.3H+13.397W-5.677Age)0.048,&homme,\\
(447.593+479.9H+9.247W-4.330Age)0.048,&femme.
\end{cases}
$$

Le travail externe utilise :

$$
Mwork=(PAR-1)Mbase_{all}
$$

distribué par les coefficients locaux. Le code propose également l’équation japonaise/Ganpule.

Le frisson dépend du signal de froid central et de l’âge :

$$
Sig_{shiv}=24.36\,CLDS\,(-Err_{cr,cb})
$$

avec une distribution locale `shivf`, une limitation optionnelle de variation temporelle et un seuil de température cutanée.

La thermogenèse non frissonnante est pilotée par la thermogenèse du tissu adipeux brun :

$$
BAT=10^{-0.10502BMI+2.7708}
$$

puis corrigée par l’âge et l’acclimatation au froid. Le signal est limité par :

$$
Sig_{nst}=\min(2.8CLDS,1.80BAT+2.43+5.62).
$$

## 7. Résolution numérique

Le code utilise une discrétisation implicite de type backward difference. Après assemblage des capacités, conductances, échanges sanguins et pertes environnementales, la résolution prend la forme :

$$
A\,T^{n+1}=b,
$$

avec un pas `dtime`. La matrice est construite à partir de `arrA`, le terme environnemental de `arrB` et le vecteur de puissance `arrQ`. Les puissances externes `ex_q` sont ajoutées au vecteur de chaleur avant la résolution.

Le pas par défaut est de 60 s. La stabilité numérique est améliorée par la formulation implicite, mais la précision temporelle reste dépendante de `dtime`, notamment pour le frisson, la sudation et le couplage OpenFOAM.

## 8. Audit équation–code

| Domaine | Article | Code JOS-3 original | Copie FoamPilot | Écart et impact |
|---|---|---|---|---|
| Bilan thermique | Éq. (1)–(8) | Implémenté dans `jos3.py` et `matrix.py` | Reproduit | Aucun écart scientifique identifié |
| Pertes sèches | Éq. (9)–(10) | `heat_resistances`, `dry_r`, `_run` | Reproduit en mode `native` | Le mode `external_surface` désactive volontairement ce terme interne |
| Évaporation | Éq. (11)–(12) | `wet_r`, `evaporation` | Reproduit | Aucun écart identifié |
| Respiration | Éq. (13) | `resp_heatloss` | Reproduit | Le terme est appliqué au thorax comme dans l’article |
| Rayonnement solaire | Éq. (14)–(16) | Injection externe générique via `ex_q` | Partiel | Aucun calcul dédié `SW_dir`/`SW_dif` n’est assemblé dans `_run`; documenter ou implémenter explicitement |
| BSA | Éq. (17)–(20) | `construction.py` | Reproduit | FoamPilot doit conserver la même équation choisie |
| Capacités | Éq. (21) | `capacity` | Reproduit | Attention aux unités Wh/K puis J/K |
| Conduction | Éq. (22)–(23) | `conductance` et matrice | Reproduit | Les conductances restent segmentaires |
| Signaux | Éq. (24)–(28) | `error_signals` | Reproduit | Vecteurs locaux de longueur 17 |
| Métabolisme | Éq. (29)–(32) | `basal_met`, `local_mbase`, `local_mwork` | Reproduit | Choix BMR à documenter dans chaque cas |
| Frisson | Éq. (33)–(35) | `shivering` | Reproduit | Options de seuil et limitation temporelle à expliciter |
| NST/BAT | Éq. (36)–(39) | `nonshivering` | Reproduit | Le code contient une branche d’âge suspecte `elif age < 50` répétée, à vérifier/corriger séparément |
| Vieillissement | Éq. (44), (45), (50) | Facteurs dans thermorégulation et `_run` | Reproduit | Validation nécessaire par cas d’âge |
| Échange CFD distribué | Absent de l’article | Extension FoamPilot | Nouveau | Ne fait pas partie de la validation JOS-3 original |

## 9. Écarts et points à vérifier

### 9.1 Incohérence 83 contre 85 nœuds

L’abstract de l’article indique 83 nœuds, alors que la section de construction et le code définissent 85 états [1]. Pour FoamPilot, la valeur de référence doit être **85**, car elle correspond à `NUM_NODES`, aux index de `matrix.py` et à la résolution numérique réellement exécutée.

### 9.2 AVA et commentaire du dépôt

Le dépôt JOS-3 signale une correction de la fonction AVA : les signaux main/pied utilisent les erreurs cutanées et centrales dans un ordre corrigé. Le code actuel contient cette formulation corrigée. Cette correction doit être conservée et citée comme différence historique par rapport aux versions antérieures.

### 9.3 Thermogenèse non frissonnante

La fonction `nonshivering` contient une condition d’âge répétée pour `age < 50`, ce qui rend une branche intermédiaire inatteignable. Il s’agit d’un point de code à auditer avant toute correction scientifique, car la branche attendue semble probablement concerner une autre tranche d’âge.

### 9.4 `Mnshiv` et débit sanguin

La documentation du dépôt précise que `Mnshiv` dans une équation de l’article n’est pas utilisé directement dans le code pour augmenter le débit sanguin ; le code augmente les débits du muscle ou du core avec `Mwork + Mshiv`. Cette différence est documentée par les auteurs et doit être distinguée d’une erreur de transcription [2].

### 9.5 Extension FoamPilot

La copie FoamPilot ajoute `set_environment_mode("external_surface")`, l’injection de `ex_q` et le réseau de surface distribué. Ces extensions changent volontairement le chemin des pertes sèches : JOS-3 conserve la physiologie interne et FoamPilot calcule l’échange CFD local. Une comparaison avec JOS-3 original n’est donc exacte que lorsque le mode `native` ou un flux externe nul est utilisé.

## 10. Procédure de vérification recommandée

L’audit numérique doit être organisé par niveaux :

| Niveau | Test | Critère |
|---|---|---|
| Structure | `NUM_NODES`, index et noms | 85 états, 17 segments et index identiques |
| Propriétés | BSA, capacités, conductances | Écart relatif inférieur à `1e-12` sur les mêmes paramètres |
| Thermorégulation | fonctions `hc`, `hr`, `evaporation`, `shivering`, `nonshivering` | Comparaison terme par terme |
| Résolution | températures après un pas | Écart inférieur à la tolérance numérique définie |
| Transitoire | exemple officiel `example_v2.py` | séries `Tsk`, `Tcr`, `TskMean` identiques en mode natif |
| Extension CFD | réseau distribué | conservation des capacités, conductances et puissances par zone |

La commande de comparaison existante est :

```bash
python3 examples/thermoregulation/openfoam_jos3_coupling/compare_official_example.py
```

Elle valide le JOS-3 original, la copie FoamPilot et le chemin couplé à flux nul. Elle ne valide pas encore les résultats expérimentaux de l’article ; ces derniers nécessitent les séries expérimentales et les conditions précises des tableaux et figures de validation.

## Références

[1]: https://doi.org/10.1016/j.enbuild.2020.110575 "Y. Takahashi et al., Thermoregulation model JOS-3 with new open source code, Energy and Buildings, 2021"
[2]: https://github.com/TanabeLab/JOS-3 "Dépôt officiel TanabeLab/JOS-3"
[3]: https://github.com/stevendaix/foampilot "Dépôt FoamPilot"
