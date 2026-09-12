# Théorie appliquée : écoulement biomédical, vent extérieur et thermorégulation humaine

Ce chapitre explique **pourquoi** un modèle est choisi, **quelle loi** il représente, **quelles données** il nécessite, et **quand il devient peu fiable**. Il est volontairement plus détaillé qu'une recette de solveur. Un cas CFD n'est pas défini uniquement par le nom de l'exécutable ; il est défini par la géométrie, les lois de conservation, les relations constitutives, les données de frontière, la fermeture turbulente, la discrétisation numérique, et une stratégie de validation.

## 1. Principe de sélection du modèle

Un modèle utile n'est pas le modèle le plus complexe disponible. C'est le modèle le moins complexe qui résout la grandeur d'intérêt dans les conditions de l'expérience ou de l'application. Le choix doit être justifié par :

| Question | Conséquence |
| --- | --- |
| Is the flow compressible? | Select the density and pressure formulation. |
| Is it steady or transient? | Select a steady RANS, transient RANS, LES, or time-dependent laminar model. |
| Is heat coupled to momentum? | Use passive scalar transport, buoyancy, or a full energy/thermophysical model. |
| Are wall gradients important? | Choose wall resolution, wall functions, and mesh targets consistently. |
| Does viscosity depend on shear rate? | Use Newtonian or non-Newtonian rheology. |
| Are the boundaries measured or idealised? | Use data-driven profiles, tables, or analytical functions and quantify uncertainty. |
| Is the geometry patient-specific or geospatial? | Preserve coordinate systems, units, topology, and provenance. |

Un modèle doit toujours indiquer son **domaine de validité**. En particulier, une exécution tutorielle réussie ne valide pas les hypothèses physiques pour une prédiction biomédicale ou environnementale.

# 2. CFD biomédicale

## 2.1 Que modélise-t-on ?

La CFD biomédicale peut concerner des problèmes très différents : écoulement sanguin dans les artères, flux respiratoire, échange de chaleur autour d'un corps, écoulement à travers des dispositifs médicaux, ou transport dans un tissu poreux. Les lois et les données de frontière diffèrent entre ces cas. Cette section se concentre sur l'écoulement vasculaire et sur les interfaces entre géométrie patient-spécifique, modèles hémodynamiques, et utilitaires FoamPilot.

Pour un domaine fluide fixé, les équations de base sont la conservation de la masse et la conservation de la quantité de mouvement :

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0,
$$

$$
\rho\left(\frac{\partial\mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)
=-\nabla p+\nabla\cdot\boldsymbol{\tau}+\mathbf{f},
$$

où $\boldsymbol{\tau}$ est le tenseur des contraintes visqueuses et $\mathbf{f}$ inclut les forces volumiques ou les sources modélisées.

Pour un fluide incompressible Newtonien :

$$
\boldsymbol{\tau}=2\mu\mathbf{D},
\qquad
\mathbf{D}=\frac12\left(\nabla\mathbf{u}+\nabla\mathbf{u}^{T}\right),
$$

avec une viscosité dynamique constante $\mu$. Pour le sang, il s'agit d'une approximation dont l'adéquation dépend de la taille du vaisseau, du taux de cisaillement, de l'hématocrite et de la grandeur de sortie étudiée.

## 2.2 Sang newtonien et non newtonien

Le sang est souvent traité comme Newtonien dans les grosses artères et les régions à cisaillement élevé car la viscosité apparente tend vers une valeur presque constante. Cette simplification réduit le coût numérique et facilite la convergence. Elle peut être défendable quand l'objectif est une chute de pression globale et que le taux de cisaillement est élevé sur la plupart de la région.

Le sang présente aussi un comportement rhéologique d'amincissement par cisaillement : la viscosité apparente augmente aux faibles taux de cisaillement et diminue lorsque le cisaillement augmente. Les modèles non newtoniens deviennent plus importants dans les zones de recirculation, les anévrismes, près de la stagnation, dans les vaisseaux distaux, et lorsque la contrainte de cisaillement pariétale (WSS) ou le temps de résidence est la grandeur d'intérêt principale. Une étude comparative sur des modèles de sténose intracrânienne a montré que les hypothèses Newtonienne et non-Newtonienne peuvent avoir un faible effet sur le rapport de pression tout en produisant des différences plus visibles dans les régions de faible WSS, en particulier pendant la diastole [1].

### Loi newtonienne

$$
\mu=\mu_0.
$$

C'est le modèle le plus simple. La viscosité doit être indiquée avec ses unités et les hypothèses de température.

### Loi de Carreau–Yasuda

Une forme courante d'amincissement par cisaillement est :

$$
\mu(\dot\gamma)=\mu_\infty+(\mu_0-\mu_\infty)
\left[1+(\lambda\dot\gamma)^a\right]^{(n-1)/a},
$$

où $\dot\gamma$ est la magnitude du taux de cisaillement, $\mu_0$ et $\mu_\infty$ sont les viscosités limites, $\lambda$ est une échelle de temps, $a$ contrôle la transition, et $n<1$ produit l'amincissement par cisaillement.

### Loi de Casson

Le modèle de Casson est une autre rhéologie empirique utilisée pour le sang :

$$
\sqrt{\tau}=\sqrt{\tau_y}+\sqrt{\mu_c\dot\gamma},
$$

où $\tau_y$ est un paramètre de type contrainte de seuil et $\mu_c$ contrôle le comportement à haut cisaillement. La régularisation exacte et l'implémentation sont importantes aux faibles taux de cisaillement.

### Comment choisir

| Quantity of interest | First model to test | Additional sensitivity study |
| --- | --- | --- |
| Bulk flow rate or rough pressure ratio | Newtonian | Carreau–Yasuda or Casson if low-shear regions matter. |
| Wall shear stress | Newtonian and non-Newtonian comparison | Report low-WSS and oscillatory-WSS sensitivity. |
| Residence time or thrombosis-related indicator | Non-Newtonian candidate | Check rheological parameters and near-wall resolution. |
| Large artery with high shear | Newtonian may be sufficient | Verify against a non-Newtonian run. |
| Small vessel or strong recirculation | Non-Newtonian is more defensible | Include diameter, haematocrit, temperature, and patient variability. |

Le chemin `transportProperties` de base de FoamPilot convient naturellement aux cas à propriétés constantes. Une loi non newtonienne exige un solveur et une configuration de dictionnaire qui évaluent effectivement la viscosité à partir du taux de cisaillement local. Assigner une variable Python descriptive sans vérifier le dictionnaire OpenFOAM généré n'active pas un modèle rhéologique.

## 2.3 Pulsatilité et nombre de Womersley

L'écoulement sanguin est généralement pulsatile. Le nombre de Womersley compare l'inertie non stationnaire à la diffusion visqueuse :

$$
\alpha=R\sqrt{\frac{\omega\rho}{\mu}},
$$

où $R$ est le rayon du vaisseau et $\omega$ est la fréquence angulaire de la forme d'onde cardiaque. Un faible $\alpha$ produit un profil plus proche de l'écoulement parabolique quasi-stationnaire. Un $\alpha$ plus élevé produit un cœur plus plat et un déphasage plus important entre le gradient de pression et la réponse pariétale.

Pour les simulations pulsées, l'entrée doit être représentée par une onde de débit mesurée ou synthétique. L'onde doit être convertie en profil de vitesse en utilisant la surface d'entrée réelle et, si possible, un profil développé ou compatible avec Womersley. Une vitesse uniforme à l'entrée d'une artère fortement courbée ou ramifiée peut générer des effets d'entrée artificiels qui contaminent la région d'intérêt.

## 2.4 Conditions aux limites en CFD vasculaire

L'incertitude biomédicale la plus importante est souvent la condition aux limites plutôt que la discrétisation intérieure.

| Boundary | Common data | Physical issue |
| --- | --- | --- |
| Inlet | Flow rate, velocity profile, pressure, or patient waveform | Measured plane may be far from the computational inlet. |
| Outlet | Fixed pressure, traction, resistance, impedance, or Windkessel | Downstream vasculature is truncated. |
| Wall | No-slip rigid wall, moving wall, or fluid-structure coupling | Wall compliance can change pressure and WSS. |
| Branch | Flow split or pressure relation | Patient-specific downstream resistance is uncertain. |

### Modèle de sortie Windkessel

Un modèle Windkessel représente la résistance et la compliance du réseau vasculaire en aval d'une sortie tronquée. Un modèle à trois éléments courant combine une résistance proximale $R_1$, une compliance $C$, et une résistance distale $R_2$. En forme pression-débit :

$$
C\frac{dP_c}{dt}=Q-\frac{P_c-P_d}{R_2},
$$

$$
P=P_c+R_1Q,
$$

où $Q$ est le débit de sortie, $P$ est la pression de sortie, $P_c$ est la pression du condensateur, et $P_d$ est la pression de référence distale. Le modèle est choisi parce qu'une sortie à pression fixe ne peut pas reproduire le stockage et la réponse retardée du réseau en aval.

FoamPilot expose `WindkesselModel` comme un add-on de modèle. Avant de l'utiliser, définissez la convention de signe, les unités, la pression initiale du condensateur, la référence de pression, et le pas de couplage temporel. Calibrez $R_1$, $R_2$, et $C$ par rapport à des données pression-débit mesurées ou à une hypothèse physiologique documentée. Un modèle Windkessel est une représentation de frontière d'ordre réduit ; ce n'est pas un modèle complet de la circulation cardiovasculaire.

## 2.5 Géométrie patient-spécifique et provenance des données

Un cas biomédical commence couramment par une surface segmentée issue de CTA, MRI, CT, NIfTI, STL, VTP, ou autre. La chaîne de traitement doit documenter :

1. modalité d'imagerie, résolution, date d'acquisition et orientation ;
2. méthode de segmentation et décisions de seuillage ;
3. opérations de lissage et de fermeture des trous ;
4. longueur d'extension des entrées/sorties ;
5. tolérance de remeshing de la surface et nombre de triangles ;
6. conversion en unités métriques ;
7. étiquettes de branches et noms de patchs ;
8. qualité de maille et conservation de volume ;
9. source et calibration des conditions aux limites ;
10. anonymisation et gouvernance des données.

Les utilitaires FoamPilot incluent des helpers NIfTI-to-STL et de nettoyage de surface vasculaire. Ce sont des outils de traitement de géométrie, pas des validateurs cliniques de segmentation. Inspectez la sortie visuellement et quantitativement avant de lancer le calcul.

## 2.6 Grandeurs de validation biomédicale

Les grandeurs suivantes sont souvent rapportées :

- chute de pression et rapport de pression trans-lésionnel ;
- contrainte de cisaillement pariétale moyenne dans le temps ;
- oscillatory shear index ;
- temps de résidence relatif ;
- volume de recirculation ;
- répartition de débit aux embranchements ;
- valeurs pico-systoliques et fin-diastoliques ;
- conservation du débit à travers les sorties.

N'interprétez pas un pic local de WSS sans une étude de convergence de maille et de résolution près de la paroi. La WSS est une dérivée à la paroi et est particulièrement sensible au lissage de la surface, à l'espacement de la maille, à la résolution temporelle, et à la rhéologie.

# 3. Vent extérieur et couches limites atmosphériques

## 3.1 Pourquoi une entrée uniforme est souvent incorrecte

Une simulation de bâtiment ou d'écoulement urbain n'est pas simplement un cas automobile pivoté verticalement. Près du sol, la vitesse moyenne du vent augmente avec la hauteur et la turbulence varie avec la hauteur. Les bâtiments perturbent cette couche limite atmosphérique entrante, créant des accélérations autour des arêtes, des séparations sur les toitures, de la recirculation en canyon urbain, et des traînées.

Une entrée uniforme peut être acceptable pour une soufflerie contrôlée ou pour une étude méthodologique simplifiée. Elle n'est généralement pas cohérente avec une couche limite atmosphérique à moins que le domaine et les conditions aux limites ne soient construits délibérément pour que le profil se développe avant la région d'intérêt.

## 3.2 Équations gouvernantes

Pour le vent extérieur à faible vitesse, l'air est couramment traité comme incompressible :

$$
\nabla\cdot\mathbf{U}=0,
$$

$$
\frac{\partial\mathbf{U}}{\partial t}+\mathbf{U}\cdot\nabla\mathbf{U}
=-\frac{1}{\rho}\nabla p+\nabla\cdot[(\nu+\nu_t)\nabla\mathbf{U}],
$$

avec une viscosité turbulente $\nu_t$ fournie par une fermeture telle que $k$–$\epsilon$, realizable $k$–$\epsilon$, RNG $k$–$\epsilon$, ou $k$–$\omega$ SST.

Le bon choix dépend de la grandeur de sortie. Un modèle RANS stationnaire est efficace pour la vitesse moyenne du vent et la pression moyenne. URANS est nécessaire lorsque l'unsteadiness cohérente est importante. LES ou des méthodes hybrides sont plus appropriées lorsque les tourbillons transitoires et les fluctuations turbulentes sont des sorties primaires, mais leur coût en maillage et en pas de temps est beaucoup plus élevé.

## 3.3 Loi logarithmique du vent

La couche limite atmosphérique neutre est souvent approximée par la loi logarithmique :

$$
U(z)=\frac{u_*}{\kappa}\ln\left(\frac{z+z_0}{z_0}\right),
$$

où $u_*$ est la vitesse de frottement, $\kappa\approx0.4$ est la constante de von Kármán, et $z_0$ est la longueur de rugosité aérodynamique. Si le vent de référence est connu à la hauteur $z_r$ :

$$
U(z)=U(z_r)\frac{\ln[(z+z_0)/z_0]}{\ln[(z_r+z_0)/z_0]}.
$$

La loi logarithmique est choisie parce qu'elle représente la vitesse moyenne dans la couche superficielle sous les hypothèses de neutralité et d'homogénéité horizontale. Elle ne représente pas automatiquement la stratification thermique, les canopées forestières, le terrain complexe, ou la météo fortement transitoire.

OpenFOAM fournit des conditions aux limites de couche limite atmosphérique basées sur des profils de type loi log et des quantités turbulentes. Sa documentation décrit `atmBoundaryLayerInletVelocity`, `atmBoundaryLayerInletK`, `atmBoundaryLayerInletEpsilon`, et `atmBoundaryLayerInletOmega`, ainsi que des wall functions atmosphériques et des termes sources [2].

## 3.4 Profil en loi de puissance

Une alternative d'ingénierie est :

$$
U(z)=U_r\left(\frac{z}{z_r}\right)^\alpha,
$$

où $\alpha$ est un exposant de cisaillement empirique. La loi en puissance est pratique quand des données de vent sont disponibles à deux hauteurs ou lorsqu'une norme d'ingénierie du vent fournit un exposant. Elle est moins directement liée à la rugosité de surface que la loi log et ne doit pas être mélangée avec une longueur de rugosité sans indiquer la conversion.

## 3.5 Données turbulentes à l’entrée

L'entrée doit définir non seulement la vitesse mais aussi la turbulence. Pour un modèle $k$–$\epsilon$, une estimation courante est :

$$
 k=\frac32(UI)^2,
$$

où $I$ est l'intensité de turbulence. Une échelle de longueur $L$ peut être utilisée pour estimer :

$$
\epsilon=C_\mu^{3/4}\frac{k^{3/2}}{L},
$$

et pour un modèle $k$–$\omega$ :

$$
\omega\approx\frac{\sqrt{k}}{C_\mu^{1/4}L}.
$$

Ces formules sont des hypothèses de modélisation, pas des mesures. Les profils de $U$, $k$, $\epsilon$, ou $\omega$ doivent être mutuellement compatibles ; sinon la couche limite atmosphérique peut dériver, accélérer, ou décroître avant d'atteindre les bâtiments.

## 3.6 Stabilité et poussée d’Archimède

L'écoulement neutre néglige la stratification thermique. Des conditions atmosphériques stables ou instables exigent des hypothèses sur la température, la flottabilité, et la production de turbulence. Le signe et l'amplitude du terme de flottabilité influencent le mélange vertical, la récupération des traînées, et le vent au niveau piéton.

Pour un cas urbain thermique simplifié, une approximation de Boussinesq peut être utilisée lorsque les différences de température sont faibles. Pour une stratification plus importante ou des changements de densité, un modèle compressible ou à densité variable est plus approprié. Le choix doit être cohérent avec les données météorologiques disponibles et la formulation thermophysique du solveur.

## 3.7 Modélisation du domaine et des parois

Un domaine extérieur doit fournir une fetch amont adéquate, une longueur de traînée aval suffisante, un dégagement latéral et un dégagement en hauteur suffisants. Le sol n'est pas juste un autre mur : sa rugosité détermine le profil de vitesse et la génération de turbulence. Les wall functions, les paramètres de rugosité, la hauteur du premier élément, et le profil d'entrée atmosphérique doivent être choisis comme un système cohérent.

La raison principale d'utiliser une loi ou une condition aux limites spécifique est la **consistance d'équilibre**. Si le profil d'entrée implique une rugosité et que le mur de sol implique une autre, le profil évolue artificiellement. La première tâche est donc de vérifier un cas précurseur ou un domaine vide avant d'ajouter les bâtiments.

## 3.8 Sorties urbaines

Les sorties pertinentes incluent la vitesse moyenne à hauteur piétonne, la probabilité de dépassement si des données transitoires sont disponibles, la pression sur les façades, des indicateurs de confort au vent, l'accélération des toits, la circulation en canyon urbain, l'intensité de turbulence, et le transport de polluants lorsqu'une équation scalaire est couplée.

Un résultat de vent doit toujours préciser la hauteur de référence, la rugosité, la direction du vent, la stabilité atmosphérique, le modèle de turbulence, les dimensions du domaine, le nombre de cellules de la maille, le traitement des murs, et l'intervalle de moyenne.

# 4. Thermorégulation humaine

## 4.1 Niveaux de couplage

La thermorégulation peut être représentée à plusieurs niveaux :

| Level | Description | Suitable use |
| --- | --- | --- |
| Convective boundary condition | Prescribed heat-transfer coefficient or skin temperature. | Simple thermal CFD around a body. |
| Multi-node physiology | Core, blood, muscle, fat, and skin temperatures with regulatory responses. | Coupling CFD environment with human thermal response. |
| Detailed local physiology | Segment-level metabolism, perfusion, sweating, clothing, radiation, and posture. | Research studies requiring local response. |
| Fully coupled human-fluid model | Physiological state changes alter surface fluxes and flow conditions. | Advanced research; requires careful time coupling and validation. |

Le workflow MakeHuman/JOS-3 de FoamPilot appartient au niveau de couplage géométrie-plus-physiologie. MakeHuman fournit une surface corporelle ; JOS-3 fournit une réponse thermique multi-nœuds ; OpenFOAM résout l'écoulement environnant et le transfert de chaleur.

## 4.2 Concept du modèle JOS-3

JOS-3 est un modèle numérique de thermorégulation humaine qui prédit des grandeurs telles que la température centrale, la température cutanée, la sudation, le flux sanguin, et les réponses thermiques pour 17 segments du corps et pour l'ensemble du corps [3] [4]. Il dérive de modèles multi-nœuds antérieurs et utilise un réseau physiologique de compartiments tissulaires et de signaux régulateurs.

Le modèle contient le stockage et le transfert de chaleur à travers les tissus du corps, la perfusion sanguine, la production métabolique, les pertes respiratoires, la conduction, la convection, le rayonnement, et l'évaporation. Les réponses régulatrices peuvent inclure vasodilatation, vasoconstriction, sudation, frisson, thermogenèse non frissonnante, et des changements liés à l'activité ou à la posture.

Le modèle doit être considéré comme un modèle physiologique lumped ou multi-nœuds, pas comme un modèle CFD vasculaire résolu. Un champ CFD peut fournir la température d'air locale, la vitesse d'air, des entrées liées à l'humidité, et des conditions radiatives, tandis que JOS-3 renvoie des températures cutanées par segment et des signaux de perte de chaleur.

## 4.3 Bilan thermique humain

Un bilan thermique humain simplifié est :

$$
M-W=Q_{sk}+Q_{res}+S,
$$

où $M$ est la production de chaleur métabolique, $W$ le travail externe, $Q_{sk}$ la perte de chaleur totale par la peau, $Q_{res}$ la perte de chaleur respiratoire, et $S$ le stockage de chaleur corporelle. La perte de chaleur cutanée peut être décomposée en :

$$
Q_{sk}=Q_{conv}+Q_{rad}+Q_{cond}+Q_{evap}.
$$

Le modèle CFD résout ou approximativement estime le transfert convectif. Le rayonnement peut être modélisé avec un solveur de radiation ou représenté par une température radiante moyenne. L'évaporation dépend de l'humidité, de l'isolation des vêtements, de l'humidité de la peau (skin wettedness), et des différences de pression de vapeur ; elle n'est pas déterminée par la seule vitesse de l'air.

## 4.4 Pourquoi les données locales à 17 zones sont importantes

Une température corporelle moyenne masque l'exposition locale. Une personne peut avoir un visage chaud, des mains refroidies, un torse isolé, et un écoulement asymétrique en même temps. JOS-3 accepte des valeurs environnementales et d'habillement locales pour 17 segments du corps. Le workflow de géométrie de FoamPilot crée des patchs de surface correspondants et un `zone_mapping.csv` afin que les résultats CFD puissent être agrégés de manière cohérente.

Le mapping doit documenter :

- les noms exacts des parties du corps et leur ordre ;
- le nom du patch de surface dans le STL ou la maille OpenFOAM ;
- la surface représentée par chaque patch ;
- si un patch est exposé, vêtu, ou occlus ;
- comment la vitesse locale, la température, et le rayonnement sont moyennés ;
- la convention de signe pour le flux de chaleur ;
- l'interpolation temporelle entre CFD et physiologie.

## 4.5 Lois de convection autour du corps

Le flux de chaleur convectif s'écrit souvent :

$$
q''_{conv}=h_c(T_{skin}-T_a),
$$

où $h_c$ est un coefficient local de transfert de chaleur convectif et $T_a$ est la température de l'air. Dans un couplage CFD, $h_c$ peut être estimé à partir du flux de chaleur pariétal résolu :

$$
 h_c=\frac{q''_{conv}}{T_{skin}-T_a},
$$

ou à partir d'une corrélation empirique basée sur la vitesse locale, une longueur caractéristique, et l'orientation. La voie dérivée du CFD est plus spatialement résolue, mais elle dépend de la qualité de la maille, du traitement des murs, des conditions aux limites de température de surface, et du modèle de turbulence.

Pour une corrélation simplifiée, le nombre de Nusselt peut dépendre d'un nombre de Reynolds et du nombre de Prandtl :

$$
Nu=\frac{h_cL}{k_a}=f(Re_L,Pr),
$$

avec

$$
Re_L=\frac{U L}{\nu_a},
\qquad
Pr=\frac{\nu_a}{\alpha_a}.
$$

Le choix entre corrélation et CFD doit être explicite. Les corrélations sont peu coûteuses et utiles pour la conception préliminaire ; la CFD est utile lorsque la séparation d'écoulement, la recirculation, la posture, la géométrie des vêtements, ou l'asymétrie spatiale sont importantes.

## 4.6 Rayonnement et température radiante moyenne

Le rayonnement n'est pas équivalent à la température de l'air. Un corps peut être dans de l'air frais tout en recevant un fort rayonnement long-onde ou solaire provenant des surfaces environnantes. Un couplage pratique avec la physiologie fournit donc la température d'air $T_a$, la température radiante moyenne $T_r$, la vitesse d'air $V_a$, l'humidité relative, l'isolation des vêtements, le niveau d'activité, et la posture.

Si le cas CFD ne résout pas le rayonnement, utilisez une entrée $T_r$ documentée plutôt que de fixer silencieusement $T_r=T_a$. Si la charge solaire est importante, distinguez l'absorption du rayonnement solaire court de l'échange long-onde et enregistrez l'émissivité et l'absorptivité des surfaces.

## 4.7 Échange de données entre la CFD et JOS-3

Une boucle de couplage robuste est :

```text
MakeHuman surface
→ surface cleanup and JOS-3 patch generation
→ CFD mesh and patch mapping
→ OpenFOAM temperature/velocity/radiation solution
→ area-weighted segment averages
→ JOS-3 physiological update
→ updated skin temperature or heat-flux boundary data
→ next CFD interval
```

Le couplage peut être unidirectionnel ou bidirectionnel :

| Coupling | CFD receives | Physiology receives | Use |
| --- | --- | --- | --- |
| One-way | Fixed skin temperature or heat flux | CFD air conditions | Initial feasibility study. |
| Loose two-way | Updated segment skin temperature or flux | Local CFD temperature, speed, radiation, humidity proxy | Practical transient coupling. |
| Strong two-way | Iterated thermal boundary condition within each timestep | Converged local environmental state | Expensive research coupling. |

Le pas de temps doit résoudre à la fois les transitoires CFD et la réponse physiologique. Une mise à jour physiologique à chaque itération CFD peut être inutile ; un intervalle de couplage très grand peut rater des changements d'exposition rapides. Testez l'intervalle de couplage comme paramètre numérique.

## 4.8 Validation de la thermorégulation

La validation doit comparer séparément le modèle physiologique et le modèle CFD environnant avant de juger le système couplé. Pour la partie CFD, validez la vitesse, la température, le flux pariétal, et la convergence de maille. Pour la partie physiologie, vérifiez les températures de référence cutanée/centrale, la réponse métabolique, la sudation, le flux sanguin, et les réponses attendues à des expositions thermiques contrôlées.

Une sortie de thermorégulation telle que la température cutanée moyenne est une prédiction de modèle avec incertitude physiologique. Elle ne doit pas être présentée comme un diagnostic clinique ou comme une réponse humaine validée sans comparaison expérimentale.

## Références

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8450390/ "Liu et al., Comparison of Newtonian and Non-newtonian Fluid Models in Blood Flow Simulation"

[2]: https://www.openfoam.com/news/main-news/openfoam-v20-06/boundary-conditions "OpenFOAM: atmospheric boundary-layer boundary conditions"

[3]: https://github.com/TanabeLab/JOS-3 "TanabeLab/JOS-3: Joint system thermoregulation model"

[4]: https://doi.org/10.1016/j.enbuild.2020.110575 "Takahashi et al., Thermoregulation model JOS-3 with new open source code"
