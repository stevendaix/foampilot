# Theory Guide — Modélisation du climat urbain avec FoamPilot et OpenFOAM Foundation 13

**Version :** 1.0  
**Périmètre :** `urbanMicroclimateFoam`, cas générés par `UrbanClimateNativeCaseBuilder`  
**Auteur :** Manus AI

## Résumé

Le climat urbain est le résultat d’un couplage multi-échelle entre la géométrie bâtie, l’écoulement atmosphérique, les échanges de chaleur sensible et latente, le stockage dans les matériaux, le rayonnement solaire et infrarouge, ainsi que la végétation. Un canyon de rue ne se comporte donc pas comme une simple cavité isotherme : les bâtiments modifient la turbulence et le facteur de vue du ciel, les surfaces stockent puis restituent de l’énergie, tandis que les arbres absorbent le rayonnement, freinent l’écoulement et transpirent.

Ce guide présente d’abord la physique générale, puis les équations utilisées par le solveur multi-région porté dans FoamPilot. Il distingue explicitement les **équations continues**, les **fermetures physiques**, les **approximations numériques** et les **choix d’implémentation**. Le dépôt `urbanMicroclimateFoam` décrit le même principe général : CFD turbulent, transport de chaleur et d’humidité dans l’air, stockage HAM dans les matériaux poreux et rayonnement calculé par facteurs de vue [1].

> **Attention méthodologique.** Une exécution OpenFOAM terminée par `End` prouve la cohérence logicielle du cas et la stabilité de la chaîne de calcul sur l’horizon configuré. Elle ne prouve ni la convergence vers un état physique réaliste, ni la validité des paramètres météorologiques, ni l’accord avec des mesures.

## 1. Objet et échelles du climat urbain

Le climat urbain se décrit à plusieurs échelles. L’échelle météorologique régionale impose les conditions incidentes : vitesse et direction du vent, température, humidité, rayonnement et précipitations. L’échelle du quartier transforme ces conditions par la morphologie : hauteur et espacement des bâtiments, orientation des rues, rugosité, arbres et matériaux. L’échelle du piéton concerne les grandeurs utiles à l’usage : vitesse à 1,5–2 m, température de l’air, humidité relative, température radiante moyenne et indice de confort.

L’îlot de chaleur urbain est souvent défini comme une différence entre une grandeur urbaine et une référence rurale :

$$
\Delta T_{UHI}=T_{urbain}-T_{rural}.
$$

Cette définition est utile pour comparer des situations, mais un calcul CFD local ne fournit pas automatiquement une température rurale de référence. Il fournit surtout un champ spatial de vitesse, pression, température, humidité et flux, dont l’interprétation doit rester liée aux conditions aux limites.

La morphologie agit par plusieurs mécanismes. Une faible ouverture au ciel diminue le refroidissement radiatif nocturne ; des surfaces minérales absorbent et stockent la chaleur ; les bâtiments dissipent l’énergie anthropique ; la végétation ajoute de l’ombrage et de l’évapotranspiration mais augmente aussi la traînée et peut réduire le facteur de vue du ciel. Ces mécanismes sont recensés dans les revues du microclimat urbain [2] [3].

## 2. Variables, conventions et bilans

On note $\mathbf{x}$ la position, $t$ le temps, $\mathbf{U}$ la vitesse de l’air, $p$ la pression, $T$ la température absolue, $\rho$ la masse volumique, $\mu$ la viscosité dynamique, $\nu=\mu/\rho$ la viscosité cinématique, $k$ l’énergie cinétique turbulente et $\varepsilon$ son taux de dissipation. Le vecteur gravité est $\mathbf{g}$.

Les flux surfaciques sont exprimés en $\mathrm{W\,m^{-2}}$. Le flux de chaleur sensible est $q_H$, le flux latent est $q_E$, le flux solaire absorbé est $q_S$ et le flux infrarouge net est $q_L$. Pour une surface, le bilan énergétique idéal s’écrit :

$$
R_n + q_{anth} = q_H + q_E + q_G,
$$

avec

$$
R_n=q_S+q_L,
$$

où $q_G$ représente le flux stocké ou conduit dans la surface et $q_{anth}$ les apports anthropiques éventuels. Dans les cas présents, les apports anthropiques ne sont pas introduits comme modèle général indépendant ; ils peuvent être représentés par des flux ou des sources ajoutés dans les dictionnaires du cas.

## 3. Écoulement de l’air : équations de Navier–Stokes

### 3.1 Conservation de la masse

Pour un fluide compressible général :

$$
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf{U})=0.
$$

Dans un écoulement faiblement compressible ou incompressible à densité constante, la forme devient :

$$
\nabla\cdot\mathbf{U}=0.
$$

Le solveur utilise une formulation avec pression modifiée $p_{rgh}$ dans les cas soumis à la gravité :

$$
 p_{rgh}=p-\rho\,\mathbf{g}\cdot\mathbf{x},
$$

à un choix de constante de référence près. Cette variable sépare la contribution hydrostatique de la pression dynamique et améliore le traitement numérique des écoulements buoyants.

### 3.2 Quantité de mouvement

La conservation de la quantité de mouvement s’écrit :

$$
\frac{\partial(\rho\mathbf{U})}{\partial t}
+\nabla\cdot(\rho\mathbf{U}\otimes\mathbf{U})
=-\nabla p+\nabla\cdot\boldsymbol{\tau}+\rho\mathbf{g}+\mathbf{S}_{veg}+\mathbf{S}_{autres},
$$

avec le tenseur visqueux newtonien :

$$
\boldsymbol{\tau}=\mu\left[\nabla\mathbf{U}+(\nabla\mathbf{U})^T-\frac{2}{3}(\nabla\cdot\mathbf{U})\mathbf{I}\right].
$$

La force de traînée végétale est traitée comme une source volumique opposée à l’écoulement. Une forme générique est :

$$
\mathbf{S}_{veg}=-\rho\,C_f\,|\mathbf{U}|\,\mathbf{U},
$$

où $C_f$ possède la dimension nécessaire pour que $\mathbf{S}_{veg}$ soit une force par volume. Dans une canopée, une paramétrisation usuelle relie cette quantité à la surface foliaire par volume $LAD$ et à un coefficient de traînée $C_d$ :

$$
C_f \simeq \frac{1}{2}C_d\,LAD.
$$

Le générateur FoamPilot écrit le champ `LAD` avec la dimension $[0\,-1\,0]$, c’est-à-dire une surface foliaire par volume. Cette correction dimensionnelle est essentielle : une valeur sans dimension rendrait la source de quantité de mouvement incohérente.

### 3.3 Approximation de Boussinesq et flottabilité

Lorsque les variations de densité sont faibles et principalement dues à la température, la force de flottabilité peut être linéarisée :

$$
\rho\mathbf{g}\simeq \rho_0\mathbf{g}
-\rho_0\beta_T(T-T_0)\mathbf{g},
$$

où $\beta_T$ est le coefficient de dilatation thermique. La partie hydrostatique est absorbée dans $p_{rgh}$ ; la partie variable agit comme source convective. Le nombre de Richardson permet d’estimer l’importance relative de la flottabilité et du cisaillement :

$$
Ri=\frac{g\,\beta_T\,\Delta T\,L}{U^2}.
$$

Un $Ri$ faible indique une dynamique dominée par le vent ; un $Ri$ important signale un couplage thermique fort. Ce nombre ne remplace pas une résolution CFD, mais aide à choisir les conditions initiales et le modèle de turbulence.

## 4. Turbulence : formulation RANS

### 4.1 Décomposition de Reynolds

La formulation RANS décompose toute grandeur instantanée en moyenne et fluctuation :

$$
\mathbf{U}=\overline{\mathbf{U}}+\mathbf{u}',\qquad
T=\overline{T}+T'.
$$

Après moyenne, le terme convectif introduit le tenseur de contraintes turbulentes $-\rho\overline{u_i'u_j'}$. L’hypothèse de Boussinesq modélise ce tenseur par une viscosité turbulente $\mu_t$ :

$$
-\rho\overline{u_i'u_j'}
=2\mu_t\overline{S}_{ij}-\frac{2}{3}\rho k\delta_{ij},
$$

avec

$$
\overline{S}_{ij}=\frac{1}{2}
\left(\frac{\partial \overline U_i}{\partial x_j}
+\frac{\partial \overline U_j}{\partial x_i}\right).
$$

La viscosité effective devient $\mu_{eff}=\mu+\mu_t$. Les scalaires thermiques utilisent généralement une diffusivité turbulente :

$$
\alpha_{eff}=\alpha+\frac{\nu_t}{Pr_t},
$$

où $Pr_t$ est le nombre de Prandtl turbulent.

### 4.2 Modèle $k$–$\varepsilon$ réalisable

Les profils HAM utilisent dans le générateur le modèle `realizableKE`. Sa structure générale est :

$$
\frac{\partial k}{\partial t}+\mathbf{U}\cdot\nabla k
= P_k-\varepsilon+\nabla\cdot
\left[\left(\nu+\frac{\nu_t}{\sigma_k}\right)\nabla k\right],
$$

$$
\frac{\partial\varepsilon}{\partial t}+\mathbf{U}\cdot\nabla\varepsilon
= C_1\frac{\varepsilon}{k}P_k
-C_2\frac{\varepsilon^2}{k+\sqrt{\nu\varepsilon}}
+\nabla\cdot
\left[\left(\nu+\frac{\nu_t}{\sigma_\varepsilon}\right)\nabla\varepsilon\right],
$$

avec

$$
P_k=2\nu_t\,\overline S_{ij}\overline S_{ij}.
$$

Le modèle réalisable modifie la fermeture par rapport au $k$–$\varepsilon$ standard afin de maintenir des contraintes turbulentes admissibles dans des écoulements fortement cisaillés. La valeur de $\nu_t$ est déduite des variables turbulentes selon la fermeture choisie. Les profils purement CFD peuvent utiliser `laminar` dans le générateur, tandis que les profils de vent et HAM activent une fermeture RANS.

### 4.3 Végétation poreuse

Une canopée explicitement maillée n’est pas résolue feuille par feuille. Le volume végétalisé est homogénéisé et reçoit une force de traînée dépendant de $LAD$. Cette approche est efficace à l’échelle du quartier, mais elle ne résout pas les tourbillons entre feuilles. Le choix de $LAD$, de $C_d$ et de la distribution spatiale domine alors la prédiction de la vitesse dans la canopée.

## 5. Transport de chaleur et d’humidité dans l’air

### 5.1 Énergie sensible

Une forme conservative de l’équation d’énergie est :

$$
\frac{\partial(\rho h)}{\partial t}
+\nabla\cdot(\rho\mathbf{U}h)
=\nabla\cdot(\alpha_h\nabla h)+S_h,
$$

où $h$ est l’enthalpie spécifique. Pour un gaz à capacité calorifique constante :

$$
 h\simeq c_p T,
$$

et l’équation peut être écrite directement sur $T$ :

$$
\rho c_p\left(\frac{\partial T}{\partial t}
+\mathbf{U}\cdot\nabla T\right)
=\nabla\cdot(k_{eff}\nabla T)+S_T.
$$

Le terme $S_T$ contient la flottabilité indirecte, les échanges de surface, les sources végétales et, selon la configuration, les effets radiatifs convertis en flux thermique.

### 5.2 Humidité de l’air

On peut transporter une fraction massique de vapeur $Y_v$ :

$$
\frac{\partial(\rho Y_v)}{\partial t}
+\nabla\cdot(\rho\mathbf{U}Y_v)
=\nabla\cdot(\rho D_{v,eff}\nabla Y_v)+S_v.
$$

La source $S_v$ est positive lors de l’évaporation ou de la transpiration et négative lors de la condensation. Le flux latent associé est :

$$
q_E=L_v\,\dot m_v,
$$

où $L_v$ est la chaleur latente de vaporisation et $\dot m_v$ le flux massique de vapeur.

Dans les conditions limites HAM, les variables peuvent être formulées en humidité relative, pression de vapeur ou pression capillaire selon la région. Il faut donc toujours vérifier les dimensions et la convention du champ au lieu de supposer que `w`, `pc` et $Y_v$ représentent la même grandeur.

## 6. Modèle HAM des matériaux poreux

### 6.1 Principe

HAM signifie **Heat, Air and Moisture**. Dans les cas présents, les régions `ground` et `buildings` sont solides et le modèle résout le stockage thermique ainsi que le transport et le stockage d’humidité dans les matériaux poreux. Le couplage avec l’air s’effectue aux interfaces multi-région par température, flux thermique, pression de vapeur ou humidité.

### 6.2 Équation de chaleur dans le solide humide

Une formulation enthalpique générique est :

$$
\frac{\partial}{\partial t}
\left(\rho_s c_s T+\rho_l L_v w_l\right)
=\nabla\cdot\left(k_T\nabla T
+L_v\,\mathbf{J}_v\right)+S_T,
$$

où $\rho_s c_s T$ est le stockage sensible, $\rho_l L_v w_l$ l’énergie associée à la phase liquide et $\mathbf{J}_v$ le flux massique de vapeur. Selon le modèle de matériau, les termes de couplage peuvent être écrits sous forme de conductivités apparentes :

$$
\frac{\partial H}{\partial t}
=\nabla\cdot(K_{TT}\nabla T+K_{Tp}\nabla p_c)+S_H.
$$

### 6.3 Équation de transport de l’humidité

Une forme générale en pression capillaire $p_c$ et teneur en eau $w$ est :

$$
\frac{\partial w}{\partial t}
=\nabla\cdot\left(K_{pT}\nabla T+K_{pp}\nabla p_c\right)+S_w.
$$

La matrice de transport couplée s’écrit :

$$
\begin{bmatrix}
\partial H/\partial t\\[2pt]
\partial w/\partial t
\end{bmatrix}
=\nabla\cdot
\left[
\begin{pmatrix}
K_{TT}&K_{Tp}\\
K_{pT}&K_{pp}
\end{pmatrix}
\begin{pmatrix}
\nabla T\\
\nabla p_c
\end{pmatrix}
\right]
+\begin{bmatrix}S_H\\S_w\end{bmatrix}.
$$

Les coefficients sont calculés par les classes de matériaux. Dans le code porté, les matériaux Hamstad utilisent notamment une pression de vapeur saturante, une humidité relative et une diffusion de vapeur dépendante de la température.

### 6.4 Pression de vapeur saturante et humidité relative

Le code des matériaux utilise une corrélation de type :

$$
 p_{vsat}(T)=\exp\left(65.8094-
\frac{7066.27}{T}-5.976\ln T\right),
$$

avec $T$ en kelvins et $p_{vsat}$ en pascals dans la convention du modèle. La relation entre pression capillaire et humidité relative est :

$$
\phi=\frac{p_v}{p_{vsat}}
=\exp\left(\frac{p_c}{\rho_l R_v T}\right),
$$

où $R_v$ est la constante spécifique de la vapeur d’eau et $\rho_l$ la masse volumique de l’eau liquide. Une pression capillaire négative correspond à une humidité relative inférieure à l’unité.

La dérivée utile pour la linéarisation est :

$$
\frac{\partial p_{vsat}}{\partial T}
=p_{vsat}\left(\frac{7066.27}{T^2}-\frac{5.976}{T}\right).
$$

### 6.5 Isotherme de Van Genuchten

Pour les matériaux dont la rétention d’eau est décrite par Van Genuchten, une écriture usuelle est :

$$
S_e=\left[1+(\alpha|p_c|)^n\right]^{-m},
\qquad m=1-\frac{1}{n},
$$

puis

$$
 w=w_r+(w_s-w_r)S_e,
$$

où $w_r$ et $w_s$ sont les teneurs résiduelle et saturée. La perméabilité relative peut être calculée par la relation de Mualem–Van Genuchten :

$$
K_r=S_e^\ell
\left[1-(1-S_e^{1/m})^m\right]^2.
$$

Le code peut aussi utiliser des variantes avec diffusion de vapeur (`VanGenuchtenVapDiff`). Le point important pour l’utilisateur est que $w$, $K_r$, $K_v$ et $K_{pT}$ ne sont pas des constantes indépendantes : ils dépendent fortement de $p_c$ et $T$.

## 7. Rayonnement solaire et infrarouge

### 7.1 Bilan radiatif d’une surface

Pour une surface opaque grise, le flux net est :

$$
q_{rad}=q_{SW,abs}+q_{LW,net}.
$$

Le flux solaire absorbé dépend de l’albédo $\alpha$ :

$$
q_{SW,abs}=(1-\alpha)G,
$$

où $G$ est l’irradiance incidente. Le rayonnement thermique émis suit Stefan–Boltzmann :

$$
q_{LW,emit}=\varepsilon\sigma T^4,
$$

avec $\varepsilon$ l’émissivité, $\sigma=5.670374419\times10^{-8}\,\mathrm{W\,m^{-2}\,K^{-4}}$. Pour un échange entre surfaces $i$ et $j$ :

$$
q_{i\rightarrow j}=A_iF_{ij}\,\varepsilon_i\sigma(T_i^4-T_j^4),
$$

dans une forme simplifiée où les réflexions multiples sont négligées. Les modèles de facteurs de vue complets résolvent les échanges entre toutes les surfaces agglomérées.

### 7.2 Facteurs de vue

Le facteur de vue $F_{ij}$ est la fraction de l’énergie quittant la surface $i$ qui atteint directement $j$. Il vérifie :

$$
0\leq F_{ij}\leq1,
\qquad \sum_jF_{ij}=1
$$

lorsque toutes les destinations sont incluses. La réciprocité impose :

$$
A_iF_{ij}=A_jF_{ji}.
$$

Dans FoamPilot, les étapes `faceAgglomerate` et `viewFactorsGen` réduisent le coût en regroupant les faces et en calculant une matrice de facteurs de vue sur `finalAgglom`. Les fichiers `cellZones` identifient les volumes utilisés par la chaîne générative. La qualité de la matrice dépend directement de la géométrie, de l’orientation des normales, des surfaces visibles et du respect des sommes de lignes.

Le facteur de vue du ciel, ou sky view factor, mesure la fraction d’une direction hémisphérique ouverte vers le ciel. Un canyon compact possède un facteur de vue réduit ; il reçoit moins de soleil direct mais peut aussi perdre moins de rayonnement infrarouge vers le ciel la nuit.

### 7.3 Modèle `viewFactorSky`

Le modèle `viewFactorSky` utilise une approche surface-à-surface enrichie par une contribution du ciel. Les champs `qr` et les conditions limites `greyDiffusiveRadiationViewFactor` représentent les échanges infrarouges sur les patches. Les paramètres `emissivityMode`, `emissivity` et `qro` doivent rester cohérents avec la version de la bibliothèque compilée.

Dans l’intégration Foundation 13, les régions radiatives utilisent des dictionnaires `radiationProperties` séparés. Les interfaces vers les régions voisines doivent employer des patches multi-région compatibles (`mapped` ou `mappedWall`) et des champs radiatifs spécialisés. Une condition `fixedValue` générique peut provoquer une erreur de cast lorsque le modèle attend `greyDiffusiveRadiationViewFactor`.

### 7.4 Modèle `directAndDiffuse`

Le rayonnement solaire incident est séparé en composante directe et diffuse :

$$
G=G_{dir}+G_{dif}.
$$

Une surface reçoit approximativement :

$$
G_i=G_{dir}\max(0,\mathbf{s}\cdot\mathbf{n}_i)
+G_{dif}\,F_{i,sky},
$$

où $\mathbf{s}$ est la direction incidente du soleil, $\mathbf{n}_i$ la normale sortante et $F_{i,sky}$ le facteur de vue du ciel. Les réflexions peuvent ajouter des contributions secondaires selon l’albédo des surfaces.

Le champ `qs` est associé à la condition limite `solarLoadRadiationViewFactor`. Le champ `qso` sert de valeur externe ou sortante selon l’implémentation. Le champ `IDN` représente l’irradiance directe normale et `Idif` la composante diffuse ; dans les cas générés, ils sont écrits sous forme de tables temporelles `Function1` :

```text
(
    (0    800)
    (3600 800)
)
```

De même, `sunPosVector` est une table de couples temps–vecteur et non un `uniformDimensionedVectorField`. Cette distinction est nécessaire pour `solarRayTracingGen`.

## 8. Végétation et bilan énergétique foliaire

### 8.1 Surface foliaire par volume et LAI

La densité foliaire est :

$$
LAD=\frac{dA_{feuille}}{dV}\quad [\mathrm{m^2\,m^{-3}}]=[\mathrm{m^{-1}}].
$$

L’indice foliaire, ou LAI, est l’intégrale de $LAD$ dans la hauteur :

$$
LAI=\int_0^H LAD(z)\,dz.
$$

Le calcul `calcLAI` produit une grandeur de bord utilisée par le modèle de végétation. Un champ $LAD$ nul ou absent empêche le calcul du LAI ; une mauvaise dimension rend la source de traînée incohérente.

### 8.2 Bilan foliaire

Pour une feuille ou un volume foliaire équivalent, le bilan stationnaire peut s’écrire :

$$
R_{n,l}=H_l+LE_l+S_l,
$$

où $H_l$ est le flux sensible, $LE_l$ le flux latent de transpiration et $S_l$ le stockage foliaire, souvent négligé lorsque la feuille est supposée quasi-stationnaire. Le flux sensible est paramétré par une résistance aérodynamique $r_a$ :

$$
H_l=\rho c_p\frac{T_l-T_a}{r_a}.
$$

Le flux latent est paramétré par une résistance de surface $r_s$ :

$$
LE_l=\rho c_p\frac{e_s(T_l)-e_a}{\gamma(r_a+r_s)},
$$

où $e_s(T_l)$ est la pression de vapeur saturante à la température foliaire, $e_a$ la pression de vapeur de l’air et $\gamma$ la constante psychrométrique. Une forme équivalente utilise directement $L_v$ et un flux massique de vapeur.

### 8.3 Résistance aérodynamique

Une approximation simple est :

$$
 r_a\simeq\frac{1}{g_a},
$$

où $g_a$ est la conductance aérodynamique. Elle diminue avec la vitesse du vent et dépend de la taille caractéristique des feuilles, de la turbulence et de la densité foliaire.

### 8.4 Résistance stomatique

La résistance stomatique dépend de la lumière, du déficit de pression de vapeur, de la température et de l’état hydrique. Une forme paramétrique générique est :

$$
 r_s=r_{s,min}\,f_Q^{-1}f_D^{-1}f_T^{-1}f_\psi^{-1},
$$

avec des fonctions de limitation $f$ comprises entre zéro et un. Dans le modèle simplifié du dépôt, les coefficients `a1`, `a2`, `a3`, `D0`, `rsMin`, `Rg0`, `Rl0`, `betaP`, `betaD`, `H`, `kc` et `l` déterminent ces réponses. Il faut interpréter ces coefficients comme une **fermeture calibrée**, non comme des constantes universelles.

Le solveur itère sur la température foliaire $T_l$ jusqu’à satisfaire le bilan. Une fois $T_l$ obtenue, la transpiration et les sources de chaleur sont transférées à l’air et aux surfaces.

### 8.5 Effets de la végétation sur l’air

La végétation modifie simultanément :

| Mécanisme | Terme ou grandeur affectée | Effet possible |
|---|---|---|
| Traînée | $\mathbf{S}_{veg}$, $C_f$, $LAD$ | Diminution de la vitesse, modification de la turbulence |
| Ombrage | $G_{dir}$, facteurs de vue | Diminution du rayonnement reçu par le sol et les façades |
| Transpiration | $S_v$, $LE_l$ | Augmentation de l’humidité, refroidissement latent |
| Rayonnement foliaire | $q_{SW}$, $q_{LW}$ | Redistribution des flux dans la canopée |
| Stockage/émission | température foliaire | Effet diurne et nocturne dépendant des paramètres |

La littérature montre que l’effet refroidissant n’est pas monotone : humidité élevée, vent faible et résistance stomatique importante peuvent limiter l’évapotranspiration [3].

## 9. Couplage multi-région

Les régions `air`, `ground`, `buildings` et `vegetation` possèdent des maillages et des champs distincts. Une interface entre deux régions doit transmettre les grandeurs pertinentes : température, flux, humidité, pression de vapeur et rayonnement.

Pour une interface thermique idéale :

$$
T_1=T_2,
\qquad
-k_1\nabla T_1\cdot\mathbf{n}_1
=k_2\nabla T_2\cdot\mathbf{n}_2.
$$

Pour un échange convectif simplifié :

$$
q_H=h_c(T_s-T_a),
$$

où $h_c$ est le coefficient de transfert convectif. Une corrélation dimensionnelle peut être exprimée par :

$$
Nu=\frac{h_cL}{k_f}=f(Re,Pr,\text{géométrie}),
$$

avec $Re=UL/\nu$ et $Pr=\nu/\alpha$. Dans les interfaces OpenFOAM, ce transfert est encodé par des conditions limites couplées plutôt que par une simple valeur imposée.

Les patches `mapped` identifient une région et un patch voisin ; `mappedWall` ajoute le comportement de paroi. Le générateur FoamPilot réécrit directement `polyMesh/boundary` lorsque l’utilitaire `changeDictionary` n’est pas disponible. Cette opération est une étape de construction du cas, pas une modification manuelle du résultat après calcul.

## 10. Méthode des volumes finis

### 10.1 Intégration sur un volume de contrôle

Pour une équation scalaire générique :

$$
\frac{\partial(\rho\phi)}{\partial t}
+\nabla\cdot(\rho\mathbf{U}\phi)
=\nabla\cdot(\Gamma\nabla\phi)+S_\phi,
$$

l’intégration sur une cellule $P$ donne :

$$
\frac{d}{dt}(\rho_P\phi_PV_P)
+\sum_f F_f\phi_f
=\sum_f\Gamma_f(\nabla\phi)_f\cdot\mathbf{S}_f
+S_{\phi,P}V_P.
$$

La méthode construit un système linéaire :

$$
 a_P\phi_P=\sum_Na_N\phi_N+b_P.
$$

Les flux de convection et diffusion sont évalués sur les faces. Les schémas `fvSchemes` déterminent l’interpolation et la discrétisation des divergences ; les solveurs, tolérances et contrôles de résidus sont définis dans `fvSolution`. OpenFOAM décrit cette approche comme une discrétisation volumes finis configurée par les dictionnaires numériques [4].

### 10.2 Temps et stabilité

Un schéma explicite est limité par une condition CFL :

$$
Co=\frac{U\Delta t}{\Delta x}\lesssim Co_{max}.
$$

Les schémas implicites autorisent des pas plus grands mais ne garantissent pas une précision suffisante. Pour les problèmes HAM, la diffusion et les non-linéarités de stockage imposent également une contrainte de résolution temporelle.

Le pas utilisé dans les exemples peut évoluer selon un contrôle automatique. Il est recommandé de surveiller simultanément `deltaT`, `Co`, les résidus et les variations physiques entre pas.

### 10.3 Couplage pression–vitesse

Une méthode de type SIMPLE/PIMPLE alterne :

1. résolution des équations de quantité de mouvement avec une pression provisoire ;
2. construction d’une équation de pression à partir du flux de vitesse ;
3. correction de pression et de flux pour satisfaire la continuité ;
4. correction des variables turbulentes, thermiques, HAM et radiatives ;
5. répétition jusqu’aux critères de résidu ou au nombre maximal d’itérations.

Une erreur de dimensions dans une source, un flux ou une pression apparaît souvent au moment de l’assemblage de l’équation de quantité de mouvement. Les dimensions OpenFOAM doivent donc être contrôlées avant toute interprétation physique.

## 11. Chaîne de calcul FoamPilot

La chaîne générée pour les six profils suit les étapes suivantes :

```text
Profil Python
    ↓
UrbanClimateNativeCaseBuilder
    ↓
0/ + constant/ + system/ générés
    ↓
blockMesh par région
    ↓
cellZones + finalAgglom générés
    ↓
faceAgglomerate
    ↓
calcLAI (profils végétation)
    ↓
viewFactorsGen
    ↓
solarRayTracingGen
    ↓
urbanMicroclimateFoam
```

Les profils disponibles sont `streetCanyon_CFD`, `streetCanyon_CFDHAM`, `streetCanyon_CFDHAM_grass`, `streetCanyon_CFDHAM_veg`, `windAroundBuildings_CFDHAM` et `windAroundBuildings_CFDHAM_veg`. La génération ne copie pas les répertoires `0`, `constant` ou `system` des tutoriels ; elle écrit les dictionnaires à partir des spécifications de région, des profils physiques et des ressources géométriques.

Les fichiers solaires `sunPosVector`, `IDN` et `Idif` sont des tables temporelles. Les champs `LAD`, `Tambient`, `wambient`, `qr`, `qs`, `qro` et `qso` sont générés lorsque la physique correspondante est activée. La séparation des `radiationProperties` par région est importante : les régions fluides et végétales ne portent pas nécessairement le même modèle radiatif.

## 12. Conditions limites recommandées

### 12.1 Entrée et sortie de l’air

Une entrée de vent peut utiliser :

$$
\mathbf{U}=\mathbf{U}_{in},
$$

avec température et humidité météorologiques imposées. La sortie utilise souvent une condition de gradient nul pour $\mathbf{U}$ et une pression de référence. Dans un calcul de canyon, imposer simultanément une pression et une vitesse incompatibles peut surcontraindre le problème.

### 12.2 Sol et bâtiments

Pour une paroi solide, la température et le flux doivent être couplés à la région solide. Une condition `fixedValue` sur la température est acceptable pour un test isolé, mais elle supprime le stockage HAM et empêche d’étudier le déphasage jour–nuit.

### 12.3 Interfaces végétation–air

Les interfaces de végétation doivent être identifiées comme patches voisins cohérents. Les champs radiatifs spécialisés doivent être utilisés lorsque les modèles `viewFactorSky` et `directAndDiffuse` attendent un type de condition limite dérivé particulier. Les valeurs d’émissivité et d’albédo doivent être documentées et différenciées par matériau.

## 13. Paramètres adimensionnels et contrôle de régime

Les nombres utiles sont :

$$
Re=\frac{UL}{\nu},
\qquad
Pe=\frac{UL}{\alpha},
\qquad
Pr=\frac{\nu}{\alpha},
$$

$$
Ra=\frac{g\beta_T\Delta TL^3}{\nu\alpha},
\qquad
Ri=\frac{Gr}{Re^2},
$$

et, pour l’humidité, un nombre de Lewis :

$$
Le=\frac{\alpha}{D_v}.
$$

$Re$ mesure l’importance de l’inertie par rapport à la viscosité, $Pe$ celle de la convection par rapport à la diffusion thermique, $Ra$ le potentiel de convection naturelle et $Le$ le rapport entre diffusion thermique et diffusion de vapeur. Ces nombres permettent de comparer les cas et de détecter une configuration non représentative, par exemple une vitesse d’entrée trop faible combinée à un pas de temps trop grand.

## 14. Vérification, validation et incertitudes

### 14.1 Vérification numérique

La vérification répond à la question : « le modèle est-il résolu correctement ? » Elle comprend :

| Contrôle | Objectif |
|---|---|
| `checkMesh` | Orthogonalité, non-orthogonalité, skewness, volumes et patches |
| Bilan de masse | Vérifier les flux entrants et sortants |
| Résidus | Vérifier la réduction des erreurs algébriques |
| Indépendance de maillage | Mesurer la sensibilité à la taille de cellule |
| Indépendance temporelle | Mesurer la sensibilité à `deltaT` |
| Conservation énergétique | Comparer apports, stockage, convection, latent et rayonnement |
| Bornes physiques | Vérifier $0\leq\phi\leq1$, $0\leq\alpha\leq1$, $0\leq F_{ij}\leq1$ |

### 14.2 Validation physique

La validation répond à la question : « le modèle représente-t-il le système réel ? » Elle demande des mesures synchronisées : vitesse et direction du vent, température et humidité à plusieurs hauteurs, températures de surface, rayonnement incident et flux de chaleur. Les indicateurs usuels sont :

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^N(y_i^{sim}-y_i^{obs})^2},
$$

$$
MAE=\frac{1}{N}\sum_{i=1}^N|y_i^{sim}-y_i^{obs}|,
$$

$$
MBE=\frac{1}{N}\sum_{i=1}^N(y_i^{sim}-y_i^{obs}).
$$

Pour les températures, on rapporte également le biais horaire et l’erreur maximale. Pour le vent, il faut évaluer séparément la norme et la direction. Pour l’humidité, l’erreur en humidité relative doit être accompagnée de l’erreur en pression de vapeur ou humidité absolue, car l’humidité relative est fortement dépendante de la température.

### 14.3 Incertitudes dominantes

Les principales incertitudes sont la turbulence, la résolution géométrique, la rugosité des surfaces, l’émissivité, l’albédo, les conditions météorologiques, le $LAD$, la résistance stomatique, les propriétés hygrothermiques des matériaux et la qualité des données solaires. Une étude robuste doit distinguer les paramètres calibrés des paramètres mesurés et effectuer une analyse de sensibilité.

## 15. Conseils de configuration et diagnostic

Un calcul qui diverge après l’activation de la végétation doit être diagnostiqué dans cet ordre : dimensions de `LAD` et de `Cf`, valeurs initiales de $T_l$, bornes de l’humidité relative, cohérence des patches mappés, présence de `qr`, `qs`, `qro` et `qso`, validité de `finalAgglom`, puis choix du pas de temps. Un calcul qui termine mais produit des températures irréalistes doit être examiné du point de vue des bilans : rayonnement net, flux latent, flux sensible et stockage.

Une erreur de type « attempt to cast » indique généralement qu’un champ porte une condition limite générique alors que le modèle attend une classe spécialisée. Une erreur de dimension indique une incohérence entre la variable et son terme source. Une erreur de mapping ou une exception flottante dans `mappedPatchBase` indique souvent des patches voisins incompatibles, des faces non correspondantes ou une géométrie dégénérée.

Pour une nouvelle ville, il est préférable de procéder par niveaux : d’abord écoulement isotherme, puis température de l’air, puis solides HAM, puis rayonnement, enfin végétation. Cette progression permet d’attribuer chaque écart à une famille de modèles.

## 16. Limites du modèle présent

Le modèle végétalise la région par une fermeture volumique et une énergie foliaire simplifiée ; il ne résout ni chaque feuille, ni la photosynthèse, ni les échanges stomatiques détaillés d’une espèce particulière. Le modèle HAM dépend des lois de matériau et de leurs paramètres ; il ne remplace pas une caractérisation expérimentale de la paroi. Le rayonnement par facteurs de vue suppose une géométrie et des propriétés optiques suffisamment bien définies. La RANS stationnaire ou quasi-stationnaire ne représente pas toutes les structures turbulentes transitoires d’un canyon réel.

Ces limites ne diminuent pas l’intérêt du solveur : elles définissent le domaine d’emploi. La méthode est particulièrement adaptée à l’étude comparative de variantes urbaines — orientation, hauteur, végétation, albédo, matériaux — lorsque les mêmes hypothèses et les mêmes conditions météorologiques sont conservées entre scénarios.

## 17. Références

[1]: https://github.com/OpenFOAM-BuildingPhysics/urbanMicroclimateFoam "OpenFOAM-BuildingPhysics — urbanMicroclimateFoam"

[2]: https://doi.org/10.1016/j.rser.2017.05.248 "Toparlar et al., A review on the CFD analysis of urban microclimate, Renewable and Sustainable Energy Reviews, 2017"

[3]: https://doi.org/10.1016/j.uclim.2021.100939 "Mughal et al., Detailed investigation of vegetation effects on microclimate by means of CFD, Urban Climate, 2021"

[4]: https://doc.cfd.direct/openfoam/user-guide-v13/models "OpenFOAM Foundation v13 User Guide — Models and physical properties"

[5]: https://www.mdpi.com/1996-1073/13/6/1414 "Tsoka et al., Urban Warming and Cities’ Microclimates: Investigation Methods and Mitigation Strategies, Energies, 2020"
