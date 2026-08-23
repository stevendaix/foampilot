#import "report-theme.typ": report-accent, report-theme

#show: report-theme.with(
  title: "JOS-3 et couplage thermo-aéraulique avec OpenFOAM",
  author: "Manus AI",
  rhythm: "report",
  running-header: true,
)

#page(margin: (top: 28%, x: 2.2cm), numbering: none, header: none)[
  #set par(first-line-indent: 0em)
  #align(center)[
    #text(size: 25pt, weight: "bold", fill: report-accent)[JOS-3 et couplage thermo-aéraulique avec OpenFOAM]
    #v(0.7em)
    #text(size: 15pt)[Support de cours détaillé pour FoamPilot]
    #v(1.5em)
    #line(length: 42%, stroke: 0.8pt + report-accent)
    #v(1.5em)
    #text(size: 11pt)[Architecture, équations, unités, algorithmes, échanges face par face et validation]
    #v(3em)
    #text(size: 11pt)[Manus AI — 21 août 2026]
  ]
]

#page(numbering: none, header: none)[
  #outline(title: [Sommaire], indent: 1.5em)
]
#counter(page).update(1)

= Objet et objectifs pédagogiques

Ce document présente JOS-3, le modèle numérique de thermorégulation humaine développé à partir de JOS-2 et du modèle multi-nœuds 65MN, puis son intégration dans FoamPilot pour échanger des grandeurs thermiques avec OpenFOAM. Il est conçu comme un support de cours : les équations sont écrites explicitement, les hypothèses sont signalées, les unités sont rappelées et chaque transfert de données est décrit au niveau du fichier et de la face de maillage.

À l’issue de la lecture, on doit pouvoir distinguer un nœud physiologique d’une face CFD, construire une matrice thermique, expliquer le rôle du débit sanguin, calculer les pertes convectives, radiatives et évaporatives, écrire le schéma de Backward Euler, préparer un mapping des 17 zones et diagnostiquer un échange `externalCoupledTemperature`.

#block(stroke: 2pt + report-accent, inset: 8pt)[*Idée centrale.* OpenFOAM résout l’écoulement et la température de l’air autour du corps. JOS-3 résout les réponses internes : températures centrales et cutanées, débit sanguin, sudation, frisson, métabolisme et pertes respiratoires. Le couplage relie ces deux descriptions par des surfaces et des coefficients d’échange.]

= Contexte scientifique et périmètre

JOS-3 est un modèle multi-nœuds : le corps est découpé en 17 parties locales et chaque partie contient plusieurs compartiments thermiques. Le modèle publié comporte 83 nœuds et utilise une intégration temporelle par différence arrière, c’est-à-dire une forme de Backward Euler [1]. Il a été conçu pour les environnements stables, non uniformes et transitoires, avec des extensions pour l’âge, le tissu adipeux brun et le rayonnement solaire à la peau.

Dans FoamPilot, l’objectif n’est pas de remplacer JOS-3 par une température moyenne. Les faces CFD restent individualisées. Le mapping agrège leurs surfaces dans les 17 zones physiologiques pour l’appel au modèle, puis redistribue une température cutanée de zone vers les faces correspondantes. Cette distinction est essentielle pour conserver l’information spatiale du maillage tout en respectant les degrés de liberté physiologiques disponibles.

#table(
  columns: (2.2cm, 3.2cm, 8cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Objet*], [*Échelle*], [*Rôle*],
  [Face CFD], [millimètre à mètre], [Surface géométrique, température locale, flux et HTC],
  [Zone JOS-3], [17 parties], [Agrégation physiologique et réponse cutanée locale],
  [Nœud thermique], [83 nœuds], [Stockage de chaleur, conduction et transport sanguin],
  [Corps entier], [1 système], [Métabolisme, respiration, confort et bilans globaux],
)

= Architecture du modèle JOS-3

== Les 17 zones anatomiques

L’ordre canonique utilisé par JOS-3 est : `Head`, `Neck`, `Chest`, `Back`, `Pelvis`, `LShoulder`, `LArm`, `LHand`, `RShoulder`, `RArm`, `RHand`, `LThigh`, `LLeg`, `LFoot`, `RThigh`, `RLeg`, `RFoot`. Les tableaux de température, de surface, de vitesse d’air, d’habillement et de flux doivent respecter cet ordre lorsqu’ils sont fournis sous forme de vecteur de longueur 17.

La géométrie MakeHuman n’enseigne pas automatiquement l’anatomie à JOS-3. C’est le mapping qui détermine la zone : pour chaque face `f`, FoamPilot stocke un identifiant `z_f`, une aire `A_f` et le centre géométrique `x_f`. La température moyenne de zone peut être définie par une moyenne surfacique :

#align(center)[$ T_z = frac(sum_(f in z) A_f T_f, sum_(f in z) A_f) .$]

De manière analogue, un coefficient d’échange de zone est une moyenne pondérée :

#align(center)[$ h_z = frac(sum_(f in z) A_f h_f, sum_(f in z) A_f) .$]

Le retour vers le CFD est ensuite une projection constante par zone, ou une interpolation plus fine si un modèle local est ajouté :

#align(center)[$ T_f^"return" = T_(z_f), quad f in z_f .$]

== Les 83 nœuds thermiques

La décomposition exacte dépend de la partie anatomique et de la présence de muscle ou de graisse. Les familles de nœuds sont le cœur, le muscle, la graisse et la peau, auxquelles s’ajoutent les réseaux sanguins artériel et veineux et les nœuds auxiliaires nécessaires aux échanges. Chaque nœud possède une température, une capacité thermique et des conductances vers d’autres nœuds.

#table(
  columns: (3.2cm, 3.1cm, 7.1cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Grandeur*], [*Unité*], [*Interprétation*],
  [$T_i$], [°C ou K], [Température du nœud thermique $i$],
  [$C_i$], [Wh/K], [Capacité thermique du nœud],
  [$G_(i j)$], [W/K], [Conductance entre deux nœuds],
  [$Q_i$], [W], [Production métabolique locale],
  [$H_i$], [W], [Perte ou gain thermique local],
  [$B F_i$], [L/h], [Débit sanguin local],
  [$A_z$], [m²], [Surface physique associée à une zone],
)

== Construction corporelle et surface

La surface corporelle totale peut utiliser l’équation de DuBois :

#align(center)[$ B S A = 0.202 h^0.725 m^0.425 $,]

avec $h$ en mètres, $m$ en kilogrammes et $B S A$ en mètres carrés. Pour la configuration de référence FoamPilot, la surface CFD body-only est conservée face par face ; la somme des aires du mapping doit satisfaire :

#align(center)[$ A_"CFD" = sum_(f=1)^N A_f = sum_(z=1)^17 A_z .$]

Cette identité est le premier contrôle de conservation géométrique. Une erreur de facteur 100, 1 000 ou 10 000 entre millimètres et mètres contamine directement le flux, le métabolisme surfacique et les températures.

= Équation thermique fondamentale

== Bilan différentiel d’un nœud

Pour un nœud $i$, le bilan général s’écrit :

#align(center)[$ C_i frac(d T_i, d t) = sum_(j in cal(N)_i) G_(i j)(T_j - T_i) + Q_i - H_i + S_i . $]

$cal(N)_i$ est l’ensemble des nœuds connectés à $i$. $Q_i$ rassemble les productions de chaleur, $H_i$ les pertes vers l’environnement et $S_i$ les sources ou termes externes additionnels. Dans les nœuds traversés par le sang, le terme de transport peut être représenté par une conductance de perfusion :

#align(center)[$ Q_("blood", i) = G_("blood", i)(T_("blood") - T_i), quad G_("blood", i) = rho_b c_b dot(V)_i .$]

Dans cette écriture, $rho_b$ est la masse volumique du sang, $c_b$ sa capacité thermique massique et $dot(V)_i$ le débit volumique converti dans une unité cohérente. JOS-3 expose les débits en L/h, alors que la conductance exige une conversion en m³/s ou une constante de conversion documentée.

== Forme matricielle

En assemblant les 83 équations, on obtient une équation linéaire :

#align(center)[$ C dot(T) = A T + b, $]

avec $C$ matrice diagonale des capacités, $A$ matrice des conductances et $b$ vecteur des productions et pertes indépendantes de $T$. Les termes diagonaux de $A$ sont négatifs et les termes hors diagonale sont positifs lorsque la convention est celle d’un flux entrant depuis le voisin.

== Discrétisation Backward Euler

À l’instant $n+1$, le schéma implicite est :

#align(center)[$ C frac(T^(n+1) - T^n, Delta t) = A T^(n+1) + b^(n+1). $]

La matrice à résoudre devient :

#align(center)[$ (C/Delta t - A) T^(n+1) = C/Delta t T^n + b^(n+1). $]

Le schéma est plus robuste qu’Euler explicite pour les petites capacités thermiques et les grandes conductances, car la température inconnue apparaît implicitement. Il reste toutefois sensible à la cohérence des flux, des unités et du pas transmis au modèle. Dans FoamPilot, `dtime` doit être égal au `deltaT` physique d’OpenFOAM pour éviter d’avancer la physiologie plus vite que le CFD.

== Stockage et température

Pour un nœud homogène :

#align(center)[$ C_i = m_i c_i = rho_i V_i c_i .$]

La conversion entre degrés Celsius et kelvins ne change pas une différence de température, mais elle change la valeur absolue utilisée dans les lois de pression de vapeur et dans certaines corrélations radiatives. La règle de couplage est donc : températures thermodynamiques en K dans les fichiers OpenFOAM, températures physiologiques généralement manipulées en °C dans l’API JOS-3, conversion explicite à la frontière.

= Échanges de chaleur avec l’environnement

== Convection

La perte convective d’une surface $z$ s’écrit :

#align(center)[$ Q_("conv", z) = h_(c, z) A_z (T_("sk", z) - T_("a", z)). $]

Pour une face CFD :

#align(center)[$ q_("conv", f) = h_f (T_(f) - T_("a", f)), quad Q_("conv", f) = q_("conv", f) A_f .$]

JOS-3 dispose de corrélations locales. La contribution forcée est représentée dans le code par :

#align(center)[$ h_(c, f) = 12.1 sqrt(v_a), $]

avec $v_a$ en m/s et $h_c$ en W/(m² K). La convection naturelle utilise une loi de type :

#align(center)[$ h_(c, n) = 2.38 abs(T_"sk" - T_"a")^0.25 .$]

Le coefficient retenu combine les contributions selon l’implémentation et la posture. En mode CFD, le coefficient calculé par OpenFOAM peut remplacer cette corrélation, à condition de définir sans ambiguïté s’il s’agit d’un HTC local, d’une moyenne de patch ou d’un coefficient déjà multiplié par une aire.

== Rayonnement

Le flux radiatif exact entre une peau et un environnement gris peut s’écrire :

#align(center)[$ q_("rad") = epsilon sigma (T_"sk,K"^4 - T_"r,K"^4), $]

où $epsilon$ est l’émissivité, $sigma$ la constante de Stefan–Boltzmann et les températures sont en kelvins. Sous forme linéarisée :

#align(center)[$ q_("rad") approx h_r (T_"sk" - T_"r"), quad h_r = epsilon sigma (T_"sk,K" + T_"r,K")(T_"sk,K"^2 + T_"r,K"^2). $]

JOS-3 utilise des coefficients radiatifs dépendant de la posture et de la surface. Dans le couplage actuel, le CFD fournit principalement la convection ; $T_"r"$ et $h_r$ doivent être imposés ou calculés séparément si le rayonnement est activé.

== Température opérative

Lorsque l’air et le rayonnement sont combinés par un modèle linéarisé :

#align(center)[$ T_o = frac(h_c T_"a" + h_r T_"r", h_c + h_r). $]

Cette température ne doit être utilisée que lorsque les coefficients et les hypothèses le permettent. Elle ne remplace pas un champ CFD non uniforme face par face.

== Résistances sèche et humide

La résistance sèche totale peut être organisée comme :

#align(center)[$ R_t = R_c + R_r + R_"cl", $]

avec les résistances convective, radiative et vestimentaire. Pour l’habillement :

#align(center)[$ R_"cl" = 0.155 I_"cl", $]

où $I_"cl"$ est exprimé en clo et $R_"cl"$ en m² K/W. La résistance humide est liée au transfert de vapeur et au rapport de Lewis ; son exactitude dépend fortement de l’humidité relative et de la définition de la surface mouillée.

= Évaporation, transpiration et respiration

== Pression de vapeur

La pression de vapeur saturante de la peau $p_("sk", s)$ dépend de $T_"sk"$. La pression partielle de l’air $p_a$ est calculée à partir de l’humidité relative $R H$ :

#align(center)[$ p_a = R H / 100 dot p_("sat")(T_"a"). $]

JOS-3 implémente des corrélations de type Antoine ou Tetens. Le déficit de vapeur est :

#align(center)[$ Delta p = p_("sk", s) - p_a .$]

== Perte évaporative maximale

Le maximum évaporatif d’une zone s’écrit dans la structure du modèle :

#align(center)[$ E_("max", z) = frac((p_("sk", s) - p_a) B S A_z, R_("et", z)), $]

avec $R_"et"$ en Pa m²/W si la pression est exprimée en Pa, ou avec une constante de conversion explicite lorsque la pression est en kPa. Cette compatibilité dimensionnelle doit être vérifiée avant toute comparaison.

La sudation régulatrice fournit une fraction effective du potentiel :

#align(center)[$ E_("sweat", z) = f_("sweat", z) E_("max", z), quad 0 <= f_("sweat", z) <= 1. $]

La perte totale cutanée inclut une part insensible et la sudation :

#align(center)[$ E_("sk", z) = E_("insensible", z) + E_("sweat", z). $]

== Respiration

La respiration comporte une part sensible et une part latente :

#align(center)[$ Q_("resp") = Q_("resp", "sensible") + Q_("resp", "latent"). $]

Le modèle dépend de la température, de la pression et du niveau métabolique. La respiration est globale mais peut être distribuée dans les résultats selon la structure de sortie.

= Régulation physiologique

== Signaux d’erreur

JOS-3 compare les températures centrales et cutanées à des températures de référence. On peut écrire les erreurs régulatrices sous la forme :

#align(center)[$ e_"cr" = T_"cr" - T_("cr", "set"), quad e_"sk" = T_"sk" - T_("sk", "set"). $]

Les signaux sont ensuite pondérés par zone et séparés en composantes chaude et froide :

#align(center)[$ e_"cr"^+ = max(e_"cr",0), quad e_"cr"^- = max(-e_"cr",0), $]

#align(center)[$ e_"sk"^+ = max(e_"sk",0), quad e_"sk"^- = max(-e_"sk",0). $]

Ces signaux pilotent la vasodilatation, la vasoconstriction, la sudation, le frisson, la thermogenèse sans frisson et les débits des anastomoses artério-veineuses.

== Débit sanguin cutané

Le débit cutané augmente lorsque le corps est chaud et diminue lorsqu’il est froid. Une forme générale, utile pour lire le code, est :

#align(center)[$ B F_"sk" = B F_("base", "sk") + k_("warm") e_"sk"^+ + k_("cold") e_"sk"^- .$]

Le modèle impose des bornes physiologiques et distribue ensuite le débit par région. Les conductances sanguines transforment ce débit en transfert thermique entre le sang et les tissus.

== Anastomoses des mains et des pieds

Les mains et les pieds disposent de débits AVA spécifiques. Les signaux AVA sont des combinaisons linéaires des erreurs centrale et cutanée. Dans la version JOS-3 documentée par le dépôt officiel, la forme corrigée est :

#align(center)[$ sigma_("AVA", "hand") = 0.265(e_"msk" + 0.43) + 0.953(e_"bcr" + 0.1905) + 0.9126, $]

#align(center)[$ sigma_("AVA", "foot") = 0.265(e_"msk" - 0.997) + 0.953(e_"bcr" + 0.0095) + 0.9126. $]

Les coefficients et la définition des erreurs doivent être lus avec la version précise du code. Une modification de l’ordre `e_"msk"`/`e_"bcr"` change fortement la réponse au froid.

== Métabolisme basal et activité

Le métabolisme total est la somme de plusieurs contributions :

#align(center)[$ M = M_"base" + M_"work" + M_"shiv" + M_"nst". $]

Le métabolisme basal peut être exprimé en fonction de la taille, du poids, de l’âge et du sexe selon l’équation choisie. Le ratio d’activité physique est :

#align(center)[$ P A R = M / M_"base" .$]

Ainsi, pour une production surfacique visée $M_A$ :

#align(center)[$ M = M_A B S A .$]

Le modèle distribue ensuite la puissance par zone, puis par couche. La thermogenèse sans frisson est associée notamment au tissu adipeux brun et aux caractéristiques individuelles ; le frisson est activé sous un seuil froid et possède une dynamique avec mémoire du pas précédent.

== Frisson

Le frisson est une production musculaire ou centrale déclenchée par le froid. Une représentation pédagogique est une loi à seuil :

#align(center)[$ M_"shiv" = 0 quad "si le signal froid est inactif"; quad M_"shiv" = k_"shiv" f(e_"cr", e_"sk") quad "si le signal froid est actif". $]

Le code ajoute des limites et des fonctions de saturation pour empêcher une production irréaliste. Le terme de frisson ne doit pas être confondu avec une perte de chaleur : c’est une source $Q_i$ qui augmente la température et le débit sanguin associé.

== Sudation

La sudation est activée par des signaux chauds centraux et cutanés. Une forme conceptuelle est :

#align(center)[$ S_z = max(0, k_("cr") e_"cr"^+ + k_("sk") e_"sk"^+), $]

puis :

#align(center)[$ E_("sweat", z) = min(S_z, E_("max", z)). $]

La saturation par $E_max$ est indispensable : sans elle, le flux latent demandé pourrait dépasser le potentiel d’évaporation permis par l’air ambiant.

= Température de confort et indicateurs

JOS-3 produit des sorties physiologiques, tandis que les indicateurs PMV/PPD peuvent être calculés séparément. Le PMV utilise notamment le métabolisme, l’habillement, la température d’air, la température moyenne radiante, la vitesse d’air et l’humidité. Il ne doit pas être confondu avec la température cutanée : une personne peut avoir une peau proche de 34 °C tout en ressentant un environnement froid ou chaud selon le bilan complet.

#table(
  columns: (3.5cm, 3cm, 6.8cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Sortie*], [*Unité*], [*Sens*],
  [`Tsk`], [°C], [Température cutanée locale],
  [`Tcr`], [°C], [Température centrale locale],
  [`TskMean`], [°C], [Moyenne cutanée globale],
  [`Wet`], [-], [Mouillure locale de la peau],
  [`BFsk`, `BFcr`], [L/h], [Débits sanguins cutané et central],
  [`Met`], [W], [Production thermique totale],
  [`THLsk`], [W], [Pertes thermiques cutanées],
  [`RES`], [W], [Pertes respiratoires],
  [`BSA`], [m²], [Surface corporelle locale],
)

= Couplage avec OpenFOAM

== Principe bidirectionnel

À chaque pas, le CFD calcule une condition thermique sur le patch `human`. Le protocole transmet, face par face, l’aire, la température, le flux ou gradient et le HTC. FoamPilot utilise les aires et le mapping pour agréger les entrées dans les 17 zones, avance JOS-3, puis projette la température cutanée calculée vers les faces.

Le graphe logique est :

#align(center)[
  #box(stroke: 0.7pt + report-accent, inset: 8pt)[OpenFOAM : $T_f$, $h_f$, $A_f$]
  #h(1em) $arrow$ #h(1em)
  #box(stroke: 0.7pt + report-accent, inset: 8pt)[Mapping : $z_f$, $A_z$]
  #h(1em) $arrow$ #h(1em)
  #box(stroke: 0.7pt + report-accent, inset: 8pt)[JOS-3 : $T_"sk"$, régulation]
]

Puis le retour :

#align(center)[$ T_("surface", f)^(n+1) = cal(P)(T_("sk", z)^(n+1)), $]

où $cal("P")$ est l’opérateur de projection du vecteur de zones vers les faces CFD.

== Fichiers du protocole externalCoupled

OpenFOAM utilise un verrou pour synchroniser les deux programmes. Le cycle est :

+ OpenFOAM écrit `data.out` et retire le verrou `OpenFOAM.lock`.
+ Le pilote attend un fichier complet et lit exactement le nombre attendu de faces.
+ FoamPilot convertit les unités, agrège les faces et avance JOS-3.
+ Le pilote écrit `data.in`, puis recrée le verrou.
+ OpenFOAM lit `data.in`, applique la condition mixte et reprend son calcul.

La documentation OpenFOAM décrit une ligne par face et des fichiers `*.in`/`*.out` dans un répertoire de communication [3]. Le fichier de géométrie contient les faces et points collectés par OpenFOAM, généralement générés par `createExternalCoupledPatchGeometry`.

== Format des données

Dans l’exemple FoamPilot, les données CFD sont structurées ainsi :

#raw(block: true, lang: "text", "# Patch: human\narea[m2]  T[K]  qDot[W/m2]  htc[W/m2/K]\n0.000142 307.15 12.4 4.8\n...")

Le retour thermique contient les champs nécessaires à une condition mixte :

#raw(block: true, lang: "text", "# Patch: human\nT_surface[K]  snGrad  valueFraction\n307.20 0 1\n...")

Les noms d’unités doivent être dans l’en-tête documentaire et les températures absolues rester en kelvins dans l’échange OpenFOAM. Les valeurs de flux doivent indiquer s’il s’agit d’un flux surfacique en W/m² ou d’une puissance intégrée en W.

== Synchronisation temporelle

Si OpenFOAM avance de `deltaT` et JOS-3 de `dtime`, la cohérence exige :

#align(center)[$ Delta t_"JOS" = Delta t_"CFD" .$]

Dans le pilote, `deltaT` est lu depuis `system/controlDict`. Une ancienne version avançait JOS-3 d’une seconde par échange alors que le CFD avançait de 0,05 s ; cette erreur produisait une accélération physiologique artificielle. Le pilote corrigé transmet le vrai `dtime`.

== Sous-relaxation du retour

Pour limiter les oscillations, la température de retour peut être sous-relaxée :

#align(center)[$ T_"return"^(n+1) = (1-alpha) T_"return"^n + alpha T_"JOS"^(n+1), quad 0 < alpha <= 1. $]

Avec $alpha=0.1$, 90 % de la valeur précédente sont conservés. Cette stabilisation numérique ne doit pas masquer une erreur d’unité ou de signe ; elle doit être accompagnée d’un suivi des flux et des résidus.

= Mapping MakeHuman et contrôle des surfaces

== Pipeline géométrique

Le pipeline recommandé est : export MakeHuman vers `base.npz`, sélection du groupe `body`, conversion meshio vers STL/OBJ/VTK, contrôle topologique, maillage `snappyHexMesh`, extraction des faces réelles du patch `human`, puis classification des centres de faces dans les 17 zones.

Le fichier source MakeHuman peut contenir de nombreux groupes auxiliaires. Si tous les groupes sont exportés, les yeux, dents, cheveux, joints et helpers créent des composantes déconnectées et des bords ouverts. Le maillage body-only retenu présente une seule composante et une surface fermée avant son insertion CFD.

== Conservation de l’aire

Pour chaque zone :

#align(center)[$ r_z = frac(A_z, sum_(k=1)^17 A_k), quad sum_(z=1)^17 r_z = 1 .$]

Pour chaque face :

#align(center)[$ A_f >= 0, quad A_f > 0 quad "pour toute face valide" .$]

Le contrôle doit comparer l’aire STL, l’aire du patch OpenFOAM et l’aire du CSV de mapping. Ces trois aires peuvent différer légèrement après snapping, mais l’écart doit être mesuré et expliqué.

== Classification des zones

La classification pratique utilise les coordonnées normalisées du centre de face : hauteur $z$, latéralité $x$ et profondeur $y$. Elle doit appliquer des règles exclusives dans un ordre stable. Les règles bras/jambes ne doivent pas se recouvrir ; sinon une zone comme `LArm` peut être écrasée par `LLeg`. La validation minimale consiste à produire un histogramme des 17 identifiants et à vérifier l’absence de zone vide.

= Algorithme complet d’un pas couplé

#raw(block: true, lang: "text", "1. OpenFOAM avance le champ d’air et la température.\n2. externalCoupledTemperature écrit une ligne par face dans data.out.\n3. FoamPilot attend la fin d’écriture et vérifie N_faces.\n4. Le pilote lit A_f, T_f, qDot_f et h_f.\n5. Il applique z_f et agrège les grandeurs par aire.\n6. JOS-3 avance de dtime = deltaT CFD.\n7. Il récupère Tsk_z, les flux et les sorties physiologiques.\n8. La température de zone est projetée sur chaque face.\n9. Le pilote écrit data.in et recrée le verrou.\n10. OpenFOAM applique le retour et écrit les résidus.")

La condition de terminaison d’un pas est la présence simultanée d’un fichier complet et d’un nombre de lignes égal à $N_f$. Une lecture concurrente d’un fichier partiellement écrit est une erreur de protocole ; le pilote doit attendre une taille stable ou un indicateur de fin avant le parsing.

= Bilans énergétiques et audits

== Bilan physiologique global

À chaque pas, on peut vérifier :

#align(center)[$ Q_("met") + Q_("blood", "net") + Q_("solar") - Q_("skin") - Q_("resp") approx sum_i C_i frac(T_i^(n+1)-T_i^n, Delta t). $]

Le résidu de bilan est :

#align(center)[$ R_E = Q_("in") - Q_("out") - Q_("stored"). $]

Un résidu qui augmente monotoniquement signale une incohérence de signe, d’unité, de surface ou de synchronisation.

== Bilan CFD sur la peau

Pour le patch humain :

#align(center)[$ Q_("CFD", "skin") = sum_f h_f A_f (T_("a", f) - T_("sk", f)). $]

Le signe dépend de la convention du flux OpenFOAM. Il faut l’écrire explicitement dans le rapport et comparer la puissance intégrée au flux reçu par JOS-3. Une température cutanée retournée en °C alors qu’OpenFOAM attend des K provoquerait une erreur de 273,15 K ; une aire en cm² utilisée comme m² provoquerait un facteur 10 000.

== Tests de non-régression

#table(
  columns: (4.3cm, 3.2cm, 5.8cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Test*], [*Entrée*], [*Critère*],
  [Référence OpenFOAM], [buoyantCavity, coolingSphere], [Convergence et champs physiques],
  [JOS-3 officiel], [Conditions uniformes], [Écart aux sorties publiées documenté],
  [Surface mapping], [17 zones], [Somme des ratios = 1],
  [Couplage fictif], [HTC et température contrôlés], [Nombre de faces constant, pas sans NaN],
  [Couplage transitoire], [deltaT variable], [JOS-3 avance au même temps physique],
  [Open boundary], [plafond ouvert], [Continuité et température bornées],
)

= Résultats de l’exemple FoamPilot

La configuration validée utilise une géométrie MakeHuman body-only, une hauteur d’environ 1,7 m après échelle, un domaine d’air cubique et une formulation Boussinesq avec gravité. Le plafond est l’unique ouverture ; les anciennes faces latérales ne sont pas utilisées comme entrées et sorties.

#table(
  columns: (5.1cm, 3.2cm, 5cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Indicateur*], [*Valeur*], [*Commentaire*],
  [Faces humaines], [9 418], [Faces du patch OpenFOAM réellement échangées],
  [Zones physiologiques], [17], [Ordre JOS-3 canonique],
  [Pas de couplage], [0,05 s], [Identique à `deltaT`],
  [Durée couplée observée], [≈ 29,2 s], [584 échanges complets],
  [Température finale retournée], [33,55–34,07 °C], [Plage face par face],
  [HTC observé], [1,351–13,44 W/m²/K], [Plage du pilote],
  [Surface body-only], [≈ 3,21 m²], [Somme des faces CFD],
)

La référence JOS-3 seule sur 29,2 s donne une température cutanée moyenne passant d’environ 34,38 à 34,31 °C. Cette comparaison est un contrôle de cohérence et non encore une validation stricte, car le champ HTC y est imposé ou figé. Le benchmark strict doit enregistrer les séries temporelles `h`, `Ta`, `Tr`, températures et puissances par zone, puis les rejouer à JOS-3 seul.

= Procédure pratique de reproduction

== Installation

```bash
source /opt/openfoam13/etc/bashrc
cd examples/thermoregulation/makehuman
bash install_makehuman_ubuntu.sh
python3 -m pip install --user numpy meshio trimesh pyvista
```

Le dépôt JOS-3 et le code FoamPilot doivent être accessibles dans l’environnement Python. Les chemins sont configurables dans le pilote.

== Export et conversion

```bash
python3 export_makehuman_socket.py --out output
python3 convert_makehuman_meshio.py \
  --input output/base.npz \
  --output output/makehuman_body_meshio \
  --group body
python3 audit_makehuman_source.py
python3 audit_makehuman_topology.py
```

== Maillage et mapping

```bash
cd openfoam_cube_case
python3 ../create_openfoam_cube_case.py
./Allrun
```

`Allrun` prépare les champs, lance `blockMesh`, `snappyHexMesh`, crée la géométrie du protocole, génère le mapping et exécute `checkMesh`. Les sorties locales ne doivent pas être commitées dans le dépôt.

== Couplage

```bash
source /opt/openfoam13/etc/bashrc
python3 ../../openfoam_jos3_coupling/openfoam13_jos3_driver.py "$PWD" > jos3.log 2>&1 &
foamRun -solver fluid > openfoam.log 2>&1
```

== Référence JOS-3 seul et visualisation

```bash
python3 ../../openfoam_jos3_coupling/run_jos3_only_comparison.py \
  "$PWD" 584 0.05
python3 ../../validation/plot_openfoam_with_foampilot.py "$PWD"
```

FoamPilot/PyVista doit être utilisé pour cartographier `T`, `U`, `p_rgh`, les coefficients d’échange et les flux sur le patch humain. La visualisation est une étape de contrôle : une simulation qui converge numériquement peut encore présenter une température ou un flux non physique.

= Diagnostic des erreurs fréquentes

#table(
  columns: (4.1cm, 4.2cm, 5.2cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Symptôme*], [*Cause probable*], [*Contrôle*],
  [Face count mismatch], [Lecture partielle ou mauvais patch], [Comparer `patchFaces`, CSV et `data.out`],
  [Température ≈ 273 K trop basse], [°C interprétés comme K ou inversement], [Tracer l’en-tête et les conversions],
  [Flux 10 000 fois trop grand], [cm² utilisés comme m²], [Vérifier $A_f$ et l’intégrale],
  [JOS-3 avance trop vite], [dtime différent de deltaT], [Comparer les horodatages des deux journaux],
  [OpenFOAM diverge dès le retour], [Signe ou unité du retour], [Coupler une température constante contrôlée],
  [Zone physiologique vide], [Règles de classification recouvrantes], [Histogramme des 17 zones],
  [Fichier data.out incohérent], [Lecture pendant l’écriture], [Attendre une taille stable et le verrou],
)

= Exercices de cours

*Exercice 1 — unités.* À partir d’une face de 0,00015 m², d’un HTC de 5 W/m²/K et d’un écart de 10 K, calculer la puissance convective. Refaire le calcul si l’aire est accidentellement traitée en cm².

*Exercice 2 — intégration.* Pour un nœud de capacité 20 Wh/K, une conductance de 4 W/K, une source de 50 W et un pas de 10 s, écrire la ligne correspondante de la matrice Backward Euler.

*Exercice 3 — mapping.* Construire un tableau de 100 faces, classer leurs centres en 17 zones et démontrer que la moyenne surfacique est différente de la moyenne arithmétique lorsque les aires sont hétérogènes.

*Exercice 4 — couplage.* Remplacer temporairement JOS-3 par un coupler qui renvoie 307,15 K à toutes les faces. Comparer la dérive CFD avec le cas température fixe et identifier les effets qui viennent réellement de la physiologie.

*Exercice 5 — audit.* Calculer séparément le flux convectif OpenFOAM, la puissance reçue par JOS-3 et le stockage thermique. Construire un graphe du résidu $R_E$ en fonction du temps.

= Synthèse des hypothèses et limites

JOS-3 est un modèle physiologique global et local, mais il ne résout pas directement la géométrie 3D. Le CFD fournit la physique de l’air et de la surface ; le mapping fournit le lien anatomique. Une grande résolution CFD n’ajoute pas automatiquement de nouveaux degrés de liberté physiologiques : elle enrichit le champ de surface et la distribution des HTC.

Le modèle actuel est particulièrement adapté à l’étude de la réponse transitoire et non uniforme sous hypothèses documentées. Il faut encore ajouter une validation expérimentale du cas humain, une description contrôlée de l’humidité et du rayonnement, une vérification indépendante des propriétés de l’air et une comparaison stricte par zone avec les mêmes séries d’entrée dans JOS-3 seul.

La stabilité numérique ne prouve pas la justesse physique. Une simulation peut terminer avec des résidus faibles tout en conservant une erreur de signe, une surface mal mise à l’échelle ou une zone anatomique mal classée. Le support recommande donc toujours trois niveaux de contrôle : topologie et aire, unités et équations, puis bilan énergétique et comparaison physiologique.

= Références

[1] Y. Takahashi et al., *Thermoregulation Model JOS-3 with New Open Source Code*, Energy & Buildings 231, 110575, 2021. #link("https://doi.org/10.1016/j.enbuild.2020.110575")[Article et DOI].

[2] Tanabe Laboratory, *JOS-3 — Joint System Thermoregulation Model*. #link("https://github.com/TanabeLab/JOS-3")[Dépôt officiel et documentation].

[3] OpenFOAM Documentation, *externalCoupled function object*. #link("https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/externalCoupled/")[Documentation du protocole fichier].

[4] OpenFOAM Foundation, *OpenFOAM v13*. #link("https://openfoam.org/version/13/")[Version du solveur utilisée dans l’exemple].

[5] FoamPilot, *Couplage thermorégulation MakeHuman–JOS-3–OpenFOAM*. #link("https://github.com/stevendaix/foampilot/tree/feature/makehuman-jos3-openfoam-coupling/examples/thermoregulation")[Scripts et cas reproductibles].

[6] R. F. et al., *Development of a 65MN Multi-node Human Thermal Model*. #link("https://doi.org/10.1016/S0378-7788(02)00014-2")[Origine du modèle multi-nœuds].

= Annexe — conventions de notation

#table(
  columns: (2.5cm, 2.5cm, 8.5cm),
  inset: 5pt,
  fill: (_, row) => if row == 0 { report-accent.lighten(70%) } else { none },
  [*Symbole*], [*Unité SI*], [*Définition*],
  [$T$], [K], [Température absolue ; les différences peuvent être en K ou °C],
  [$t$], [s], [Temps physique],
  [$C$], [J/K ou Wh/K], [Capacité thermique],
  [$G$], [W/K], [Conductance thermique],
  [$h_c$], [W/m²/K], [Coefficient convectif],
  [$h_r$], [W/m²/K], [Coefficient radiatif linéarisé],
  [$q$], [W/m²], [Flux surfacique],
  [$Q$], [W], [Puissance intégrée],
  [$A$], [m²], [Aire],
  [BF], [L/h], [Débit sanguin exposé par JOS-3],
  [$R H$], [%], [Humidité relative],
  [$I_"cl"$], [clo], [Isolation vestimentaire],
)

#align(center)[
  #text(size: 9pt, fill: luma(90))[Fin du support — toujours citer la version du code, les paramètres individuels et les conventions de signe.]
]
