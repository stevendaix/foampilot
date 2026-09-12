
#set document(title: "Transition VOF vers DPM : revue théorique et audit OpenFOAM 13", author: "Manus AI — foampilot")
#set page(paper: "a4", margin: 2.5cm, numbering: "1 / 1")
#set text(font: "New Computer Modern", size: 11pt, lang: "fr")
#set heading(numbering: "1.1.")
#set par(justify: true)
#show figure.caption: it => [
  #text(weight: "bold", size: 0.9em)[#it.supplement #it.counter.display():] #it.body
]


= Résumé exécutif
Cette note examine la transition d’une description eulérienne VOF vers une description lagrangienne DPM/LPT dans le contexte de l’atomisation. Elle sépare la détection d’un fragment, la décision de transition, la création d’un parcel et la consommation du liquide VOF. L’audit conclut que le code actuel calcule correctement les propriétés intégrales des fragments en mode offline et valide le pont solver–cloud OpenFOAM 13, mais ne réalise pas encore l’insertion dynamique conservative dans un fvModel vivant. Cette distinction est indispensable pour ne pas confondre un prototype d’extraction avec une méthode de transition complète.

= Problème physique et motivation
La méthode VOF est adaptée aux nappes, ligaments et interfaces résolues, tandis que le DPM est adapté aux gouttes ponctuelles et au spray dilué. La littérature hybride exploite cette complémentarité pour couvrir plusieurs échelles de l’atomisation ［@heinrich2020］. Le coût du VOF augmente lorsque les petites gouttes doivent être résolues par le maillage ; le DPM réduit ce coût mais perd l’interface géométrique. La transition doit donc être déclenchée dans une zone où la représentation ponctuelle est physiquement et numériquement acceptable.

#align(center)[$ V = sum_(i in F) alpha dot V $] <eq_volume>
#align(center)[#text(size: 9pt)[Volume liquide physique d’une composante VOF]]

#align(center)[$ x = sum_(i in F) alpha dot V dot x / V, quad U = sum_(i in F) alpha dot V dot U / V $] <eq_moments>
#align(center)[#text(size: 9pt)[Centre et vitesse moyennés par le volume liquide]]

= Méthodes VOF et DPM
La méthode VOF introduit une fraction volumique alpha et résout son transport avec une reconstruction ou une compression d’interface ［@hirt1981］. Une cellule mixte n’est pas une goutte entière : son volume liquide vaut alpha V_cell. Le DPM représente chaque parcel par une position, une vitesse, une masse, un diamètre équivalent et des lois de force. Une équation générique s’écrit m_p d U_p/d t = F_drag + F_pressure + F_virtual-mass + m_p g + dots. La réaction du DPM apparaît comme une source dans l’équation eulérienne de quantité de mouvement.

= Les quatre étapes d’une transition correcte
La première étape est la détection : construire des composantes connectées à partir d’un masque alpha. La deuxième est la décision : appliquer des critères de diamètre, résolution locale, forme, distance à l’interface et persistance temporelle. La troisième est la création : calculer masse, position, vitesse et éventuellement température du parcel. La quatrième est la consommation : retirer le volume converti du champ VOF et transférer exactement les grandeurs conservées. Une implémentation qui réalise seulement les trois premières étapes double-compte le liquide.

#figure(table(columns: 3, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  table.header([* Condition *], [* Fonction *], [* Défaut associé *]),
  [Critère],
  [Rôle],
  [Risque si absent],
  [Connectivité face-à-face],
  [Séparer les composantes],
  [Fusion artificielle de gouttes],
  [Diamètre équivalent],
  [Écarter les grosses structures],
  [Conversion d’un ligament],
  [d / Delta],
  [Vérifier la sous-résolution],
  [Parcel mal représenté par VOF],
  [Sphéricité / forme],
  [Sélectionner les gouttes],
  [Conversion d’une nappe ou d’un filament],
  [Persistance / hystérésis],
  [Éviter les reconversions],
  [Création répétée de parcels],
  [Distance à l’interface],
  [Éviter la zone primaire],
  [Transition trop proche du jet],
), caption: [Critères de décision à distinguer du simple seuil alpha]) <tab_criteria>

= Critères de transition VOF vers DPM
Le seuil alpha >= alpha_threshold est un seuil de sélection de cellules et non un critère de goutte. Il doit être combiné à un seuil de diamètre équivalent, à une résolution locale d_eq / Delta, et à un indicateur de forme. Les travaux fondés sur Connected Component Labeling calculent le volume, le centre et la vitesse d’une composante par pondération alpha V, puis déclenchent la conversion lorsque la taille et la sphéricité sont compatibles ［@chen2025］. Pour des gouttes proches ou en contact, CCL peut fusionner des objets ; une segmentation watershed ou une analyse de distance peut alors être nécessaire.

= Conservation de masse, volume et quantité de mouvement
Pour un liquide incompressible de densité rho_l, la masse du fragment est m_F = rho_l V_F. Après conversion, le bilan global doit inclure le VOF résiduel et les parcels. La vitesse initiale U_F doit être la moyenne volumique. Le retour vers le fluide doit être opposé au transfert reçu par le parcel et discrétisé sur le même intervalle Delta t. En compressible, retirer la masse modifie aussi rho, l’énergie et potentiellement la pression ; un simple terme de quantité de mouvement ne constitue pas un couplage thermodynamique complet ［@chen2025］.

#align(center)[$ M = sum_i (rho alpha V) + sum_p m $] <eq_mass>
#align(center)[#text(size: 9pt)[Bilan de masse total VOF plus DPM]]

#align(center)[$ P = sum_i (rho alpha V U) + sum_p m U $] <eq_momentum>
#align(center)[#text(size: 9pt)[Bilan de quantité de mouvement total]]

= Audit du code Python foampilot
Le convertisseur VofToDpmConverter implémente correctement le parcours des composantes connexes, le poids alpha V, le centroïde, la vitesse moyenne et le diamètre équivalent. Les contrôles sur alpha, volumes, dimensions et indices rendent les erreurs explicites. Les filtres min_cells et min_volume sont identifiés comme potentiellement dissipatifs : un fragment rejeté doit être comptabilisé et rester dans VOF dans une future transition temps réel. Le lecteur actuel est ASCII seulement et ne résout pas la réconciliation MPI.

#figure(``` python
weights = alpha[indices] * cell_volumes[indices]
volume = sum(weights)
centroid = sum(centres[indices] * weights[:, None]) / volume
velocity = sum(U[indices] * weights[:, None]) / volume
```, caption: [Noyau mathématique audité du calcul des propriétés d’un fragment])

= Audit de l’utilitaire C++ vofToDpm
L’utilitaire C++ reproduit l’algorithme Python avec mesh.cellCells⁽⁾ et un parcours de composantes. Il écrit les positions, les propriétés et un rapport des volumes sélectionné, converti et rejeté. Le garde-fou serial-only est correct pour un prototype, car une composante coupée par une frontière MPI ne peut pas être traitée localement sans fusion de labels et réduction de moments. L’option rhoLiquid est adaptée à l’incompressible homogène mais doit devenir une intégration locale rho_l alpha V pour le compressible.

= Audit des fvModels OpenFOAM 13
Les modèles incompressibleVoFClouds et compressibleVoFClouds recherchent le mélange de phases, construisent ou récupèrent la viscosité du cloud, créent un parcelCloudList et font évoluer le cloud une seule fois par timeIndex. Le hook addSup ajoute le terme source de quantité de mouvement du cloud à l’équation U. Les smoke tests démontrent la sélection du modèle, l’évolution du cloud et une quantité de mouvement Lagrangienne non nulle. Ils utilisent cependant manualInjection : ils ne démontrent pas la conversion d’un fragment VOF.

#figure(table(columns: 3, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  table.header([* Fonction *], [* État audité *], [* Exigence suivante *]),
  [Propriété],
  [Actuellement],
  [Pour production],
  [Détection CCL],
  [Offline Python/C++],
  [Dans le cycle solver],
  [Création parcel],
  [manualInjection],
  [Insertion dynamique transactionnelle],
  [alpha],
  [Non consommé],
  [Décrément borné et atomique],
  [MPI],
  [Série],
  [Fusion des composantes aux frontières],
  [Momentum],
  [Retour cloud SU],
  [Bilan opposé à la conversion],
  [Compressible],
  [Couplage mécanique],
  [Masse + énergie + thermo],
), caption: [Matrice de maturité de l’implémentation]) <tab_maturity>

= Mauvaises simplifications à éviter
La première simplification incorrecte serait de remplacer alpha par 0 ou 1 avant l’intégration : elle détruit le volume d’interface. La deuxième serait de pondérer le centre par le nombre de cellules : elle fausse la position dans un maillage non uniforme. La troisième serait de convertir sur alpha seul : alpha mesure une fraction locale et non la sphéricité ou la résolution d’une goutte. La quatrième serait de créer un parcel sans retirer alpha : elle viole la conservation de masse. La cinquième serait d’appliquer un retour de force sans vérifier le signe et le pas de temps. La sixième serait de traiter chaque partition MPI séparément : elle peut produire plusieurs parcels pour une seule goutte.

= Architecture recommandée
La prochaine couche doit être un service VofFragmentTransition avec un état par fragment. La phase detect produit les composantes et leurs moments. La phase decide applique taille, résolution, forme, distance et hystérésis. La phase create prépare la masse et la cinématique du parcel. La phase consume modifie alpha et les champs associés, puis confirme la transition seulement si les bornes et bilans sont satisfaits. Un identifiant stable doit empêcher la reconversion immédiate. En parallèle, les labels et moments doivent être fusionnés avant la décision.

= Plan de vérification recommandé
La vérification doit commencer par des cas analytiques à une, deux et plusieurs cellules, puis passer à des gouttes sphériques, des ligaments, des gouttes en contact et des fragments coupés par MPI. Chaque cas doit mesurer masse, volume, centre, moment, nombre de parcels et volume rejeté. Une étude de raffinement doit vérifier que le résultat ne dépend pas artificiellement du nombre de cellules choisies. Enfin, les cas compressibles doivent comparer masse, énergie et pression avant et après conversion.

#figure(table(columns: 2, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  table.header([* Campagne *], [* Vérification *]),
  [Test],
  [Critère d’acceptation],
  [Fragment analytique],
  [V et rho V exacts à la tolérance flottante],
  [Vitesse pondérée],
  [sum⁽m_p U_p⁾ = sum⁽rho alpha V U⁾],
  [Filtre],
  [Volume rejeté explicitement rapporté],
  [Double conversion],
  [Aucun parcel répété pour le même fragment],
  [MPI],
  [Un fragment global, pas un fragment par partition],
  [Compressible],
  [Masse, énergie et fermeture thermo cohérentes],
), caption: [Plan minimal de qualification scientifique]) <tab_tests>

= Conclusion
Le code actuel ne contient pas de mauvaise simplification dans son noyau offline lorsqu’il est interprété comme un extracteur de propriétés pondérées par alpha V. En revanche, deux simplifications deviennent incorrectes si l’on prétend parler de transition complète : l’absence de consommation de alpha et l’utilisation d’une injection manuelle à la place d’une création dynamique liée à la détection. La note recommande donc de conserver l’architecture actuelle comme base pédagogique et d’ajouter une couche transactionnelle de transition avant toute revendication de conservation automatique ou de validation d’atomisation.


#bibliography("vof_to_dpm.bib", style: "ieee")