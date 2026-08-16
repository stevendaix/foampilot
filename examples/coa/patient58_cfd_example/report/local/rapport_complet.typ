
#set document(title: "Validation du Modèle Windkessel", author: "Foampilot Automated Report")
#set page(paper: "a4", margin: 2.5cm, numbering: "1 / 1")
#set text(font: "New Computer Modern", size: 11pt, lang: "fr")
#set heading(numbering: "1.1.")
#set par(justify: true)
#show figure.caption: it => [
  #text(weight: "bold", size: 0.9em)[#it.supplement #it.counter.display():] #it.body
]


= Validation du Modèle Windkessel


= Résumé

Cette étude présente la validation du modèle Windkessel à 5 éléments (Rc, Rp, C, L, Cprox) 
contre des données de référence expérimentales. Le modèle est un modèle lumped-parameter 
cardiovasculaire utilisé pour simuler la pressurisation aortique.


= Théorie du Modèle Windkessel


== Le modèle Windkessel est un modèle lumped-parameter cardiovasculaire basé sur l'analogie entre les systèmes hydraulique et électrique :

- Pression P (Pa) <-> Tension V (unités : Pa ou mmHg)
- Débit Q (m³/s) <-> Courant I (unités : m³/s)
- Résistance R <-> Résistance R (unités : Pa·s/m³)
- Compliance C <-> Capacitance C (unités : m³/Pa)
- Inertance L <-> Inductance L (unités : Pa·s²/m³)


== Modèle à 4 Éléments (Rc, Rp, C, L)
Le modèle 4-éléments inclut une inertance L en série avec Rc.

#figure($ C ((d P_2) / (d t)) + P_2 / R_p = Q(t) $, caption: [Équation différentielle au nœud de compliance]) <eq:4e_ode>

#figure($ P_1(t) = P_2(t) + R_c Q(t) + L ((d Q) / (d t)) $, caption: [Reconstruction de la pression aortique]) <eq:4e_recon>

#figure($ tau = R_p C $, caption: [Constante de temps diastolique (tau = Rp x C)]) <eq:tau>

== Modèle à 5 Éléments (Rc, Rp, C, L, Cprox)
Le modèle 5-éléments ajoute une compliance proximale Cprox.

#figure($ C_("prox") ((d P_("prox")) / (d t)) = Q(t) - (P_("prox") - P_2) / R_c $, caption: [Équation pour la compliance proximale]) <eq:5e_prox>

#figure($ C ((d P_2) / (d t)) = (P_("prox") - P_2) / R_c - P_2 / R_p $, caption: [Équation pour la compliance distale]) <eq:5e_dist>

#figure($ P_1(t) = P_("prox")(t) + L ((d Q) / (d t)) $, caption: [Reconstruction de la pression aortique]) <eq:5e_recon>

= Paramètres du Modèle


#figure(table(columns: 3, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  table.header([* Paramètre *], [* Valeur *], [* Unité *]),
  [Paramètre],
  [Valeur],
  [Unité],
  [Résistance caractéristique (Rc)],
  [1.00e+06],
  [Pa·s/m³],
  [Résistance périphérique (Rp)],
  [2.00e+09],
  [Pa·s/m³],
  [Compliance distale (C)],
  [2.00e-07],
  [m³/Pa],
  [Inertance (L)],
  [5.00e+03],
  [Pa·s²/m³],
  [Compliance proximale (Cprox)],
  [1.00e-08],
  [m³/Pa],
), caption: [Paramètres du modèle Windkessel 5 éléments (avec Cprox)]) <tab:parameters>

== Constante de Temps Théorique


=== 
La constante de temps diastolique théorique est calculée par τ = Rp × C.

τ = 2.00e+09 × 2.00e-07 = 400.0000 s



= Métriques de Validation


#figure(table(columns: 4, stroke: 0.5pt, inset: 7pt, align: center + horizon,
  table.header([* Métrique *], [* Valeur *], [* Seuil *], [* Statut *]),
  [Métrique],
  [Valeur],
  [Seuil],
  [Statut],
  [NRMS error],
  [0.03%],
  [≤ 15%],
  [✅ PASS],
  [Peak time shift],
  [-34.43 ms],
  [≤ 50 ms],
  [✅ PASS],
  [Relative tau error],
  [8.00%],
  [≤ 20%],
  [✅ PASS],
), caption: [Métriques de validation du modèle]) <tab:metrics>

= Analyse Détaillée


== Constante de Temps Diastolique


=== 
La constante de temps diastolique (τ) caractérise la décroissance exponentielle 
de la pression durante la diastole.

- τ simulé : 0.2371 s
- τ référence : 0.2578 s



== Comparaison des Formes d'Onde


=== 
La forme d'onde de pression simulée est comparée à la référence après 
mise à l'échelle affine pour s'affranchir des différences d'amplitude.

L'erreur NRMS de 0.03% indique un écart global 
acceptable entre les deux signaux.



= Visualisations


#figure(image("validation_waveform.png", width: 80%), caption: [Comparaison des formes d'onde de pression - Simulé vs Référence]) <fig:waveform>

= Conclusion


== La validation du modèle Windkessel est ACCEPTÉE. Toutes les métriques sont dans les tolérances définies:
- L'erreur NRMS est inférieure à 15%
- Le décalage temporel du pic est inférieur à 50ms
- L'erreur relative sur τ est inférieure à 20%

Le modèle Windkessel avec les paramètres définis reproduit correctement la dynamique de pressurisation aortique.
