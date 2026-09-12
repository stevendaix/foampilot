# Cours : de la méthode VOF au DPM avec Python et foampilot

## Objectifs pédagogiques

À l’issue de ce cours, l’étudiant doit être capable d’expliquer pourquoi une simulation d’atomisation combine souvent une description eulérienne d’interface et une description lagrangienne de gouttes, de calculer les propriétés intégrales d’un fragment liquide, d’identifier les conditions de transition VOF–DPM, d’utiliser le convertisseur Python de foampilot et de vérifier séparément la conservation de masse, de volume et de quantité de mouvement.

Le cours adopte une position méthodologique importante : **la détection d’un fragment et la conversion physique de ce fragment sont deux opérations distinctes**. La première regroupe les cellules connectées ; la seconde doit créer un parcel et retirer du champ VOF exactement le volume converti. Confondre ces deux étapes conduit à compter deux fois le liquide.

## 1. Pourquoi un couplage hybride ?

La méthode VOF représente une interface entre deux fluides par une fraction volumique `alpha` dans un maillage eulérien. Elle est adaptée aux nappes, ligaments et zones de rupture primaire lorsque l’interface doit être résolue. Le DPM/LPT représente au contraire des gouttes sous forme de particules ponctuelles dont la position et la vitesse sont intégrées le long de trajectoires. Cette représentation est plus économique lorsque le spray est dilué et que les gouttes sont petites devant la maille. Cette complémentarité est au cœur des méthodes hybrides proposées pour l’atomisation [1] [2].

La contrepartie est une zone de transition délicate. Une goutte suffisamment petite pour être mal résolue par VOF ne doit pas rester dans la fraction liquide ; inversement, un ligament ou une nappe encore résolue ne doit pas être remplacé prématurément par un point matériel. Le critère de transition doit donc combiner au minimum taille équivalente, résolution par rapport à la maille, morphologie et connectivité.

## 2. Rappels VOF

Dans une cellule de volume `V_i`, la fraction liquide `alpha_i` définit un volume liquide `alpha_i V_i`. Pour un mélange incompressible de deux phases, la fermeture locale est généralement `alpha_liquid + alpha_gas = 1`. Les propriétés mélangées peuvent être écrites, dans une approximation simple, comme `rho = alpha rho_l + (1-alpha) rho_g` et `mu = alpha mu_l + (1-alpha) mu_g` [3].

La quantité de mouvement eulérienne reçoit les contributions de pression, viscosité, gravité, tension superficielle et interaction avec les gouttes. Dans une formulation de couplage bidirectionnel, la force exercée par le DPM sur le fluide est la réaction de la force exercée par le fluide sur les parcels.

## 3. Rappels DPM/LPT

Une goutte de masse `m_p` et de vitesse `U_p` obéit à une équation de type Newton :

```text
m_p dU_p/dt = F_drag + F_pressure + F_virtual_mass + m_p g + ...
```

Dans un spray dilué gaz–liquide, la traînée domine souvent. Avec un diamètre `d_p`, une vitesse relative `U-U_p` et un Reynolds particulaire `Re_p`, une loi de Schiller–Naumann constitue un modèle de base courant [3]. Le cloud accumule les transferts et les redistribue dans les cellules porteuses, notamment sous la forme d’une matrice source de quantité de mouvement.

## 4. Détection d’un fragment VOF

Le convertisseur foampilot sélectionne les cellules satisfaisant `alpha_i >= alpha_threshold`, puis construit les composantes connexes sur la connectivité face-à-face du maillage. Pour une composante `F`, les grandeurs physiques sont calculées avec le poids liquide `w_i = alpha_i V_i` :

```text
V_F = sum(i in F) alpha_i V_i
x_F = sum(i in F) alpha_i V_i x_i / V_F
U_F = sum(i in F) alpha_i V_i U_i / V_F
m_F = rho_l V_F
d_eq = (6 V_F / pi)^(1/3)
```

Le seuil `alpha_threshold` ne renormalise jamais `alpha`. Il sélectionne des cellules, mais la masse conservée continue d’utiliser la fraction physique. Cette distinction est essentielle lorsque des cellules d’interface contiennent `0 < alpha < 1`.

## 5. Critères de transition

Un critère minimal est `d_eq < d_transition`, mais il est insuffisant seul. On recommande d’ajouter un critère de résolution, par exemple `d_eq / Delta < C_d`, où `Delta` est une longueur caractéristique de maille, ainsi qu’un critère morphologique et un critère de connectivité. Les travaux récents utilisent notamment le Connected Component Labeling et des indicateurs de sphéricité ou de forme [3] [4].

Une conversion robuste doit également imposer une hystérésis ou un registre des fragments déjà convertis. Sans mémoire temporelle, une cellule proche du seuil peut produire des parcels répétés à chaque pas.

## 6. Conservation lors de la conversion

La conservation de masse nécessite deux opérations atomiques : créer un parcel de masse `m_F = rho_l V_F`, puis retirer le volume `V_F` du champ VOF. Dans le cas incompressible, cela signifie modifier `alpha` de façon bornée ; dans le cas compressible, il faut être cohérent avec `rho`, l’énergie et la pression.

La vitesse initiale du parcel doit être la moyenne volumique `U_F`. Le moment transféré est alors `m_F U_F`. Si une correction est appliquée au fluide, elle doit être opposée au moment donné au parcel. Une vérification de bilan doit comparer avant/après :

```text
M_liquid_before = sum(alpha_i V_i rho_l) + sum(m_p)
P_before = sum(alpha_i V_i rho_l U_i) + sum(m_p U_p)
```

La différence tolérée doit être reliée aux erreurs d’intégration temporelle, de reconstruction d’interface, de parallélisation et de troncature numérique ; elle ne doit pas être masquée par une simple tolérance arbitraire.

## 7. Utilisation Python avec foampilot

```python
from foampilot.utilities.vof_to_dpm import VofToDpmConverter

converter = VofToDpmConverter(
    alpha_threshold=0.5,
    min_volume=0.0,
    min_cells=1,
    strict=True,
)

fragments = converter.extract_case(
    case_directory="case",
    time_directory="0.01",
    alpha_name="alpha.liquid",
    velocity_name="U",
)

outputs = converter.write_openfoam_outputs(
    fragments,
    output_directory="case/constant",
    cloud_name="vofToDpmCloud",
)

print(f"Fragments: {len(fragments)}")
print(f"Liquid volume: {converter.total_volume(fragments):.6e} m3")
print(outputs["report"])
```

Le module sépare volontairement la lecture ASCII OpenFOAM, l’algorithme de composantes connexes et l’écriture des sorties. Cette séparation permet de tester l’algorithme sur des tableaux NumPy synthétiques avant de l’exposer à des problèmes de maillage, de champs binaires ou de parallélisme.

## 8. Travaux pratiques

**TP 1 — pondération physique.** Construire deux cellules de volumes différents avec `alpha = 1` et `alpha = 0.5`. Vérifier que le centre et la vitesse sont pondérés par `alpha V`, et non par le nombre de cellules.

**TP 2 — connectivité.** Construire quatre cellules dont deux paires sont connectées. Vérifier que le résultat comporte deux fragments et que chaque volume est conservé.

**TP 3 — filtres.** Activer `min_cells` et `min_volume`. Calculer explicitement le volume sélectionné, le volume converti et le volume rejeté. Conclure qu’un filtre représente une perte physique tant que le liquide rejeté n’est pas réinjecté dans VOF.

**TP 4 — audit de cas.** Lire un cas OpenFOAM ASCII avec `extract_case`, produire le rapport JSON et vérifier les invariants suivants : indices valides, volumes positifs, fractions dans `[0,1]`, diamètre positif et cohérence entre le rapport et les fichiers de positions.

**TP 5 — passage vers OpenFOAM.** Utiliser les positions et propriétés produites pour une injection manuelle contrôlée. Comparer le nombre de parcels et leur masse avec le rapport Python. Cette étape ne doit pas être appelée conversion temps réel : elle ne retire pas encore automatiquement le volume du champ VOF.

## 9. Limites de l’implémentation actuelle

La version Python lit les fichiers ASCII et ne prend pas en charge les champs binaires. Elle travaille en série et ne réconcilie pas les composantes coupées par les frontières MPI. Elle écrit des descriptions de parcels, mais n’insère pas directement de particules dans un cloud déjà vivant et ne modifie pas `alpha`. La couche C++ `incompressibleVoFClouds`/`compressibleVoFClouds` valide l’évolution du cloud et le retour de quantité de mouvement, mais son test actuel utilise une injection manuelle. Ces limites sont des éléments pédagogiques du cours : elles montrent précisément la différence entre **extraction offline**, **couplage solver–cloud** et **conversion conservative temps réel**.

## Références

[1]: https://doi.org/10.1016/j.softx.2020.100483 "Heinrich & Schwarze, 3D-coupling of Volume-of-Fluid and Lagrangian particle tracking for spray atomization simulation in OpenFOAM"

[2]: https://www.nhr4ces.de/project/automatic-coupling-method-of-volume-of-fluid-and-lagrangian-particle-tracking-for-spray-atomization-simulation-in-openfoam/ "NHR4CES, Automatic coupling method of Volume-of-Fluid and Lagrangian particle tracking"

[3]: https://www.mdpi.com/2076-3417/15/9/4928 "Review on the recent numerical studies of liquid atomization"

[4]: https://www.mdpi.com/2073-4441/17/3/394 "Chen et al., Investigation of Splashing Characteristics During Spray Impingement Using VOF–DPM Approach"
