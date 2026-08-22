# Notes bibliographiques VOF–DPM — corpus initial

## [1] NHR4CES — Automatic coupling method of Volume-of-Fluid and Lagrangian particle tracking for spray atomization simulation in OpenFOAM
URL: https://www.nhr4ces.de/project/automatic-coupling-method-of-volume-of-fluid-and-lagrangian-particle-tracking-for-spray-atomization-simulation-in-openfoam/

La page présente un couplage automatique VOF–LPT destiné à couvrir l’atomisation complète sans interface localisée imposée entre les deux régimes. VOF est utilisé pour la fragmentation primaire et LPT pour le suivi des gouttes issues de la fragmentation secondaire et du spray dilué. Les gouttes sont traitées comme des masses ponctuelles sans volume résolu. Les cas cités sont un jet liquide en écoulement transversal et un atomiseur swirl, avec comparaison à des mesures expérimentales.

## [2] Heinrich & Schwarze (2020), SoftwareX 11, 100483 — 3D-coupling of Volume-of-Fluid and Lagrangian particle tracking for spray atomization simulation in OpenFOAM
DOI: https://doi.org/10.1016/j.softx.2020.100483
URL: https://www.sciencedirect.com/science/article/pii/S2352711020300303

L’article motive le couplage par la coexistence de nombreuses échelles spatiales et temporelles. VOF est adapté à la fragmentation primaire, tandis que le suivi lagrangien est adapté à la fragmentation secondaire et au spray dilué. Le travail propose un couplage tridimensionnel dans OpenFOAM, associé à l’AMR pour réduire le coût numérique, et le valide sur un jet de carburant en écoulement transversal.

## Conséquence pour notre implémentation

Notre modèle `fvModel` OpenFOAM 13 démontre actuellement le raccordement solver–cloud, l’évolution Lagrangienne et le retour de quantité de mouvement. Il ne doit pas être présenté comme une conversion automatique complète tant que la détection des composantes connexes VOF, la sélection des fragments, la création dynamique des parcels et la soustraction conservative de la masse liquide dans `alpha` ne sont pas raccordées au chemin d’exécution.

## [3] Chen et al. (2025), Water — Investigation of Splashing Characteristics During Spray Impingement Using VOF–DPM Approach
URL: https://www.mdpi.com/2073-4441/17/3/394

L’article explicite une formulation VOF–DPM bidirectionnelle. La fraction volumique VOF satisfait une fermeture air/eau, tandis que l’équation de quantité de mouvement du mélange reçoit une force de réaction des gouttes et une force de tension superficielle. La dynamique d’une goutte est formulée par la deuxième loi de Newton, avec traînée de Schiller–Naumann et, selon le régime, force de masse ajoutée et gradient de pression.

Pour la transformation DPM vers VOF, la fraction volumique de goutte dans une cellule est définie par `alpha_p = V_p/V_cell`. Lorsque la goutte doit revenir dans la description eulérienne, elle est supprimée du nuage et sa masse/quantité de mouvement sont injectées dans le champ. L’article donne un terme de quantité de mouvement de forme `S = m_p (U_p-U)/(V_cell Delta t)`.

Pour la transformation VOF vers DPM, l’article utilise le Connected Component Labeling (CCL). Les cellules d’une composante liquide sont regroupées, puis le volume est calculé par `V_p = sum(alpha_i V_i)`. Le diamètre équivalent est `d_p = (6 V_p/pi)^(1/3)`, le centre est pondéré par `sum(alpha_i V_i x_i)/V_p`, et la vitesse par `sum(alpha_i V_i U_i)/V_p`. La conversion est conditionnée par un diamètre inférieur à un seuil et par une condition de sphéricité/forme. Ce schéma correspond directement à la lacune identifiée dans notre couche C++ actuelle.

## Points théoriques à intégrer au cours

Le cours devra distinguer : (i) la représentation VOF d’une interface résolue, (ii) le DPM/LPT comme représentation de gouttes ponctuelles, (iii) le critère de transition basé sur taille, résolution et morphologie, (iv) le calcul des propriétés intégrales d’un fragment par pondération `alpha*V`, (v) la suppression de la masse liquide VOF lors de la conversion, et (vi) le retour de la réaction de quantité de mouvement vers l’équation eulérienne.

La conservation de masse n’est pas obtenue par la seule création d’un parcel : il faut retirer exactement le volume converti du champ VOF. De même, la conservation de quantité de mouvement exige d’initialiser `U_p` par la moyenne volumique du fragment et d’éviter de réinjecter deux fois la même force ou la même masse.
