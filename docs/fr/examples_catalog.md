# Catalogue complet des exemples et tutoriels exécutables

Cette page est la carte des exemples exécutables actuellement distribués avec FoamPilot. Chaque tutoriel est volontairement assez petit pour être inspecté, mais chacun suit aussi le même schéma reproductible : définir le cas en Python, générer les dictionnaires OpenFOAM, créer ou importer le maillage, appliquer les conditions aux limites, lancer le solveur, post-traiter le résultat et archiver les artefacts générés.

## Comment lire un tutoriel

Chaque tutoriel doit être lu à trois niveaux. Le niveau physique explique les lois de conservation, les hypothèses constitutives et les nombres adimensionnels. Le niveau OpenFOAM identifie le solveur, les champs, les dictionnaires, les conditions aux limites et les function objects. Le niveau FoamPilot montre quels objets Python génèrent ces fichiers et comment le flux de travail peut être reproduit ou paramétré.

Avant d'exécuter un tutoriel, vérifiez la version d'OpenFOAM indiquée dans son README, la géométrie externe ou les données de tutoriel requises par le script, et la disponibilité des dépendances Python optionnelles. Les cas générés doivent être inspectés et validés avec `checkMesh` et le journal du solveur avant d'interpréter les graphiques.

## Matrice récapitulative

| Tutorial | Physique principale | Famille de solveurs | Stratégie de maillage | Sorties principales |
| --- | --- | --- | --- | --- |
| `01_cavity_laminar` | Récirculation incompressible laminaire | `icoFoam` / écoulement incompressible transitoire | `blockMesh` | Vitesse, pression, résidus, figures, rapport |
| `02_simpleCar_turbulent` | Aérodynamique externe turbulente stationnaire | `simpleFoam` / RANS incompressible | Gmsh ou géométrie importée avec patches de bord | Vitesse, pression, forces sur la paroi, rapport |
| `03_pitzDaily_step` | Séparation et réattachement sur marche arrière | `simpleFoam` | Gmsh ou géométrie canalisée structurée | Longueur de réattachement, résidus, profils |
| `04_damBreak_multiphase` | Surface libre transitoire eau-air | `interFoam` | Domaine 2-D style Gmsh/block | Évolution de l'interface, fraction de phase, animation |
| `05_scalarTransport` | Transport de scalaire passif ou analogue température | `scalarTransportFoam` function object | Maillage de canal | Contours du scalaire, historiques temporels, données CSV |
| `06_buildingAero` | Vent urbain externe et sillage | `simpleFoam` | Fond `blockMesh` plus `snappyHexMesh` | Champ de vent, turbulence, statistiques de sillage et de bâtiments |
| `07_motorBike` | Aérodynamique externe d'un véhicule | `simpleFoam` ou OpenFOAM-13 `incompressibleFluid` path | `blockMesh` plus `snappyHexMesh` | Traînée, pression, sillage, animation, rapport |
| `08_thermalBuoyancy` | Convection naturelle avec flottabilité | Écoulement thermique Boussinesq/compressible | `blockMesh` | Température, `U`, `p_rgh`, résidus, rapport thermique |
| `09_CHT_heatedDuct` | Transfert de chaleur conjugué en régions fluide et solide | `chtMultiRegionFoam` | `blockMesh`, zones, séparation de régions | Températures par région, flux de chaleur, nombre de Nusselt, bilan |
| Muffler case study | Écoulement interne, perte de charge, analyse acoustique/fluidique | Flux de travail OpenFOAM spécifique au cas | Configuration pilotée par JSON/géométrie | Pression, vitesse, rapport acoustique ou d'écoulement |
| SimpleCar case study | Cas externe scripté avec configuration de maillage JSON | Flux incompressible spécifique au cas | Génération de maillage basée sur JSON | Dictionnaires du cas, champs, figures et rapport |

## 1. Lid-driven cavity: écoulement transitoire laminaire

### Objectif

La cavité est le cas canonique de vérification pour un écoulement visqueux incompressible. Une cavité carrée contient un fluide, la paroi supérieure se déplace à une vitesse prescrite, et les parois restantes sont fixes. Le cas isole la diffusion visqueuse, le couplage pression-vitesse, les conditions d'adhérence à la paroi et le développement des cellules de recirculation.

### Modèle mathématique

Le tutoriel résout les équations de Navier–Stokes incompressibles :

$$
\nabla\cdot\mathbf{U}=0,
$$

$$
\frac{\partial\mathbf{U}}{\partial t}+\nabla\cdot(\mathbf{U}\mathbf{U})
=-\nabla p+\nu\nabla^2\mathbf{U}.
$$

Le nombre de Reynolds est en général maintenu suffisamment bas pour obtenir une solution de référence laminaire. Le problème est transitoire car l'écoulement démarre du repos et tend vers un état recirculant stationnaire.

### Flux de travail FoamPilot

Le script crée un cas `blockMesh` bidimensionnel, écrit `controlDict`, `fvSchemes`, `fvSolution`, et `transportProperties`, puis applique une vitesse de couvercle mobile et des parois en non-glissement. Il exécute le solveur transitoire, extrait les résidus et génère des graphiques ou un rapport.

### Que vérifier

Les quantités de vérification principales sont les profils de vitesse sur la ligne centrale, le nombre et la position des cellules de recirculation, la décroissance des résidus et la sensibilité au raffinement du maillage et au pas de temps. Un vortex visuellement plausible n'est pas suffisant : comparez les profils sur la ligne centrale à une référence publiée ou au cas de référence OpenFOAM.

## 2. SimpleCar : aérodynamique externe turbulente stationnaire

### Objectif

Ce cas présente un écoulement externe autour d'un véhicule simplifié. Il montre comment une géométrie de corps est placée dans un domaine de type soufflerie, comment la turbulence d'entrée est prescrite, et comment les forces de pression et de cisaillement sont extraites de la surface du corps.

### Modèle et hypothèses

L'écoulement est incompressible et turbulent. Une fermeture RANS, couramment `kOmegaSST` dans la famille de cas, remplace les contraintes turbulentes non résolues par un modèle de viscosité turbulente. Le modèle SST est choisi car il mélange la sensibilité proche des parois de la famille $k$–$\omega$ avec un comportement plus tolérant au libre écoulement loin des parois. C'est un compromis pratique pour la séparation autour de corps massifs ou profilés ; ce n'est pas un substitut à la validation du modèle.

Le coefficient de traînée est obtenu par :

$$
C_D=\frac{F_D}{\tfrac12\rho U_\infty^2 A_\mathrm{ref}},
$$

où $F_D$ est la force dans la direction d'écoulement, $\rho$ est la densité, $U_\infty$ est la vitesse de référence du vent, et $A_\mathrm{ref}$ est la surface de référence choisie.

### Maillage et conditions aux limites

Le domaine d'arrière-plan doit être suffisamment long en amont et en aval pour éviter de contaminer le champ de pression autour du véhicule. La surface de la voiture nécessite un patch de paroi nommé. La vitesse d'entrée, l'énergie cinétique turbulente et la dissipation turbulente ou la dissipation spécifique sont prescrites de manière cohérente ; la sortie doit éviter les réflexions artificielles ; le traitement du sol doit correspondre au fait que le véhicule soit immobile, en mouvement, ou représenté dans un référentiel à sol mobile.

### Post-traitement

Extraire la pression de surface, le WSS, les coefficients de force intégrés, les régions de séparation et les profils de vitesse dans le sillage. Toujours rapporter la surface de référence, la vitesse de référence, la densité, le modèle de turbulence, le traitement des parois et les statistiques du maillage conjointement avec $C_D$.

## 3. PitzDaily : marche arrière

### Objectif

La marche arrière est un écoulement interne séparé utilisé pour étudier le développement de la couche de cisaillement, la recirculation, le réattachement et la sensibilité au modèle de turbulence.

### Physique

L'écoulement entrant traverse une expansion brusque. Une bulle de séparation se forme derrière la marche, et la longueur de réattachement dépend du nombre de Reynolds, du profil d'entrée, du modèle de turbulence, de la résolution des parois et des schémas numériques. Le cas est stationnaire dans sa configuration nominale de solveur, mais l'écoulement séparé peut présenter un comportement instationnaire si le maillage, le pas de temps ou le modèle le permettent.

### Diagnostics principaux

La sortie la plus importante est la longueur de réattachement, normalement exprimée relative à la hauteur de la marche. Les diagnostics complémentaires sont la pression à la paroi, le WSS, la vitesse sur l'axe central, la longueur de la zone de recirculation et les historiques de résidus. Le résultat ne doit pas être jugé uniquement à partir des résidus car un solveur stationnaire peut converger vers une solution numériquement stable mais physiquement biaisée.

## 4. DamBreak : VOF multiphasique transitoire

### Objectif

Le cas DamBreak illustre un problème transitoire à surface libre. Une colonne d'eau s'effondre sous l'effet de la gravité et déplace l'air. L'interface est représentée par un champ de fraction de phase, normalement `alpha.water`.

### Modèle gouvernant

L'approche VOF résout une équation de transport pour la fraction de phase :

$$
\frac{\partial\alpha}{\partial t}+\nabla\cdot(\alpha\mathbf{U})=0,
$$

avec compression d'interface et contrôles de bornitude. La densité et la viscosité du mélange sont reconstruites à partir des fractions de phase. La gravité entraîne l'effondrement et la pression doit être interprétée de manière cohérente avec la contribution hydrostatique.

### Priorités numériques

Le nombre de Courant, la compression de l'interface, le respect des bornes de la fraction de phase, et l'adaptation du pas de temps sont plus importants que d'augmenter simplement le nombre d'itérations. Inspectez `alpha.water` à plusieurs instants, vérifiez que $0\leq\alpha\leq1$, et assurez-vous que le volume liquide est conservé dans la tolérance numérique attendue.

### Sorties

Le tutoriel convient pour exporter des instantanés d'interface, des animations, des historiques de hauteur de surface libre, des champs de pression et des résidus. Le même schéma peut être réutilisé pour des cas de tangage, remplissage, vidange ou impact de vagues, mais chaque application exige une validation séparée de la tension de surface, du mouillage et des hypothèses sur la ligne de contact.

## 5. Transport de scalaire

### Objectif

Ce cas transporte un scalaire passif à travers un canal. Le scalaire peut représenter une concentration, un traceur, un polluant, ou une grandeur analogue à la température lorsque l'équation d'énergie est volontairement simplifiée.

### Équation

Pour une diffusivité constante $D$ :

$$
\frac{\partial C}{\partial t}+\nabla\cdot(\mathbf{U}C)
=\nabla\cdot(D\nabla C)+S_C.
$$

Le scalaire ne modifie pas l'écoulement à moins qu'un couplage par flottabilité, variation de densité, réaction ou modèle de source ne soit ajouté. Cette séparation rend le cas utile pour tester la numérisation advection-diffusion et les conditions aux limites pilotées par CSV.

### Diagnostics

Comparez le profil du scalaire avec l'échelle de longueur convective-diffusive attendue. Reportez le nombre de Peclet du scalaire, le profil d'entrée, la diffusivité, le traitement de la sortie, le respect des bornes et le schéma numérique. Si le scalaire représente la température, précisez clairement s'il s'agit d'un champ passif ou d'un modèle thermique entièrement couplé.

## 6. Aérodynamique de bâtiment : vent urbain externe

### Objectif

Le cas bâtiment introduit un groupe d'obstacles dans un domaine de type couche limite atmosphérique ou soufflerie. Il illustre la différence entre un maillage héxaédrique d'arrière-plan et un raffinement local basé sur les surfaces.

### Modèle physique

L'écoulement est généralement incompressible et turbulent. Pour un premier modèle d'ingénierie, une fermeture RANS stationnaire telle que $k$–$\epsilon$ ou $k$–$\omega$ SST est souvent choisie car elle offre un coût gérable pour les prévisions de vent moyen et de sillage. Le modèle ne résout pas tous les tourbillons transitoires ; il prédit leur effet moyen via la viscosité turbulente.

Pour les applications atmosphériques, une entrée uniforme n'est acceptable que lorsque le problème physique correspond à une soufflerie contrôlée. Pour une vraie couche limite atmosphérique, les champs de vitesse et de turbulence en entrée doivent dépendre de la hauteur et être mutuellement cohérents. Voir [Outdoor wind theory](theory_applied.md#outdoor-wind-and-atmospheric-boundary-layers).

### Flux de maillage

La séquence typique est :

```text
background blockMesh
→ surfaceFeatureExtract
→ snappyHexMesh castellated mesh
→ snap to building surfaces
→ optional boundary layers
→ checkMesh and patch validation
```

Les bâtiments, le sol, l'entrée, la sortie, les limites latérales et la limite supérieure doivent avoir des noms stables. Le raffinement doit être concentré autour des arêtes des bâtiments, des lignes de toit, des passages en canyon et des régions de sillage plutôt que d'être appliqué uniformément.

### Sorties

Les sorties utiles incluent la vitesse au niveau piéton, les vecteurs de vitesse, la pression, l'énergie cinétique turbulente, la recirculation sur les toits et dans les canyons de rue, et les statistiques sur des patches de bâtiments sélectionnés. Reportez le profil d'entrée, les hypothèses de rugosité, les fonctions de paroi, l'étendue du domaine, les niveaux de raffinement et le nombre de cellules.

## 7. MotorBike : géométrie externe complexe

### Objectif

L'exemple MotorBike est un cas d'aérodynamique externe basé sur la surface, plus exigeant. Il teste l'importation de géométrie, l'extraction de caractéristiques, le snapping de surface, le raffinement local, les patches de paroi, l'intégration des forces et l'animation.

### Choix du modèle

Les scripts du dépôt et le README contiennent des références dépendantes de la version. Inspectez le script réel avant d'exécuter : certaines configurations utilisent une voie `simpleFoam`/`incompressibleFluid`, tandis que la documentation du script fait aussi référence à un modèle RAS Spalart–Allmaras. Le solveur sélectionné, le modèle de turbulence, le traitement des parois et la source de géométrie doivent être consignés dans le cas généré.

Spalart–Allmaras est attractif pour les écoulements aérodynamiques externes attachés ou modérément séparés car il est relativement peu coûteux et résout une seule variable de turbulence transportée. $k$–$\omega$ SST peut être préféré lorsque le comportement en séparation et la robustesse face aux gradients de pression défavorables sont plus importants. Aucun choix n'est universellement supérieur ; le maillage et les données de validation dominent souvent l'incertitude.

### Maillage et validation

Utilisez un maillage grossier pour valider l'orientation de la géométrie et les noms de patches, puis affinez les bords d'attaque, les roues, les carénages, les contacts avec le sol et le sillage. Vérifiez que la surface n'a pas de fuites ni de normales inversées et que la taille locale des cellules supporte le traitement des parois prévu. Comparez la traînée et les distributions de pression uniquement après avoir fixé les quantités de référence pour les forces.

## 8. Flottabilité thermique : convection naturelle

### Objectif

L'exemple de flottabilité thermique modélise une pièce ou une cavité chauffée avec gravité, différences de température et écoulement entraîné par la flottabilité.

### Approximation de Boussinesq

Pour des différences de température modérées, les variations de densité peuvent être négligées dans la continuité et les termes d'inertie et ne conservées que dans la force de flottabilité. Une relation typique est :

$$
\rho\approx\rho_0[1-\beta(T-T_0)],
$$

et la contribution de flottabilité est proportionnelle à $\rho_0\beta(T-T_0)\mathbf{g}$. C'est économiquement plus avantageux qu'un traitement pleinement compressible du gaz parfait, mais cela ne doit pas être utilisé lorsque les variations de densité sont importantes, que la compressibilité a un rôle ou que l'approximation linéaire en température n'est pas valide.

### Conditions aux limites et diagnostics

Le cas prescrit des parois chaudes et froides, des surfaces adiabatiques ou isolées, la gravité et un modèle de turbulence thermique. Surveillez $T$, $U$, `p_rgh`, $k$, $\epsilon$ ou $\omega$, et `alphat` selon le cas. Les principaux nombres adimensionnels sont le nombre de Rayleigh, le nombre de Prandtl et le nombre de Nusselt. Validez les différences de température, les cellules de circulation et les taux de transfert de chaleur contre une référence lorsque cela est possible.

## 9. Heated duct : transfert de chaleur conjugué

Le cas de conduit chauffé est documenté en détail dans [CHT case setup](cht_workflow.md). Il s'agit de l'exemple de référence pour la création de régions fluide-sol, les champs spécifiques par région, les propriétés des matériaux, les interfaces couplées, `chtMultiRegionFoam`, le post-traitement direct ou via VTK, et le rapport de bilan thermique.

## 10. Muffler case study

Le cas muffler est un exemple plus vaste et orienté application. Il montre comment FoamPilot peut combiner la gestion de la géométrie, la modélisation d'écoulement interne, l'analyse de perte de charge, le post-traitement acoustique ou fluidique, et la génération de rapports. La page pertinente est [Detailed muffler example](example/muffler/detailled_example_muffler.md).

Les décisions majeures de modélisation sont le volume interne et les passages perforés ou connectés, les données de pression/débit d'entrée et de sortie, les hypothèses de rugosité des parois, la compressibilité ou l'incompressibilité, et la gamme de fréquences si des quantités acoustiques sont interprétées. Un champ de pression seul n'est pas une prédiction acoustique ; les hypothèses acoustiques et la stratégie d'échantillonnage doivent être documentées.

## 11. SimpleCar case study

La page détaillée SimpleCar complète le tutoriel turbulent exécutable. Elle se concentre sur une configuration de cas pilotée par JSON, la configuration du maillage, la manipulation des dictionnaires OpenFOAM, les conditions aux limites et la génération automatique de rapports. Utilisez-la pour apprendre comment un script de niveau projet peut générer un cas complet plutôt que de reproduire seulement un petit benchmark.

## 12. Exemples additionnels de thermique et de géométrie

Le dépôt contient aussi des exemples spécialisés et des utilitaires autour de la conversion de géométrie, du traitement de surfaces d'aorte, des entrées météo/EPW, des profils de vent, de géométries humaines, de MakeHuman/JOS-3 pour la thermorégulation, et du couplage CSV. Ceux-ci ne sont pas tous équivalents à des tutoriels de solveur : certains sont des flux de prétraitement ou d'échange de données. Leur documentation doit donc indiquer le format des données d'entrée, le système de coordonnées, le logiciel externe, les artefacts générés et les vérifications de validation.

## Artefacts des tutoriels et reproductibilité

Les répertoires de tutoriels peuvent contenir des scripts d'exécution, des fichiers de géométrie, des exportations de résidus, des images, des animations et des rapports générés. Séparez les résultats générés des entrées sources lorsque vous adaptez un tutoriel. Enregistrez la version d'OpenFOAM, l'environnement Python, le nombre de cellules du maillage, les paramètres du solveur, les critères de convergence et toute modification manuelle des dictionnaires générés.

## Ce que chaque tutoriel ne démontre pas

Un tutoriel démontre un flux de travail ; il n'établit pas la précision industrielle. La précision requiert une convergence en maillage, des études de pas de temps ou de nombre de Courant, la sensibilité au modèle, des contrôles de conservation, la comparaison avec des données analytiques ou expérimentales, et une déclaration d'incertitude. Plus la géométrie ou la physiologie est complexe, plus ces vérifications deviennent importantes.
