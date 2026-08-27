# Validation progressive des calculs Foundation 13

## Test DTC moving overset

Le script `openfoam13/DTCMoving_Overset_Foundation13/Allrun.Foundation13` a été exécuté jusqu’à `t = 0.01 s`. Le calcul a écrit les temps intermédiaires de `0.001` à `0.01 s`, s’est terminé par `End`, et n’a produit aucune erreur fatale. Les erreurs de continuité finales sont de l’ordre de `1.9e-20` en somme locale et `2.1e-22` globalement. Le journal montre également l’appel du mouvement rigide 6-DoF à chaque pas.

Une réserve reste ouverte : le journal contient des avertissements de patches `hull` absents du maillage background. Ils n’empêchent pas ce test de référence de terminer, mais devront être supprimés ou documentés dans le cas final.

## Smoke test MRF/actuationDisk

Un cas temporaire basé sur le maillage DTC background a été préparé avec une zone `rotor` de 46 cellules, `MRFProperties` et `fvModels/actuationDisk`. Après adaptation des dictionnaires Foundation 13 et ajout d’un patch `MRFnoSlip` temporaire pour satisfaire le contrôle de `MRFZone`, le calcul `marineFoam -solver incompressibleVoF` a terminé un pas à `t = 1e-5 s` sans erreur fatale.

Le modèle `actuationDisk` est sélectionné correctement. Les résidus de pression atteignent environ `4e-8`, et les erreurs de continuité restent autour de `1e-10`. Ce test valide l’intégration logicielle MRF/actuationDisk, mais ne constitue pas une validation hydrodynamique de l’hélice réelle, car le maillage rotor/stator et les interfaces AMI ne sont pas encore présents.

## Limites actuelles

Le squelette `propellerFoundation13` demande encore `compressibleVoF`, alors que `marineFoam` expose actuellement `incompressibleVoF`. Le test propeller physique doit donc soit être porté vers le module Foundation 13 incompressible, soit être complété par un module `compressibleVoF` réellement implémenté. Le maillage rotor/stator et les patches `AMI1`/`AMI2` restent également à construire.

## Réserve physique DTC détectée

Le calcul DTC termine correctement, mais `background/postProcessing/forces/0/forces.dat` ne contient qu’une ligne de forces nulles. Le `dynamicMeshDict` du background référence `patches (hull)`, alors que le maillage background possède `atmosphere`, `inlet`, `outlet`, `bottom`, `side` et `midPlane`, sans patch `hull`. Foundation 13 émet donc des avertissements de patch absent. La cinématique 6-DoF est bien appelée, mais les forces hydrodynamiques ne sont pas encore mesurées sur la coque. Cette configuration ne doit pas être présentée comme une validation physique complète du DTC ; il faut relier la fonction de forces au maillage hull ou agréger correctement les contributions inter-maillages dans le runtime overset custom.

## Propeller FoamPilot Foundation 13

Le tutoriel Foundation 13 `incompressibleVoF/propeller` a été converti dans `openfoam13/FoamPilotCases/propellerFoundation13` et maillé par `Allmesh.FoamPilot` avec `blockMesh`, `surfaceFeatures`, `snappyHexMesh`, `createBaffles`, `splitBaffles`, `renumberMesh` et `createNonConformalCouples`. Le maillage contient 525 586 cellules, 11 patches, 1 cellZone et 2 faceZones. `checkMesh` détecte 4 faces incorrectement orientées et 15 faces fortement gauches ; le maillage est donc exploitable pour un smoke test mais doit être amélioré avant une validation hydrodynamique quantitative.

Le calcul `marineFoam -solver incompressibleVoF` a néanmoins terminé un pas à `t = 1e-5 s` avec le mouvement rotatif Foundation 13, les couples non conformes et le modèle de cavitation `VoFCavitation/SchnerrSauer`. Aucun `FOAM FATAL` ni erreur d’exécution n’a été produit. Les erreurs de continuité sont de l’ordre de `1e-14`, et le résidu final de `p_rgh` est inférieur à `1e-9`. Aucun MRF n’est encore actif dans cette conversion ; la rotation est actuellement réalisée par `solidBody/rotatingMotion`, conformément au tutoriel Foundation 13.

## Extraction des forces propeller

Après correction de l’inclusion `#includeFunc functions`, le calcul produit `postProcessing/forces/0/forces.dat`. À `t = 1e-5 s`, la force totale mesurée sur `propellerStem` et `propellerTip` est approximativement `(0.0931, 2258.51, 0.1353) N`, avec une contribution visqueuse `(3.16e-7, -9.44e-3, 1.65e-6) N`. Le moment total autour de l’origine est approximativement `(0.0191, 6.7851, -0.0167) N·m`, avec une contribution visqueuse de l’ordre de `1.07e-3 N·m` sur l’axe y. Ces valeurs sont un premier résultat de smoke test, et non encore une valeur convergée de poussée/couple, car le calcul ne couvre qu’un seul pas de temps.

## Vérification de la création du patch hull

La vérification du tutoriel et de notre pipeline confirme que le patch `hull` est créé pendant `snappyHexMesh` du sous-cas hull, après copie de `hull.stl` dans `constant/geometry` et génération de `hull.eMesh` par `surfaceFeatures`. Le dictionnaire Foundation 13 doit déclarer `file "hull.stl";`, contrairement à l’ancienne syntaxe utilisée auparavant.

Après correction, `snappyHexMesh -overwrite` termine avec une couche de paroi sur `hull` et le maillage final contient les patches `oversetHull`, `hullMidPlane` et `hull`. Le maillage compte 417 454 cellules, 469 632 points et 3 patches ; `checkMesh` retourne `Mesh OK`. Le défaut du calcul DTC précédent venait donc bien de l’absence d’exécution de `snappyHexMesh` sur le sous-cas hull, et non d’une création différée mystérieuse du patch par le solveur.

## Résultat du test avec le vrai maillage hull

Le pipeline `Allrun.Foundation13.snappy` confirme que `snappyHexMesh` crée bien le patch `hull` : le maillage hull Foundation 13 contient `oversetHull`, `hullMidPlane` et `hull`, avec 417 454 cellules et `checkMesh` valide. Le calcul séparé background + hull termine jusqu’à `t = 0.01 s` avec des erreurs de continuité de l’ordre de `10^-20`.

Les forces du fichier `background/postProcessing/forces/0/forces.dat` restent toutefois nulles, car la fonction `forces` est encore attachée au maillage background, qui ne contient pas le patch `hull`. Le solveur custom actuel calcule le background séparément et ne dispose pas encore d’une agrégation des contraintes de surface issues du sous-maillage hull. La création du patch est donc corrigée ; la prochaine correction doit porter sur l’extraction/agrégation des forces dans l’architecture overset séparée.

## Diagnostic du couplage donor après création de hull

Le code `marineFoam` charge un donor uniquement avec l’option `-donor-region`, et tente alors de lire `U`, `p_rgh`, `alpha.water`, `k`, `omega`, `epsilon` et `nut` dans le temps courant du donor. Après génération `snappyHexMesh`, le sous-cas `hull` possède bien `constant/polyMesh/boundary` avec le patch `hull`, mais ne possède pas encore de répertoire `0` ni ces champs initiaux. Le runner actuel ne passe pas `-donor-region hull`, car le donor n’est donc pas initialisé.

Cette vérification explique pourquoi le calcul précédent pouvait terminer avec une cinématique 6-DoF et des forces background nulles : le hull était maillé mais pas encore résolu ni chargé comme donor. La prochaine correction doit initialiser les champs du sous-maillage hull et faire évoluer le runner vers un vrai calcul multi-mailles ou vers une agrégation de forces de surface dédiée ; il ne faut pas simplement ajouter `-donor-region hull` sans ces champs.

## Contrat donor multi-région découvert

Le test `marineFoam -solver incompressibleVoF -donor-region ../hull` échoue avant la lecture des champs avec `Cannot find file "points" in directory "../hull/polyMesh" in times "0" down to constant`. La source `marineFoam.C` construit le donor avec `IOobject(donorRegion, runTime.timePath().name(), runTime)`, ce qui suppose un nom de région situé dans le même cas multi-région, et non un chemin vers un cas frère. Notre organisation actuelle `background/` et `hull/` en deux cas séparés ne respecte donc pas encore ce contrat. La solution propre est de créer une arborescence multi-région Foundation 13 ou d’ajouter au solver une option dédiée acceptant un chemin externe et ses champs.

## Premier calcul DTC avec couplage inter-mailles actif

Un harness Foundation 13 a été construit avec le maillage hull comme maillage receveur, le maillage background comme région donor interne et le dictionnaire de stencils orienté correctement (background→hull). Les champs donor ont été initialisés dans `0/background`, et `marineFoam -donor-region background` reconnaît désormais la région : `marineOversetConstraint: inter-mesh donor region=background, donor cells=134064`.

Le calcul atteint `t = 1.8548293e-4 s` sans erreur fatale et termine par `End`. Les erreurs de continuité restent de l’ordre de `10^-13`. Le post-traitement runtime `rigidBodyForces` fonctionne avec `p p_rgh` et produit des forces non nulles sur le patch hull. Au dernier pas, la force visqueuse est environ `(-85.321565, 1.356749, 0.397326) N` et le moment visqueux environ `(-0.287738, -17.528595, -16.090037) N m`; la contribution de pression est nulle dans ce smoke test car les champs donor/hull ont été initialisés uniformément. Le chemin runtime est donc validé, mais ces valeurs ne constituent pas encore une validation hydrodynamique DTC convergée.

## Cas DTC réaliste généré par FoamPilot

Le générateur `build_realistic_dtc_foampilot.py` utilise désormais la base Foundation 13 `DTCHullWave`, convertit `0/*.orig` en champs initiaux, conserve le domaine fluide complet, ajoute `marineProperties`, `sigma = 0.072`, et génère le mouvement 6-DoF par l’API FoamPilot avec les paramètres physiques du tutoriel : masse 412.73, inertie `(40 0 0 921 0 921)`, transformation `(2.929541 0 0.2)`, joints `Pz/Ry` et amortissements 8596/11586.

Le pipeline `Allmesh.FoamPilot` Foundation 13 termine avec succès : `surfaceFeatures`, `blockMesh`, `refineMesh`, `snappyHexMesh` et `renumberMesh`. `setFields` termine également avec succès et initialise l’eau sous le niveau `z = 0.244`.

Le calcul `marineFoam -solver incompressibleVoF` termine à `t = 2.60204e-4 s` sans erreur fatale. La fraction volumique reste bornée à 0/1, avec une fraction moyenne d’eau de 0.812159 et seulement des écarts numériques inférieurs à environ `6e-22` hors bornes. Les erreurs globales de continuité restent de l’ordre de `1e-12`. Le mouvement reste stable sur ce calcul court, avec une rotation et des translations finies.

`rigidBodyForces` produit des valeurs de pression et de cisaillement non nulles. Au dernier temps écrit, les composantes de force de pression sont environ `(24678.36, 26964.82, 103909.9) N` et les composantes visqueuses environ `(-3.03565, 0.00712, 0.06099) N`; le moment de pression est environ `(-18380.0, 363544.9, -196696.8) N m`. Ces résultats valident la chaîne physique eau/air, mouvement et forces pour un calcul court, mais une étude de convergence temporelle et de maillage reste nécessaire avant comparaison quantitative.
