# FoamPilot — PR #23 : cas marins et portage OpenFOAM Foundation 13

## Objet de la PR

Cette branche ajoute à FoamPilot une première architecture pour les calculs de mécanique navale sous **OpenFOAM Foundation 13**. Elle fournit les abstractions Python de génération de cas marins, un runner `marineFoam`, une bibliothèque C++ d’overset/inter-mailles, des modèles de propulsion MRF/actuation disk, ainsi que des cas et tests reproductibles.

La PR est ouverte sur [`feature/marine-pr`](https://github.com/stevendaix/foampilot/pull/23). Le commit de consolidation actuellement publié est `2c022bd`.

> La PR vise d’abord à rendre la chaîne de génération, de compilation et de calcul reproductible. La validation hydrodynamique définitive des trois références — Turning35, propeller et DTCMoving_Overset — nécessite encore des études de convergence en maillage et en temps.

## Principes d’architecture

Le code Python FoamPilot génère et valide les dictionnaires OpenFOAM. Les extensions C++ sont compilées séparément contre l’environnement Foundation 13. Les calculs utilisent les applications et API disponibles dans Foundation 13 ; les dictionnaires provenant d’OpenFOAM.com ou de versions plus anciennes ne sont pas utilisés sans adaptation.

L’overset n’étant pas fourni sous la même forme par Foundation 13, la PR introduit une implémentation expérimentale fondée sur `zoneID`, des stencils inter-mailles et une contrainte `fvConstraints` qui injecte les relations fortes dans les matrices. Le maillage et le calcul restent séparés conceptuellement : `snappyHexMesh` construit les patches et les volumes, tandis que `MarineOversetConstraint` applique le couplage au runtime.

## Arborescence de la PR

```text
.
├── foampilot/
│   ├── src/foampilot/mesh/
│   ├── src/foampilot/solver/
│   └── test/
├── openfoam13/
│   ├── marineFoam/
│   ├── marineOversetProbe/
│   ├── marine*Test/
│   ├── FoamPilotCases/
│   ├── DTCMoving_Overset_Foundation13/
│   └── build_*.py / export_*.py / prepare_*.py
├── examples/marine_config/
└── README_MARINE_PR23.md
```

## Fonction de chaque dossier

| Dossier | Fonction | Contenu principal |
|---|---|---|
| `foampilot/src/foampilot/mesh/` | API Python de maillage, mouvement et overset | `marine_motion.py`, `marine_mrf.py`, `marine_overset.py` |
| `foampilot/src/foampilot/solver/` | API Python des solveurs et modèles marins | `marine_case.py`, `marine_controls.py`, `marine_forces.py`, `marine_actuation_disk.py` |
| `foampilot/test/` | Tests unitaires et d’intégration Python | Tests des cas, du mouvement, de l’overset, MRF et propulsion |
| `openfoam13/marineFoam/` | Runner et solver modulaire Foundation 13 | `marineFoam.C`, `setDeltaT.C/H`, `Make/files`, `Make/options` |
| `openfoam13/marineOversetProbe/` | Bibliothèque C++ overset/inter-mailles | État des cellules, stencils, interpolation, matrices et contrainte `fvConstraints` |
| `openfoam13/marineInterMeshCouplingTest/` | Harness C++ de lecture et injection inter-mailles | Test de chargement des régions et stencils |
| `openfoam13/marineInterMeshStencilTest/` | Test du parseur et de l’état des stencils | Vérification du format `marineInterMeshStencils` |
| `openfoam13/marineOversetInterpolationTest/` | Test de l’interpolation overset | Vérification des coefficients et des cellules receveuses |
| `openfoam13/marineOversetMatrixTest/` | Test de contrainte forte dans une matrice | Cas OpenFOAM minimal avec `zoneID`, stencils et `fvConstraints` |
| `openfoam13/FoamPilotCases/` | Cas Foundation 13 générés ou convertis par FoamPilot | Cas DTC réaliste et propeller Foundation 13 |
| `openfoam13/DTCMoving_Overset_Foundation13/` | Pipeline de reproduction DTC avec coque et background séparés | Runners, maillages, stencils, préparation donor et post-traitement |
| `openfoam13/build_realistic_dtc_foampilot.py` | Générateur du cas DTC réaliste | Copie de `DTCHullWave`, génération des dictionnaires et validation FoamPilot |
| `openfoam13/build_dtc_intermesh_stencils.py` | Générateur de stencils inter-mailles DTC | Recherche donor, cellules receveuses et poids d’interpolation |
| `openfoam13/export_dtc_intermesh_dictionary.py` | Export du dictionnaire lu par le C++ | Conversion des stencils Python vers le format runtime |
| `examples/marine_config/` | Documentation d’état et exemples de configuration | État de l’intégration marine |

## API Python FoamPilot

### `marine_case.py`

`MarineCaseConfig` décrit les paramètres d’un cas marin, notamment le mode de calcul, le solver Foundation 13, la présence d’une surface libre, le mouvement et les modèles de propulsion. Les méthodes de validation vérifient la structure des répertoires et la présence des dictionnaires nécessaires avant l’exécution.

### `marine_motion.py`

Ce module génère le `dynamicMeshDict` Foundation 13 pour un corps rigide. Il prend en charge la masse, l’inertie, les joints, les distances overset, les coefficients d’amortissement et l’origine de transformation du corps. L’ajout de `transform_origin` permet de reproduire le placement utilisé par le tutoriel `DTCHullWave`.

### `marine_overset.py`

Ce module produit la configuration de zone overset, les identifiants de cellules et les stencils inter-mailles attendus par la bibliothèque C++. Il sépare la génération des données de couplage de leur application dans les matrices du solveur.

### `marine_mrf.py` et `marine_actuation_disk.py`

Ces modules génèrent respectivement les dictionnaires `MRFProperties` et `fvModels` pour les modèles Foundation 13 de rotation et de disque actuateur. Ils sont utilisés dans le cas propeller et dans les smoke tests MRF/actuation disk.

### `marine_forces.py` et `marine_controls.py`

Ils fournissent les fonctions de génération des sorties de forces/moments et les paramètres de contrôle de calcul, notamment `rigidBodyForces`, `p_rgh`, `deltaT`, les contrôles de Courant et les fonctions de post-traitement.

## Composants C++ Foundation 13

### `marineFoam`

`marineFoam` est le point d’entrée de calcul. Il charge un solver modulaire — actuellement `incompressibleVoF` pour la chaîne validée —, configure le contrôle du pas de temps, charge les modèles `fvModels` et `fvConstraints`, et permet de sélectionner la région donor selon la convention FoamPilot.

Compilation depuis un environnement Foundation 13 :

```bash
source /opt/openfoam13/etc/bashrc
cd openfoam13/marineOversetProbe
./Allwmake  # si le script local est présent

cd ../marineFoam
wmake
```

Les chemins de compilation exacts peuvent dépendre de l’installation locale de Foundation 13. Les fichiers `Make/files` et `Make/options` indiquent les dépendances attendues.

### `marineOversetProbe`

La bibliothèque contient les composants suivants :

| Composant | Rôle |
|---|---|
| `MarineOversetCellState` | Classification des cellules overset et états receveur/donor |
| `MarineInterMeshStencilState` | Lecture et conservation des stencils inter-mailles |
| `MarineOversetInterpolation` | Calcul/interpolation des valeurs donor vers les cellules receveuses |
| `MarineInterMeshMatrix` | Représentation du couplage entre régions |
| `MarineOversetMatrix` | Injection de contraintes fortes dans une matrice OpenFOAM |
| `MarineOversetConstraint` | Intégration runtime par `fvConstraints` |
| `marineOversetProbe` | Point d’entrée de la bibliothèque et enregistrement runtime |

## Cas DTC réaliste Foundation 13

Le cas `openfoam13/FoamPilotCases/DTCRealisticFoundation13` est basé sur le tutoriel Foundation 13 `DTCHullWave`. Il fournit un domaine fluide complet, le patch `hull`, les patchs d’entrée/sortie/atmosphère, les propriétés eau/air, une surface libre et un mouvement 6-DoF.

Le pipeline de reconstruction est :

```text
surfaceFeatures
    → blockMesh
    → refineMesh
    → snappyHexMesh
    → renumberMesh
    → setFields
    → marineFoam
    → rigidBodyForces
```

Commandes :

```bash
cd openfoam13
python3 build_realistic_dtc_foampilot.py
cd FoamPilotCases/DTCRealisticFoundation13
chmod +x Allmesh.FoamPilot
./Allmesh.FoamPilot
source /opt/openfoam13/etc/bashrc
setFields
marineFoam -solver incompressibleVoF
```

`setFields` initialise `alpha.water = 1` sous le niveau de surface libre et `alpha.water = 0` au-dessus. Les propriétés sont basées sur une eau de densité approximative `998,8 kg/m³`, une viscosité cinématique d’environ `1,09×10⁻⁶ m²/s`, un air de densité approximative `1 kg/m³`, et une tension de surface `sigma = 0,072 N/m`.

Les forces sont extraites avec `rigidBodyForces` sur le patch `hull`, en utilisant `p_rgh`. Le cas a atteint environ `2,6×10⁻⁴ s` dans un calcul court avec une fraction d’eau moyenne de `0,812159`, des erreurs globales de continuité de l’ordre de `10⁻¹²` et des forces/moments non nuls.

## Cas Turning35 Foundation 13

Le cas `openfoam13/FoamPilotCases/Turning35Foundation13` est le portage FoamPilot du cas de manœuvre `Turning35`. Il contient les géométries `hull.stl` et `rudder.stl`, les champs eau/air de départ, un mouvement rigide 6-DoF Foundation 13, une zone rotor `rotor`, le modèle de propulsion `actuationDisk`, ainsi que les sorties de forces et de moments.

Le pipeline documenté est :

```text
blockMesh
    → snappyHexMesh
    → topoSet (zone rotor)
    → setFields
    → marineFoam -solver incompressibleVoF
    → rigidBodyForces
```

Le cas est actuellement une validation mono-région du mouvement, de la surface libre et de la propulsion. Le couplage overset/inter-mailles n’est pas présenté comme validé sur Turning35 ; il reste couvert par les harnesses `marineInterMesh*` et le cas DTC multi-région. La stabilité numérique et les valeurs hydrodynamiques finales nécessitent encore une exécution Foundation 13 complète et une étude de convergence.

## Cas propeller Foundation 13

Le cas `openfoam13/FoamPilotCases/propellerFoundation13` reprend la structure du tutoriel Foundation 13 `incompressibleVoF/propeller`. Il contient notamment :

| Fichier ou dossier | Fonction |
|---|---|
| `constant/MRFProperties` | Définition de la zone rotor et de la vitesse angulaire |
| `constant/fvModels` | Modèles de propulsion, dont `actuationDisk` |
| `constant/dynamicMeshDict` | Mouvement ou rotation du domaine associé |
| `system/snappyHexMeshDict` | Raffinement de la géométrie et création des patches |
| `system/functions` | Inclusion des sorties de post-traitement |
| `system/forces` | Extraction des forces et moments |
| `Allmesh.FoamPilot` | Reconstruction séquentielle du maillage |

Le smoke test FoamPilot a produit un maillage d’environ `525 586` cellules et un calcul court réussi avec MRF/actuation disk. Une extraction de forces et moments a été produite au premier pas. Ces valeurs doivent être recalculées sur plusieurs tours lorsque le maillage rotor/stator final et les interfaces AMI sont stabilisés.

## Pipeline DTC overset custom

Le dossier `openfoam13/DTCMoving_Overset_Foundation13` contient le pipeline expérimental complet pour une coque mobile et un background séparé. Le sous-cas `hull` est maillé avec `snappyHexMesh`, qui crée réellement le patch `hull`. Le sous-cas background fournit le domaine externe et les cellules donor.

La région hull doit être initialisée avant de lancer le calcul. La convention vérifiée par le runner est :

```text
hull       = région receveuse
background = région donor
```

Les stencils sont calculés avant l’exécution et la contrainte `MarineOversetConstraint` est chargée par `fvConstraints`. Le cas permet de tester séparément la lecture des régions, le sens donor/receveur, l’application des contraintes et l’extraction des forces sur `hull`.

## Tests

### Tests Python

Depuis la racine du dépôt :

```bash
cd foampilot
PYTHONPATH=src pytest -q
```

La suite marine/Foundation 13 validée pendant cette PR comprend les tests de cas, mouvement, MRF, actuation disk, forces, contrôles, overset et solver. Les fixtures historiques absentes doivent être distinguées des régressions marines lors de l’analyse de la suite complète.

### Tests C++ Foundation 13

Les tests C++ se trouvent dans les dossiers `marineInterMesh*` et `marineOverset*`. Ils vérifient notamment :

1. la compilation et l’enregistrement runtime de la bibliothèque ;
2. la lecture de `zoneID` et des stencils ;
3. l’injection de contraintes dans les matrices ;
4. l’orientation donor/receveur ;
5. le couplage inter-mailles sur le cas DTC minimal.

### Validation d’un cas généré

```bash
cd foampilot
PYTHONPATH=src python test/run_foundation13_case_validation.py
PYTHONPATH=src python test/validate_converted_foundation13.py
```

## Fichiers générés exclus du dépôt

Les sorties de calcul ne doivent pas être commités. Le `.gitignore` exclut notamment `postProcessing`, `polyMesh` générés, les temps numériques, les logs, les fichiers de compilation et les caches Python. Les scripts `Allmesh`, `Allrun`, `build_*.py` et les dictionnaires sources doivent permettre de reconstruire ces fichiers.

## Limites connues

Le portage direct de cfMesh v2406 contre Foundation 13 n’est pas retenu comme dépendance de calcul, car cfMesh utilise des API internes incompatibles, notamment l’ancien mécanisme de gestion de capacité de `UList`. Les pipelines Foundation natifs `blockMesh`, `surfaceFeatures`, `refineMesh` et `snappyHexMesh` sont donc privilégiés.

Le solver `marineFoam` validé dans la chaîne actuelle est `incompressibleVoF`. Le nom `compressibleVoF` présent dans certains squelettes historiques ne doit pas être utilisé tant qu’un module correspondant n’est pas effectivement exposé et compilé dans Foundation 13.

Le cas DTC réaliste possède maintenant une surface libre et produit des forces non nulles, mais le calcul reste court. Une validation scientifique complète demande une étude de sensibilité au pas de temps, au raffinement proche coque, au modèle de turbulence, au niveau de surface libre et aux paramètres de mouvement 6-DoF.

## Références

[1]: https://github.com/stevendaix/foampilot/pull/23 "FoamPilot — PR #23"
[2]: https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials/incompressibleVoF/DTCHullWave "OpenFOAM Foundation 13 — DTCHullWave"
[3]: https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials/incompressibleVoF/propeller "OpenFOAM Foundation 13 — incompressibleVoF propeller"
[4]: https://github.com/balabibo/maneuveringLib "maneuveringLib — cas de manœuvre"
[5]: https://github.com/skfelix/propeller-OpenFOAM "propeller-OpenFOAM — cas hélice"
[6]: https://github.com/myozinaung/DTCMoving_Overset "DTCMoving_Overset — cas overset DTC"
