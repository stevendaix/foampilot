
- [Architecture du module](#architecture-du-module)
  - [foampilot.base](#foampilotbase)
  - [foampilot.boundaries](#foampilotboundaries)
  - [foampilot.commons](#foampilotcommons)
  - [foampilot.constant](#foampilotconstant)
  - [foampilot.mesh](#foampilotmesh)
  - [foampilot.postprocess](#foampilotpostprocess)
  - [foampilot.report](#foampilotreport)
  - [foampilot.solver](#foampilotsolver)
  - [foampilot.system](#foampilotsystem)
  - [foampilot.utilities](#foampilotutilities)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Améliorations et Contributions](#améliorations-et-contributions)

## Introduction

`foampilot` est une plateforme Python qui simplifie et automatise le processus de configuration, d'exécution et de post-traitement des simulations OpenFOAM. Elle permet aux utilisateurs de définir l'intégralité de leur workflow de simulation en Python, offrant une reproductibilité accrue et un gain de temps significatif.

## Architecture du module

Le module `foampilot` est structuré en plusieurs sous-modules, chacun gérant un aspect spécifique du workflow OpenFOAM.

### foampilot.base

Ce sous-module contient les classes de base pour la gestion des fichiers OpenFOAM et les opérations de maillage.

### foampilot.boundaries

Gère la définition et la manipulation des conditions aux limites.

### foampilot.commons

Contient des utilitaires communs, notamment pour la lecture des fichiers de maillage.

### foampilot.constant

Gère le répertoire `constant` d'OpenFOAM, y compris les propriétés de transport et de turbulence.

### foampilot.mesh

Contient les classes pour la création et la gestion des maillages, notamment `blockMesh`.

### foampilot.postprocess

Fournit des outils pour le post-traitement des résultats de simulation, y compris la visualisation avec PyVista.

### foampilot.report

Gère la génération de rapports automatiques, par exemple au format PDF.

### foampilot.solver

Contient les classes pour l'exécution des solveurs OpenFOAM, comme `incompressibleFluid`.

### foampilot.system

Gère le répertoire `system` d'OpenFOAM, y compris les fichiers `controlDict`, `fvSchemes` et `fvSolution`.

### foampilot.utilities

Ce sous-module regroupe diverses fonctions utilitaires.

## Installation

Pour installer `foampilot`, clonez le dépôt GitHub et installez les dépendances nécessaires :

```bash
git clone https://github.com/stevendaix/foampilot.git
cd foampilot
pip install -r requirements.txt # (Assurez-vous que ce fichier existe ou créez-le)
```

## Utilisation

(Cette section sera complétée avec des exemples de code et des explications détaillées sur l'utilisation des différentes classes et fonctions du module.)

## Exemples

(Cette section présentera des exemples concrets d'utilisation du module pour des cas de simulation typiques.)

## Améliorations et Contributions

(Cette section décrira comment contribuer au projet et les améliorations futures envisagées.)




### `foampilot.base.meshing.Meshing`

La classe `Meshing` gère le processus de maillage pour un cas OpenFOAM. Elle permet de configurer et d'exécuter `blockMesh`.

**Constructeur:**

```python
Meshing(path_case)
```

- `path_case` (str ou Path): Le chemin vers le répertoire du cas OpenFOAM.

**Méthodes:**

- `add_file(file_name, file_content)`:
  Ajoute un fichier additionnel au processus de maillage.
  - `file_name` (str): Le nom du fichier à ajouter.
  - `file_content` (dict): Le contenu du fichier.

- `load_from_json(json_path)`:
  Charge la configuration du maillage à partir d'un fichier JSON et met à jour `blockMeshDict`.
  - `json_path` (str ou Path): Le chemin vers le fichier JSON.

- `write()`:
  Écrit les fichiers dans leurs répertoires respectifs au sein du répertoire `system`.

- `run_blockMesh()`:
  Exécute la commande `blockMesh` dans le chemin du cas spécifié et enregistre la sortie.
  - **Lève:** `FileNotFoundError`, `NotADirectoryError`, `RuntimeError` en cas d'échec.





### `foampilot.base.openFOAMFile.OpenFOAMFile`

`OpenFOAMFile` est une classe de base pour la création et la gestion des fichiers de configuration OpenFOAM. Elle gère l'en-tête standard `FoamFile` et l'écriture des attributs.

**Constructeur:**

```python
OpenFOAMFile(object_name, **attributes)
```

- `object_name` (str): Le nom de l'objet pour l'en-tête du fichier OpenFOAM.
- `**attributes`: Arguments mots-clés arbitraires représentant les attributs spécifiques du fichier.

**Méthodes:**

- `write(filepath)`:
  Écrit le contenu du fichier OpenFOAM dans le chemin spécifié.
  - `filepath` (str ou Path): Le chemin vers le fichier à écrire.





### `foampilot.boundaries.boundaries_dict.Boundary`

La classe `Boundary` gère les conditions aux limites pour les simulations OpenFOAM. Elle permet de définir et de manipuler les conditions aux limites pour divers champs (vitesse, pression, paramètres de turbulence).

**Constructeur:**

```python
Boundary(parent, turbulence_model="kEpsilon")
```

- `parent`: L'objet parent du cas OpenFOAM.
- `turbulence_model` (str): Le modèle de turbulence à utiliser (par défaut : "kEpsilon").

**Méthodes:**

- `load_boundary_names(case_path)`:
  Charge les noms et types des limites à partir du fichier `polyMesh/boundary`.
  - `case_path` (Path): Le chemin vers le répertoire du cas OpenFOAM.
  - **Retourne:** Un dictionnaire mappant les noms des patchs à leurs types.
  - **Lève:** `FileNotFoundError` si le fichier de limites n'existe pas.

- `initialize_boundary()`:
  Initialise les champs de limites avec les noms des patchs et applique automatiquement les conditions de paroi.

- `apply_condition_with_wildcard(field, pattern, condition)`:
  Applique une condition à toutes les limites correspondant à un motif.
  - `field`: Le champ auquel appliquer la condition (par exemple, "U", "p").
  - `pattern`: Motif d'expression régulière pour faire correspondre les noms des limites.
  - `condition`: Dictionnaire contenant les paramètres de la condition aux limites.

- `set_velocity_inlet(pattern, velocity, turbulence_intensity=None)`:
  Définit une condition aux limites d'entrée de vitesse.
  - `pattern`: Motif d'expression régulière.
  - `velocity`: Tuple de 3 objets `ValueWithUnit` (u, v, w) représentant les composantes de vitesse.
  - `turbulence_intensity` (float, optionnel): Intensité de turbulence entre 0 et 1.
  - **Lève:** `ValueError` si les unités de vitesse sont incorrectes.

- `set_pressure_inlet(pattern, pressure, turbulence_intensity=None)`:
  Définit une condition aux limites d'entrée de pression.
  - `pattern`: Motif d'expression régulière.
  - `pressure`: Valeur de pression en tant qu'objet `ValueWithUnit`.
  - `turbulence_intensity` (float, optionnel): Intensité de turbulence entre 0 et 1.
  - **Lève:** `ValueError` si les unités de pression sont incorrectes.

- `set_pressure_outlet(pattern, velocity)`:
  Définit une condition aux limites de sortie de pression.
  - `pattern`: Motif d'expression régulière.
  - `velocity`: Tuple de 3 objets `ValueWithUnit` (u, v, w) représentant les composantes de vitesse.

- `set_mass_flow_inlet(pattern, mass_flow_rate, density)`:
  Définit une condition aux limites d'entrée de débit massique.
  - `pattern`: Motif d'expression régulière.
  - `mass_flow_rate`: Débit massique en tant qu'objet `ValueWithUnit`.
  - `density`: Densité en tant qu'objet `ValueWithUnit`.
  - **Lève:** `ValueError` si les unités sont incorrectes.

- `set_wall(pattern, friction=True, velocity=None)`:
  Définit une condition aux limites de paroi.
  - `pattern`: Motif d'expression régulière.
  - `friction` (bool): Si `True` (par défaut), utilise la condition de non-glissement. Si `False`, utilise la condition de glissement.
  - `velocity` (tuple, optionnel): Tuple de 3 objets `ValueWithUnit` pour une paroi à vitesse fixe.

- `set_symmetry(pattern)`:
  Définit une condition aux limites de symétrie (vide).
  - `pattern`: Motif d'expression régulière.

- `set_no_friction_wall(pattern)`:
  Définit une condition aux limites de paroi sans frottement (glissement).
  - `pattern`: Motif d'expression régulière.

- `get_wall_function(field, vel_cond)`:
  Obtient la fonction de paroi appropriée pour un champ donné.
  - `field` (str): Le nom du champ ("k", "epsilon" ou "omega").
  - `vel_cond` (bool): Indique si une condition de vitesse est appliquée.
  - **Retourne:** Un dictionnaire contenant les paramètres de la fonction de paroi.

- `set_uniform_normal_fixed_value_all_fields(patch_pattern, mode="intakeType3", ref_value=1.2)`:
  Définit la condition `uniformNormalFixedValue` ou `surfaceNormalFixedValue` sur tous les champs.
  - `patch_pattern`: Motif d'expression régulière.
  - `mode` (str): Type de condition à appliquer.
  - `ref_value` (float): Valeur de référence pour la condition.
  - **Lève:** `ValueError` si le mode est invalide.





### `foampilot.commons.read_polymesh.BoundaryFileHandler`

La classe `BoundaryFileHandler` gère les fichiers `boundary` d'OpenFOAM situés dans `constant/polyMesh/`. Elle fournit des utilitaires pour analyser, mettre à jour et écrire le dictionnaire `boundary` qui définit les patchs de maillage dans OpenFOAM.

**Constructeur:**

```python
BoundaryFileHandler(base_path, all_data=False)
```

- `base_path` (str ou Path): Chemin vers le répertoire racine du cas OpenFOAM (doit contenir `constant/polyMesh/boundary`).
- `all_data` (bool, optionnel): Si `True`, tous les attributs des patchs sont extraits (nFaces, startFace, etc.). Par défaut à `False`.
- **Lève:** `FileNotFoundError` si le fichier `boundary` n'existe pas.

**Méthodes:**

- `comment_remover(text)`:
  Supprime les commentaires de ligne (`//`) et de bloc (`/* */`) d'un texte tout en préservant les littéraux de chaîne.
  - `text` (str): Texte d'entrée duquel les commentaires doivent être supprimés.
  - **Retourne:** Le texte avec les commentaires supprimés.

- `parse_boundary_file(all_data)`:
  Analyse un fichier `boundary` d'OpenFOAM et extrait les patchs.
  - `all_data` (bool): Si `True`, extrait les attributs détaillés des patchs.
  - **Retourne:** Un dictionnaire contenant les patchs et leurs attributs.

- `write_boundary_file(data=None)`:
  Écrit les données de patch dans le fichier `boundary` d'OpenFOAM.
  - `data` (dict, optionnel): Dictionnaire contenant les données de patch à écrire. Si `None`, utilise `self.data`.

- `update_patch(patch_name, patch_data)`:
  Met à jour ou ajoute un patch dans les données de limite.
  - `patch_name` (str): Nom du patch à mettre à jour ou à ajouter.
  - `patch_data` (dict): Dictionnaire d'attributs pour le patch.





### `foampilot.constant.constantDirectory.ConstantDirectory`

La classe `ConstantDirectory` représente le répertoire `constant` dans un cas OpenFOAM, qui contient des fichiers tels que `transportProperties` et `turbulenceProperties`.

**Constructeur:**

```python
ConstantDirectory(parent)
```

- `parent`: L'objet parent du cas OpenFOAM.

**Méthodes:**

- `write()`:
  Écrit les fichiers à leurs emplacements respectifs dans le répertoire `constant`.





### `foampilot.constant.transportPropertiesFile.TransportPropertiesFile`

La classe `TransportPropertiesFile` représente le fichier de configuration OpenFOAM `transportProperties`. Ce fichier définit le modèle de transport et les propriétés du fluide, telles que la viscosité, nécessaires aux simulations CFD.

**Constructeur:**

```python
TransportPropertiesFile(transportModel="Newtonian", nu="1e-05")
```

- `transportModel` (str, optionnel): Le modèle de transport utilisé dans la simulation (par défaut : `"Newtonian"`).
- `nu` (str, optionnel): La viscosité cinématique (m²/s) sous forme de chaîne de caractères (par défaut : `"1e-05"`).

**Notes:**

- Cette classe hérite de `foampilot.base.openFOAMFile.OpenFOAMFile`.
- Les paramètres sont stockés au format dictionnaire OpenFOAM.





### `foampilot.constant.turbulencePropertiesFile.TurbulencePropertiesFile`

La classe `TurbulencePropertiesFile` représente un fichier de configuration OpenFOAM `turbulenceProperties`. Cette classe permet de configurer les modèles de turbulence, le type de simulation et les paramètres associés.

**Attributs:**

- `AVAILABLE_MODELS` (dict): Dictionnaire mappant les noms de modèles conviviaux aux codes OpenFOAM internes.

**Constructeur:**

```python
TurbulencePropertiesFile(simulationType="RAS", RASModel="k-epsilon", turbulence="on", printCoeffs="on")
```

- `simulationType` (str, optionnel): Le type de simulation de turbulence (par défaut : `"RAS"`).
- `RASModel` (str, optionnel): Le modèle de turbulence à utiliser. Doit être dans `AVAILABLE_MODELS` ou un modèle personnalisé (par défaut : `"k-epsilon"`).
- `turbulence` (str, optionnel): Indique si la turbulence est activée (`"on"`/`"off"`) (par défaut : `"on"`).
- `printCoeffs` (str, optionnel): Indique s'il faut imprimer les coefficients du modèle (`"on"`/`"off"`) (par défaut : `"on"`).

**Méthodes de classe:**

- `add_turbulence_model(model_name, model_code)`:
  Ajoute un nouveau modèle de turbulence au dictionnaire des modèles disponibles.
  - `model_name` (str): Nom convivial pour le modèle.
  - `model_code` (str): Code interne pour le modèle dans OpenFOAM.





### `foampilot.mesh.BlockMeshFile.BlockMeshFile`

La classe `BlockMeshFile` représente le fichier `blockMeshDict` dans OpenFOAM. Elle permet de construire, modifier et exporter ce fichier qui définit la topologie du maillage pour les simulations OpenFOAM.

**Attributs:**

- `scale` (float): Facteur d'échelle appliqué au maillage.
- `vertices` (list de listes): Liste des coordonnées des sommets (x, y, z).
- `blocks` (list): Liste des définitions de blocs.
- `edges` (list): Liste des définitions d'arêtes.
- `defaultPatch` (dict): Définition du patch par défaut.
- `boundary` (dict): Dictionnaire des patchs de limite et de leurs conditions.
- `mergePatchPairs` (list de tuples): Liste des paires de patchs à fusionner.

**Constructeur:**

```python
BlockMeshFile(scale=1, vertices=None, blocks=None, edges=None, defaultPatch=None, boundary=None, mergePatchPairs=None)
```

- `scale` (float, optionnel): Facteur d'échelle du maillage (par défaut : 1).
- `vertices` (list de listes, optionnel): Liste des sommets, chacun sous la forme `[x, y, z]` (par défaut : liste vide).
- `blocks` (list, optionnel): Liste des définitions de blocs (par défaut : liste vide).
- `edges` (list, optionnel): Liste des arêtes (par défaut : liste vide).
- `defaultPatch` (dict, optionnel): Définition du patch par défaut (par défaut : dictionnaire vide).
- `boundary` (dict, optionnel): Définitions des limites, par exemple `{"inlet": {"type": "patch", "faces": [...]}}`.
- `mergePatchPairs` (list de tuples, optionnel): Liste des paires de patchs à fusionner (par défaut : liste vide).

**Méthodes:**

- `load_from_json(json_path)`:
  Charge une configuration `blockMeshDict` à partir d'un fichier JSON.
  - `json_path` (str): Chemin vers le fichier de configuration JSON.
  - **Lève:** `FileNotFoundError` si le fichier JSON n'existe pas, `KeyError` si la structure du fichier JSON est invalide ou si des clés requises sont manquantes.

- `write(file_path)`:
  Écrit le contenu de `blockMeshDict` dans un fichier.
  - `file_path` (Path): Chemin de destination du fichier `blockMeshDict`.





### `foampilot.mesh.test_snappymesh.STLAnalyzer`

La classe `STLAnalyzer` est conçue pour analyser les fichiers STL, extraire des informations géométriques et préparer les données pour le maillage.

**Constructeur:**

```python
STLAnalyzer(filename)
```

- `filename` (Path): Chemin vers le fichier STL.

**Attributs:**

- `filename` (Path): Chemin vers le fichier STL.
- `mesh` (pv.PolyData): Objet maillage PyVista représentant le contenu du fichier STL.
- `reader` (pyvista.reader): L'objet lecteur utilisé pour lire le fichier STL.

**Méthodes:**

- `load()`:
  Charge le maillage à partir du fichier STL.
  - **Retourne:** `pv.PolyData` : Objet maillage PyVista.

- `get_max_dim()`:
  Retourne la dimension maximale du maillage STL basée sur la boîte englobante.
  - **Retourne:** `float` : Dimension maximale.

- `get_min_dim()`:
  Retourne la dimension minimale du maillage STL basée sur la boîte englobante.
  - **Retourne:** `float` : Dimension minimale.

- `calc_domain_size(sizeFactor=1.0)`:
  Calcule la taille du domaine basée sur la boîte englobante STL et un facteur de taille.
  - `sizeFactor` (float): Facteur pour redimensionner le domaine.
  - **Retourne:** `tuple` : Taille du domaine dans chaque dimension.

- `get_info()`:
  Obtient des informations sur le maillage, y compris les dimensions max/min et la taille du domaine.
  - **Retourne:** `dict` : Dictionnaire contenant des informations sur le maillage.
  - **Lève:** `ValueError` si le maillage n'a pas été chargé.

- `extract_features()`:
  Exécute `surfaceFeatureExtract` pour générer le fichier `.eMesh` avec les caractéristiques des arêtes.

- `get_center_of_mass()`:
  Calcule le centre de masse du maillage STL.
  - **Retourne:** `tuple` : Coordonnées du centre de masse (x, y, z).
  - **Lève:** `ValueError` si le maillage n'a pas été chargé.





### `foampilot.mesh.test_snappymesh.SnappyHexMesh`

La classe `SnappyHexMesh` est utilisée pour configurer et générer le fichier `snappyHexMeshDict` basé sur la géométrie STL.

**Constructeur:**

```python
SnappyHexMesh(base_path, stl_file, castellatedMesh=True, snap=True, addLayers=False)
```

- `base_path` (Path): Chemin vers le répertoire du cas OpenFOAM.
- `stl_file` (Path): Chemin vers le fichier de géométrie STL.
- `castellatedMesh` (bool): Active la structure de maillage `castellated` initiale (par défaut : `True`).
- `snap` (bool): Active la projection du maillage sur la surface STL (par défaut : `True`).
- `addLayers` (bool): Active l'ajout de couches limites (par défaut : `False`).

**Attributs:**

- `base_path` (Path): Chemin vers le répertoire du cas OpenFOAM.
- `stl_file` (Path): Chemin vers le fichier STL.
- `snappy_hex_mesh_dict_path` (Path): Chemin vers le fichier `snappyHexMeshDict`.
- `locationInMesh` (tuple): Coordonnées d'un point à l'intérieur du maillage.
- `castellatedMesh` (bool): Indique si le maillage `castellated` est activé.
- `snap` (bool): Indique si le `snap` est activé.
- `addLayers` (bool): Indique si l'ajout de couches est activé.
- `geometry` (dict): Définition de la géométrie pour `snappyHexMesh`.
- `castellatedMeshControls` (dict): Contrôles pour le maillage `castellated`.
- `snapControls` (dict): Contrôles pour l'étape de `snap`.
- `addLayersControls` (dict): Contrôles pour l'ajout de couches limites.
- `meshQualityControls` (dict): Contrôles de qualité du maillage.
- `debugFlags` (list): Drapeaux de débogage.
- `writeFlags` (list): Drapeaux d'écriture.

**Méthodes:**

- `add_feature(feature_file, level)`:
  Ajoute un fichier d'arêtes de caractéristique (extrait avec `surfaceFeatureExtract`) pour affiner les arêtes de la géométrie.
  - `feature_file` (str): Chemin vers le fichier `.eMesh`.
  - `level` (int): Niveau de raffinement pour les caractéristiques d'arêtes.

- `add_refinement_region(name, mode, levels)`:
  Ajoute une région de raffinement spécifique.
  - `name` (str): Nom de la région dans la géométrie.
  - `mode` (str): Mode de raffinement (par exemple, `'inside'`, `'outside'`).
  - `levels` (tuple): Niveaux de raffinement pour la région (par exemple, `((1, 2))`).

- `add_layer(surface, n_surface_layers)`:
  Définit le nombre de couches de maillage autour d'une surface spécifique.
  - `surface` (str): Nom de la surface.
  - `n_surface_layers` (int): Nombre de couches de surface.

- `write_snappyHexMeshDict()`:
  Génère le fichier `snappyHexMeshDict` avec les options définies pour `snappyHexMesh`.





### `foampilot.postprocess.openfoam_pyvista.FoamPostProcessing`

La classe `FoamPostProcessing` est une classe de post-traitement pour les cas OpenFOAM. Elle permet de convertir les données OpenFOAM en fichiers VTK, de charger et de visualiser les maillages, et de calculer diverses quantités.

**Constructeur:**

```python
FoamPostProcessing(case_path, vtk_dir="VTK")
```

- `case_path` (str): Chemin vers le répertoire du cas OpenFOAM.
- `vtk_dir` (str): Nom du répertoire pour la sortie VTK (par défaut : `"VTK"`).

**Méthodes:**

- `check_case()`:
  Vérifie si le chemin du cas OpenFOAM existe et est un répertoire.
  - **Lève:** `FileNotFoundError` si le chemin n'existe pas ou n'est pas un répertoire.

- `foamToVTK(all_regions=False, ascii=False, constant=False, latest_time=False, fields=None, no_boundary=False, no_internal=False)`:
  Convertit le cas OpenFOAM en fichiers VTK à l'aide de `foamToVTK`.
  - `all_regions` (bool): Convertit toutes les régions (par défaut : `False`).
  - `ascii` (bool): Exporte en format ASCII (par défaut : `False`).
  - `constant` (bool): Exporte les champs constants (par défaut : `False`).
  - `latest_time` (bool): Exporte uniquement le dernier pas de temps (par défaut : `False`).
  - `fields` (list ou str): Champs à exporter (par défaut : `None`, tous les champs).
  - `no_boundary` (bool): N'exporte pas les limites (par défaut : `False`).
  - `no_internal` (bool): N'exporte pas le maillage interne (par défaut : `False`).
  - **Lève:** `RuntimeError` si `foamToVTK` échoue.

- `list_time_steps()`:
  Retourne une liste triée des pas de temps disponibles basée sur les fichiers VTK dans le répertoire principal.
  - **Retourne:** `list` : Liste des pas de temps.

- `get_structure(time_step=None)`:
  Construit un dictionnaire avec le maillage principal (cellule) et toutes les limites trouvées automatiquement dans le dossier VTK.
  - `time_step` (int, optionnel): Le pas de temps à charger. Si `None`, le dernier pas de temps est utilisé.
  - **Retourne:** `dict` : Dictionnaire contenant le maillage des cellules et les maillages des limites.
  - **Lève:** `FileNotFoundError` si aucun fichier VTK n'est trouvé ou si le fichier de cellule n'existe pas.

- `load_time_step(time_step)`:
  Charge les données VTK pour un pas de temps spécifique.
  - `time_step` (int): Le pas de temps à charger.
  - **Retourne:** `dict` : Dictionnaire contenant le maillage des cellules et les maillages des limites.

- `get_all_time_steps()`:
  Retourne tous les pas de temps disponibles.
  - **Retourne:** `list` : Liste de tous les pas de temps.

- `plot_slice(structure=None, plane="z", scalars="U", opacity=0.25)`:
  Génère un tracé de tranche à partir du dictionnaire de structure donné.
  - `structure` (dict, optionnel): Dictionnaire de structure du maillage. Si `None`, lève une `RuntimeError`.
  - `plane` (str): Le plan de la tranche ("x", "y" ou "z") (par défaut : `"z"`).
  - `scalars` (str): Le champ scalaire à afficher (par défaut : `"U"`).
  - `opacity` (float): L'opacité du maillage (par défaut : `0.25`).

- `plot_contour(mesh, scalars, is_filled=True, opacity=1.0)`:
  Génère un tracé de contour.
  - `mesh`: L'objet maillage PyVista.
  - `scalars` (str): Le champ scalaire à afficher.
  - `is_filled` (bool): Si `True`, le contour est rempli (par défaut : `True`).
  - `opacity` (float): L'opacité du tracé (par défaut : `1.0`).

- `plot_vectors(mesh, vectors, scale=1.0, color='blue')`:
  Génère un tracé de vecteurs.
  - `mesh`: L'objet maillage PyVista.
  - `vectors` (str): Le champ vectoriel à afficher.
  - `scale` (float): Facteur d'échelle pour les vecteurs (par défaut : `1.0`).
  - `color` (str): Couleur des vecteurs (par défaut : `"blue"`).
  - **Lève:** `ValueError` si le champ vectoriel n'est pas trouvé.

- `plot_streamlines(mesh, vectors, n_points=100, max_time=10.0)`:
  Génère des lignes de courant.
  - `mesh`: L'objet maillage PyVista.
  - `vectors` (str): Le champ vectoriel à afficher.
  - `n_points` (int): Nombre de points de départ pour les lignes de courant (par défaut : `100`).
  - `max_time` (float): Temps maximal pour le calcul des lignes de courant (par défaut : `10.0`).

- `plot_mesh_style(mesh, style='surface', show_edges=False, color='white', opacity=1.0)`:
  Visualise le maillage avec différents styles.
  - `mesh`: L'objet maillage PyVista.
  - `style` (str): Style de visualisation (`'surface'`, `'wireframe'`, etc.) (par défaut : `'surface'`).
  - `show_edges` (bool): Affiche les arêtes (par défaut : `False`).
  - `color` (str): Couleur du maillage (par défaut : `'white'`).
  - `opacity` (float): Opacité du maillage (par défaut : `1.0`).

- `calculate_q_criterion(mesh, velocity_field='U')`:
  Calcule le critère Q.
  - `mesh`: L'objet maillage PyVista.
  - `velocity_field` (str): Le champ de vitesse (par défaut : `'U'`).
  - **Retourne:** Le maillage avec le critère Q ajouté aux données de points.
  - **Lève:** `ValueError` si le champ de vitesse n'est pas trouvé.

- `calculate_vorticity(mesh, velocity_field='U')`:
  Calcule la vorticité.
  - `mesh`: L'objet maillage PyVista.
  - `velocity_field` (str): Le champ de vitesse (par défaut : `'U'`).
  - **Retourne:** Le maillage avec la vorticité ajoutée aux données de points.
  - **Lève:** `ValueError` si le champ de vitesse n'est pas trouvé.

- `export_plot(plotter, filename, image_format="png")`:
  Exporte le tracé actuel vers un fichier image.
  - `plotter`: L'objet de traçage (par exemple, `pyvista.Plotter`).
  - `filename` (Path): Nom du fichier (avec ou sans extension).
  - `image_format` (str): Format de l'image (par défaut : `"png"`).

- `create_animation(scalars, filename, image_format='gif', fps=10)`:
  Crée une animation à travers les pas de temps.
  - `scalars` (str): Le champ scalaire à animer.
  - `filename` (str): Nom du fichier de sortie de l'animation.
  - `image_format` (str): Format de l'image (par défaut : `'gif'`).
  - `fps` (int): Images par seconde (par défaut : `10`).
  - **Lève:** `FileNotFoundError` si aucun fichier VTK n'est trouvé pour l'animation.

- `get_scalar_statistics(mesh, scalar_field)`:
  Calcule les statistiques (moyenne, écart-type, min, max) pour un champ scalaire.
  - `mesh`: L'objet maillage PyVista.
  - `scalar_field` (str): Le champ scalaire.
  - **Retourne:** `dict` : Dictionnaire des statistiques.
  - **Lève:** `ValueError` si le champ scalaire n'est pas trouvé.

- `get_time_series_data(scalar_field, point_coordinates)`:
  Extrait les données de série temporelle pour un champ scalaire à un point spécifique.
  - `scalar_field` (str): Le champ scalaire.
  - `point_coordinates` (list): Coordonnées du point.
  - **Retourne:** `dict` : Dictionnaire contenant les pas de temps et les données de série temporelle.
  - **Lève:** `FileNotFoundError` si aucun fichier VTK n'est trouvé pour l'analyse de série temporelle, `ValueError` si le champ scalaire n'est pas trouvé.

- `get_mesh_statistics(mesh)`:
  Retourne les statistiques sur le maillage lui-même (par exemple, nombre de points, cellules).
  - `mesh`: L'objet maillage PyVista.
  - **Retourne:** `dict` : Dictionnaire des statistiques du maillage.

- `get_region_statistics(structure, region_name, scalar_field)`:
  Calcule les statistiques pour un champ scalaire dans une région spécifique (cellule ou limite).
  - `structure`: Dictionnaire de structure du maillage.
  - `region_name` (str): Nom de la région ("cell" ou nom de la limite).
  - `scalar_field` (str): Le champ scalaire.
  - **Retourne:** `dict` : Dictionnaire des statistiques de la région.
  - **Lève:** `ValueError` si la région ou le champ scalaire n'est pas trouvé.





### `foampilot.report.latex_pdf.LatexDocument`

La classe `LatexDocument` facilite la création de documents LaTeX, y compris l'ajout de sections, de figures, de tableaux et la génération de PDF.

**Constructeur:**

```python
LatexDocument(title, author, filename)
```

- `title` (str): Le titre du document.
- `author` (str): L'auteur du document.
- `filename` (str): Le nom de base du fichier de sortie (sans extension).

**Méthodes:**

- `add_title()`:
  Ajoute la page de titre au document.

- `add_toc()`:
  Ajoute la table des matières au document.

- `add_section(title, content)`:
  Ajoute une nouvelle section au document.
  - `title` (str): Le titre de la section.
  - `content` (str): Le contenu de la section.

- `add_subsection(title, content)`:
  Ajoute une nouvelle sous-section au document.
  - `title` (str): Le titre de la sous-section.
  - `content` (str): Le contenu de la sous-section.

- `add_unnumbered_section(title, content)`:
  Ajoute une section non numérotée au document.
  - `title` (str): Le titre de la section non numérotée.
  - `content` (str): Le contenu de la section.

- `add_abstract(content)`:
  Ajoute un résumé au document.
  - `content` (str): Le contenu du résumé.

- `add_list(items, ordered=False)`:
  Ajoute une liste (ordonnée ou non) au document.
  - `items` (list): Une liste de chaînes de caractères pour les éléments de la liste.
  - `ordered` (bool): Si `True`, la liste est ordonnée (par défaut : `False`).

- `add_math(equation)`:
  Ajoute une équation mathématique au document.
  - `equation` (str): Chaîne formatée LaTeX représentant l'équation.

- `add_figure(image_path, caption=None, width=\'0.8\\textwidth\')`:
  Ajoute une figure au document avec une légende et une largeur optionnelles.
  - `image_path` (str): Chemin vers le fichier image.
  - `caption` (str, optionnel): Légende de l'image.
  - `width` (str): Largeur de l'image (par défaut : `\'0.8\\textwidth\'`).

- `add_table(data, headers=None, caption=\


,
  col_align="c", multicol_data=None, multirow_data=None)`:
  Ajoute un tableau avec support pour `MultiColumn` et `MultiRow`.
  - `data` (list de listes): Les données du tableau.
  - `headers` (list, optionnel): Les en-têtes de colonnes.
  - `caption` (str): Légende du tableau.
  - `col_align` (str): Alignement des colonnes (par défaut : `"c"` pour centré).
  - `multicol_data` (dict, optionnel): Dictionnaire pour les cellules `MultiColumn`.
  - `multirow_data` (dict, optionnel): Dictionnaire pour les cellules `MultiRow`.

- `dataframe_to_latex(dataframe, caption="", label="")`:
  Convertit un `pandas.DataFrame` en une chaîne de caractères de tableau LaTeX.
  - `dataframe` (pd.DataFrame): Le DataFrame à convertir.
  - `caption` (str): Légende du tableau.
  - `label` (str): Étiquette pour référencer le tableau.
  - **Retourne:** `str` : Une chaîne de caractères de tableau LaTeX.

- `add_dataframe_table(dataframe, caption="", label="")`:
  Ajoute un `pandas.DataFrame` en tant que tableau LaTeX au document.
  - `dataframe` (pd.DataFrame): Le DataFrame à ajouter.
  - `caption` (str): Légende du tableau.
  - `label` (str): Étiquette pour référencer le tableau.

- `add_bibliography(bib_path)`:
  Ajoute une bibliographie au document.
  - `bib_path` (str): Chemin vers le fichier `.bib`.

- `add_environment(env_name, content)`:
  Ajoute un environnement LaTeX personnalisé.
  - `env_name` (str): Nom de l'environnement.
  - `content` (str): Contenu de l'environnement.

- `add_appendix(title, content)`:
  Ajoute une annexe au document.
  - `title` (str): Titre de l'annexe.
  - `content` (str): Contenu de l'annexe.

- `add_package(package_name, options=None)`:
  Ajoute un package LaTeX au document.
  - `package_name` (str): Nom du package.
  - `options` (str, optionnel): Options du package.

- `add_custom_preamble(command)`:
  Ajoute une commande personnalisée au préambule LaTeX.
  - `command` (str): La commande LaTeX.

- `generate_tex()`:
  Génère le fichier `.tex`.

- `generate_pdf()`:
  Compile le fichier `.tex` en PDF.

- `generate_document(output_format="pdf")`:
  Génère le document au format spécifié (`'tex'` ou `'pdf'`).
  - `output_format` (str): Format de sortie (par défaut : `"pdf"`).





### `foampilot.solver.incompressible_fluid.incompressibleFluid`

La classe `incompressibleFluid` représente un cas OpenFOAM configuré pour fonctionner avec le solveur `incompressibleFluid`. Elle organise la structure principale du répertoire de cas OpenFOAM en initialisant les composants clés : `system`, `constant` et les dictionnaires de conditions aux limites. Elle fournit également des fonctionnalités pour mettre à jour les paramètres spécifiques au cas et pour exécuter la simulation `incompressibleFluid`.

**Attributs:**

- `case_path` (Path): Le chemin vers le répertoire du cas OpenFOAM.
- `system` (SystemDirectory): Gère le dossier `system` et ses fichiers.
- `constant` (ConstantDirectory): Gère le dossier `constant` et ses fichiers.
- `boundary` (Boundary): Gère les dictionnaires de conditions aux limites.

**Constructeur:**

```python
incompressibleFluid(path_case)
```

- `path_case` (str ou Path): Chemin vers le répertoire du cas OpenFOAM.

**Méthodes:**

- `update_case_specific_attributes()`:
  Met à jour ou définit les attributs spécifiques au cas `SimpleFoam`.

- `run_simulation(log_filename="log.incompressibleFluid")`:
  Exécute le solveur `incompressibleFluid` dans le répertoire du cas et écrit la sortie dans un fichier journal.
  - `log_filename` (str): Nom du fichier journal (par défaut : `"log.incompressibleFluid"`).
  - **Lève:** `FileNotFoundError`, `NotADirectoryError`, `RuntimeError` en cas d'échec.





### `foampilot.system.SystemDirectory.SystemDirectory`

La classe `SystemDirectory` gère le répertoire `system` d'un cas OpenFOAM. Elle gère la création, la configuration et la gestion de tous les fichiers système dans un cas OpenFOAM, y compris `controlDict`, `fvSchemes` et `fvSolution`. Elle fournit également des méthodes pour exécuter des utilitaires OpenFOAM comme `topoSet` et `createPatch`.

**Attributs:**

- `parent`: L'objet cas parent.
- `controlDict` (ControlDictFile): Le gestionnaire de fichier `controlDict`.
- `fvSchemes` (FvSchemesFile): Le gestionnaire de fichier `fvSchemes`.
- `fvSolution` (FvSolutionFile): Le gestionnaire de fichier `fvSolution`.
- `additional_files` (dict): Dictionnaire des fichiers système additionnels.

**Constructeur:**

```python
SystemDirectory(parent)
```

- `parent`: L'objet cas parent qui possède ce répertoire système.

**Méthodes:**

- `write()`:
  Écrit tous les fichiers système dans le répertoire du cas. Crée le répertoire système s'il n'existe pas et écrit `controlDict`, `fvSchemes`, `fvSolution` et tous les fichiers additionnels.

- `add_dict_file(file_name, file_content)`:
  Ajoute un fichier additionnel au répertoire système.
  - `file_name` (str): Le nom du fichier à ajouter (par exemple, `\'transportProperties\'`).
  - `file_content` (dict): Le contenu du fichier sous forme de dictionnaire.

- `to_dict()`:
  Convertit la configuration du répertoire système en un dictionnaire.
  - **Retourne:** `dict` : Un dictionnaire contenant les configurations de `controlDict`, `fvSchemes` et `fvSolution`.

- `from_dict(config)`:
  Charge la configuration du répertoire système à partir d'un dictionnaire.
  - `config` (dict): Dictionnaire contenant les configurations pour `controlDict`, `fvSchemes` et `fvSolution`.

- `run_topoSet()`:
  Exécute l'utilitaire `topoSet` dans le répertoire du cas.
  - **Lève:** `FileNotFoundError`, `NotADirectoryError`, `RuntimeError` en cas d'échec.

- `run_createPatch(overwrite=True)`:
  Exécute l'utilitaire `createPatch` dans le répertoire du cas.
  - `overwrite` (bool): Indique s'il faut ajouter l'option `-overwrite` (par défaut : `True`).
  - **Lève:** `FileNotFoundError`, `NotADirectoryError`, `RuntimeError` en cas d'échec.

- `write_functions_file(includes=None, filename="functions")`:
  Crée un fichier `functions` dans le répertoire système avec les inclusions données.
  - `includes` (list, optionnel): Liste des fichiers de fonction à inclure (par défaut : `["fieldAverage", "referencePressure", "runTimeControls"]`).
  - `filename` (str): Nom du fichier à créer (par défaut : `"functions"`).





### `foampilot.system.controlDictFile.ControlDictFile`

La classe `ControlDictFile` représente le fichier `controlDict` dans OpenFOAM. Elle gère la création et la manipulation de ce fichier qui contrôle le comportement d'exécution d'une simulation OpenFOAM.

**Attributs:**

- `application` (str): L'application OpenFOAM à exécuter (par exemple, `"simpleFoam"`).
- `startFrom` (str): Option de temps de début (`"startTime"`, `"firstTime"`, `"latestTime"`).
- `startTime` (float): Temps de simulation initial.
- `stopAt` (str): Condition d'arrêt (`"endTime"`, `"writeNow"`, `"noWriteNow"`, `"nextWrite"`).
- `endTime` (float): Temps de simulation final.
- `deltaT` (float): Taille du pas de temps.
- `writeControl` (str): Méthode de contrôle d'écriture (`"timeStep"`, `"runTime"`, `"adjustableRunTime"`).
- `writeInterval` (int): Intervalle entre l'écriture des résultats.
- `purgeWrite` (int): Nombre de répertoires de temps à conserver.
- `writeFormat` (str): Format de fichier pour la sortie (`"ascii"`, `"binary"`).
- `writePrecision` (int): Précision pour les données de sortie.
- `writeCompression` (str): Compression pour les fichiers de sortie (`"on"`, `"off"`).
- `timeFormat` (str): Format de nommage du répertoire de temps (`"general"`, `"fixed"`, `"scientific"`).
- `timePrecision` (int): Précision pour les noms de répertoire de temps.
- `runTimeModifiable` (bool): Indique si le dictionnaire peut être modifié pendant l'exécution.
- `functions` (dict): Dictionnaire des objets de fonction pour des contrôles d'exécution supplémentaires.

**Constructeur:**

```python
ControlDictFile(application="incompressibleFluid", startFrom="startTime", startTime=0,
                 stopAt="endTime", endTime=5000, deltaT=1, writeControl="timeStep",
                 writeInterval=100, purgeWrite=10, writeFormat="ascii", writePrecision=6,
                 writeCompression="off", timeFormat="general", timePrecision=6,
                 runTimeModifiable=True, functions=None)
```

- `application`: Application du solveur OpenFOAM (par défaut : `"incompressibleFluid"`).
- `startFrom`: Option de temps de début (par défaut : `"startTime"`).
- `startTime`: Temps de simulation initial (par défaut : `0`).
- `stopAt`: Condition d'arrêt (par défaut : `"endTime"`).
- `endTime`: Temps de simulation final (par défaut : `5000`).
- `deltaT`: Taille du pas de temps (par défaut : `1`).
- `writeControl`: Méthode de contrôle d'écriture (par défaut : `"timeStep"`).
- `writeInterval`: Intervalle d'écriture en pas de temps (par défaut : `100`).
- `purgeWrite`: Nombre de répertoires de temps à conserver (par défaut : `10`).
- `writeFormat`: Format de fichier de sortie (par défaut : `"ascii"`).
- `writePrecision`: Précision de sortie (par défaut : `6`).
- `writeCompression`: Compression de sortie (par défaut : `"off"`).
- `timeFormat`: Format du répertoire de temps (par défaut : `"general"`).
- `timePrecision`: Précision du répertoire de temps (par défaut : `6`).
- `runTimeModifiable`: Autoriser la modification à l'exécution (par défaut : `True`).
- `functions`: Dictionnaire des objets de fonction (par défaut : `None` -> dictionnaire vide).

**Méthodes:**

- `to_dict()`:
  Convertit les paramètres `controlDict` en un dictionnaire.
  - **Retourne:** `dict` : Un dictionnaire contenant tous les paramètres `controlDict` avec leurs valeurs actuelles.

- `from_dict(config)`:
  Crée une instance `ControlDictFile` à partir d'un dictionnaire de configuration.
  - `config` (dict): Dictionnaire contenant les paramètres `controlDict`.
  - **Retourne:** `ControlDictFile` : Une nouvelle instance initialisée avec les valeurs fournies ou par défaut.





### `foampilot.system.fvSchemesFile.FvSchemesFile`

La classe `FvSchemesFile` représente le fichier `fvSchemes` dans OpenFOAM. Elle gère la création et la manipulation de ce fichier qui définit les schémas numériques utilisés dans une simulation OpenFOAM.

**Attributs:**

- `ddtSchemes` (dict): Configuration des schémas de dérivée temporelle.
- `gradSchemes` (dict): Configuration des schémas de gradient.
- `divSchemes` (dict): Configuration des schémas de divergence.
- `laplacianSchemes` (dict): Configuration des schémas laplaciens.
- `interpolationSchemes` (dict): Configuration des schémas d'interpolation.
- `snGradSchemes` (dict): Configuration des schémas de gradient normal de surface.

**Constructeur:**

```python
FvSchemesFile(ddtSchemes=None, gradSchemes=None, divSchemes=None,
                 laplacianSchemes=None, interpolationSchemes=None, snGradSchemes=None)
```

- `ddtSchemes`: Schémas de dérivée temporelle (par défaut : `{"default": "steadyState"}`).
- `gradSchemes`: Schémas de gradient (par défaut : `{"default": "Gauss linear"}`).
- `divSchemes`: Schémas de divergence (par défaut : `{"default": "none", "div(phi,U)": "bounded Gauss upwind", "turbulence": "bounded Gauss upwind", "div(phi,k)": "$turbulence", "div(phi,epsilon)": "$turbulence", "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"}`).
- `laplacianSchemes`: Schémas laplaciens (par défaut : `{"default": "Gauss linear corrected"}`).
- `interpolationSchemes`: Schémas d'interpolation (par défaut : `{"default": "linear"}`).
- `snGradSchemes`: Schémas de gradient normal de surface (par défaut : `{"default": "corrected"}`).

**Méthodes:**

- `to_dict()`:
  Convertit la configuration des schémas en un dictionnaire.
  - **Retourne:** `dict` : Un dictionnaire contenant tous les schémas avec leur configuration actuelle.

- `from_dict(config)`:
  Crée une instance `FvSchemesFile` à partir d'un dictionnaire de configuration.
  - `config` (dict): Dictionnaire contenant la configuration des schémas.
  - **Retourne:** `FvSchemesFile` : Une nouvelle instance initialisée avec les schémas fournis.





### `foampilot.system.fvSolutionFile.FvSolutionFile`

La classe `FvSolutionFile` représente le fichier `fvSolution` dans OpenFOAM. Elle gère la création et la manipulation de ce fichier qui définit les algorithmes de résolution et les contrôles du solveur pour une simulation OpenFOAM.

**Attributs:**

- `solvers` (dict): Dictionnaire contenant les paramètres du solveur pour chaque champ.
- `SIMPLE` (dict): Paramètres de contrôle de l'algorithme SIMPLE.
- `relaxationFactors` (dict): Facteurs de sous-relaxation pour les champs et les équations.

**Constructeur:**

```python
FvSolutionFile(solvers=None, SIMPLE=None, relaxationFactors=None)
```

- `solvers`: Dictionnaire des configurations de solveur pour chaque champ (par défaut : configuration détaillée pour `p`, `U`, `k`, `epsilon`, `R`, `nuTilda`).
- `SIMPLE`: Paramètres de l'algorithme SIMPLE (par défaut : `{"nNonOrthogonalCorrectors": "0", "residualControl": {"p": "1e-2", "U": "1e-4", "k": "1e-4", "epsilon": "1e-4"}}`).
- `relaxationFactors`: Facteurs de sous-relaxation (par défaut : `{"fields": {"p": "0.3"}, "equations": {"U": "0.7", "k": "0.7", "epsilon": "0.7"}}`).

**Méthodes:**

- `to_dict()`:
  Convertit la configuration du solveur en un dictionnaire.
  - **Retourne:** `dict` : Un dictionnaire contenant tous les paramètres du solveur avec leur configuration actuelle.

- `from_dict(config)`:
  Crée une instance `FvSolutionFile` à partir d'un dictionnaire de configuration.
  - `config` (dict): Dictionnaire contenant la configuration du solveur.
  - **Retourne:** `FvSolutionFile` : Une nouvelle instance initialisée avec les configurations fournies.





### `foampilot.utilities.dictonnary.OpenFOAMDictAddFile`

La classe `OpenFOAMDictAddFile` est une classe de base pour les fichiers de configuration OpenFOAM. Elle gère l'en-tête standard `FoamFile` et l'écriture des attributs, y compris la gestion des dictionnaires imbriqués et des listes.

**Constructeur:**

```python
OpenFOAMDictAddFile(object_name, **attributes)
```

- `object_name` (str): Le nom de l'objet pour l'en-tête du fichier OpenFOAM.
- `**attributes`: Arguments mots-clés arbitraires représentant les attributs spécifiques du fichier.

**Méthodes:**

- `_write_attributes(file, attributes, indent_level=0)`:
  Écrit les attributs dans le fichier avec une indentation appropriée. Gère les cas spéciaux comme les boîtes (`box`) et les listes.
  - `file` (objet fichier): L'objet fichier dans lequel écrire.
  - `attributes` (dict): Les attributs à écrire.
  - `indent_level` (int): Le niveau d'indentation actuel.

- `write(name_dict, base_path, folder='system')`:
  Écrit le fichier OpenFOAM dans le chemin spécifié.
  - `name_dict` (str): Le nom du dictionnaire à écrire.
  - `base_path` (str ou Path): Le chemin de base du répertoire du cas OpenFOAM.
  - `folder` (str): Le sous-répertoire où écrire le fichier (par défaut : `'system'`).

### `foampilot.utilities.dictonnary.dict_tools`

La classe `dict_tools` fournit des méthodes statiques pour créer des dictionnaires de configuration OpenFOAM, notamment pour les patchs et les actions `topoSetDict`.

**Méthodes statiques:**

- `create_patches_dict(patch_names, construct_from="set", point_sync=False)`:
  Crée un dictionnaire pour plusieurs patchs dans OpenFOAM.
  - `patch_names` (list): Liste des noms des patchs à créer.
  - `construct_from` (str): Méthode de construction (`'patches'` ou `'set'`).
  - `point_sync` (bool): Indique si la synchronisation des points est activée.
  - **Retourne:** `dict` : Dictionnaire représentant la configuration des patchs.

- `create_action(name, action_type, action, source, **kwargs)`:
  Crée une action pour `topoSetDict`.
  - `name` (str): Nom de l'action.
  - `action_type` (str): Type de l'action (par exemple, `'cellSet'`, `'faceSet'`).
  - `action` (str): Type d'action (par exemple, `'new'`, `'subset'`).
  - `source` (str): Source de l'action (par exemple, `'boxToCell'`, `'patchToFace'`).
  - `**kwargs`: Autres attributs optionnels pour l'action.
  - **Retourne:** `dict` : Dictionnaire représentant l'action.
  - **Lève:** `ValueError` si le type d'action ou l'action est invalide.

- `create_actions_dict(actions)`:
  Crée un dictionnaire pour les actions dans `topoSetDict`.
  - `actions` (list): Liste des dictionnaires représentant chaque action.
  - **Retourne:** `dict` : Dictionnaire représentant les actions.





### `foampilot.utilities.epw_weather_reader.WeatherFileEPW`

La classe `WeatherFileEPW` gère les fichiers EnergyPlus Weather (EPW). Elle fournit des fonctionnalités pour lire, écrire, analyser et visualiser les données climatiques.

**Attributs:**

- `headers` (dict): Dictionnaire stockant les en-têtes du fichier EPW.
- `dataframe` (pd.DataFrame): DataFrame contenant les données climatiques.

**Constructeur:**

```python
WeatherFileEPW()
```

**Méthodes:**

- `read(fp)`:
  Lit les en-têtes et les données climatiques d'un fichier EPW.
  - `fp` (str): Chemin du fichier EPW à lire.

- `_read_headers(fp)`:
  Lit les en-têtes d'un fichier EPW.
  - `fp` (str): Chemin du fichier EPW.
  - **Retourne:** `dict` : Dictionnaire contenant les informations d'en-tête.

- `_read_data(fp)`:
  Lit les données climatiques d'un fichier EPW dans un DataFrame.
  - `fp` (str): Chemin du fichier EPW.
  - **Retourne:** `pd.DataFrame` : DataFrame contenant les données climatiques.

- `_first_row_with_climate_data(fp)`:
  Trouve la première ligne du fichier EPW qui contient des données climatiques.
  - `fp` (str): Chemin du fichier EPW.
  - **Retourne:** `int` : Index de la ligne où les données climatiques commencent.

- `write(fp)`:
  Écrit les en-têtes du fichier EPW et les données climatiques dans un nouveau fichier.
  - `fp` (str): Chemin du fichier EPW à enregistrer.

- `analyze_temperature()`:
  Calcule les statistiques pour la température de bulbe sec.
  - **Retourne:** `pd.Series` : Résumé statistique de la température de bulbe sec.
  - **Lève:** `ValueError` si aucune donnée climatique n'a été chargée.

- `analyze_humidity()`:
  Calcule les statistiques pour l'humidité relative.
  - **Retourne:** `pd.Series` : Résumé statistique de l'humidité relative.
  - **Lève:** `ValueError` si aucune donnée climatique n'a été chargée.

- `plot_wind_rose()`:
  Génère un graphique de rose des vents en utilisant la vitesse et la direction du vent.
  - **Lève:** `ValueError` si aucune donnée climatique n'a été chargée.

- `get_header()`:
  Retourne les en-têtes du fichier EPW.
  - **Retourne:** `dict` : Dictionnaire des en-têtes.
  - **Lève:** `ValueError` si les en-têtes n'ont pas été chargés.

- `get_dataframe()`:
  Retourne le DataFrame contenant les données climatiques.
  - **Retourne:** `pd.DataFrame` : DataFrame des données climatiques.
  - **Lève:** `ValueError` si les données climatiques n'ont pas été chargées.





### `foampilot.utilities.fluids_theory.FluidMechanics`

La classe `FluidMechanics` est un calculateur complet de mécanique des fluides pour les applications CFD. Elle fournit des méthodes pour calculer les paramètres clés de la mécanique des fluides et les caractéristiques de la couche limite, essentiels pour le dimensionnement du maillage CFD et la configuration de la simulation.

**Attributs:**

- `fluid_name` (FluidsList): Le fluide analysé (à partir de l'énumération `FluidsList`).
- `temperature` (ValueWithUnit): Température du fluide avec unités.
- `pressure` (ValueWithUnit): Pression du fluide avec unités.
- `velocity` (ValueWithUnit): Vitesse d'écoulement caractéristique avec unités.
- `characteristic_length` (ValueWithUnit): Échelle de longueur pertinente pour les nombres sans dimension.
- `fluid`: Objet `PyFluids Fluid` initialisé avec l'état actuel.

**Constructeur:**

```python
FluidMechanics(fluid_name, temperature, pressure, velocity, characteristic_length)
```

- `fluid_name`: Type de fluide de l'énumération `FluidsList` (par exemple, `FluidsList.Water`).
- `temperature`: Température du fluide (par exemple, `ValueWithUnit(300, 'K')`).
- `pressure`: Pression du fluide (par exemple, `ValueWithUnit(101325, 'Pa')`).
- `velocity`: Vitesse d'écoulement caractéristique (par exemple, `ValueWithUnit(2, 'm/s')`).
- `characteristic_length`: Échelle de longueur pertinente (par exemple, diamètre du tuyau).

**Méthodes:**

- `get_fluid_properties(temperature)`:
  Récupère les propriétés fondamentales du fluide à la température spécifiée.
  - `temperature`: Température à laquelle évaluer les propriétés.
  - **Retourne:** `tuple` : (densité [kg/m³], viscosité dynamique [Pa·s], conductivité thermique [W/(m·K)], chaleur spécifique [J/(kg·K)]).

- `calculate_reynolds()`:
  Calcule le nombre de Reynolds pour les conditions d'écoulement actuelles.
  - **Retourne:** `float` : Nombre de Reynolds (Re = ρvL/μ).

- `calculate_y_plus(wall_shear_stress)`:
  Calcule la valeur y+ (distance à la paroi sans dimension) pour la modélisation de la turbulence.
  - `wall_shear_stress`: Contrainte de cisaillement à la paroi avec unités.
  - **Retourne:** `float` : Valeur y+ (y+ = τ_w·L/μ).

- `calculate_prandtl()`:
  Calcule le nombre de Prandtl pour l'état actuel du fluide.
  - **Retourne:** `float` : Nombre de Prandtl (Pr = μ·c_p/k).

- `calculate_thermal_expansion_coefficient(temperature1, temperature2, density1, density2, density_ave)`:
  Calcule le coefficient de dilatation thermique (β) en utilisant des différences finies.
  - `temperature1`: Premier point de température.
  - `temperature2`: Deuxième point de température.
  - `density1`: Densité à `temperature1`.
  - `density2`: Densité à `temperature2`.
  - `density_ave`: Densité moyenne entre les deux points.
  - **Retourne:** `float` : Coefficient de dilatation thermique [1/K].

- `calculate_rayleigh(temperature1, temperature2)`:
  Calcule le nombre de Rayleigh pour l'analyse de la convection naturelle.
  - `temperature1`: Température de la paroi froide.
  - `temperature2`: Température de la paroi chaude.
  - **Retourne:** `float` : Nombre de Rayleigh (Ra = g·β·ΔT·L³/(ν·α)).

- `characteristic_thickness_turbulent()`:
  Estime l'épaisseur de la couche limite turbulente en utilisant une corrélation empirique.
  - **Retourne:** `ValueWithUnit` : Épaisseur de la couche limite avec unités (δ = 0.37L/Re^(1/5)).

- `calculate_layers_for_cell_size(target_cell_size, expansion_ratio=1.2)`:
  Calcule le nombre de cellules de couche limite nécessaires pour atteindre la taille cible.
  - `target_cell_size`: Taille de cellule souhaitée au bord de la couche limite.
  - `expansion_ratio`: Rapport de croissance entre les couches adjacentes (par défaut : `1.2`).
  - **Retourne:** `int` : Nombre de couches requises.
  - **Lève:** `ValueError` si l'épaisseur de la couche limite est non positive.





### `foampilot.utilities.manageunits.ValueWithUnit`

La classe `ValueWithUnit` est un wrapper pour les quantités Pint afin de gérer les valeurs avec des unités physiques. Elle fournit une interface pratique pour stocker des valeurs avec des unités, les convertir dans d'autres unités et les afficher.

**Attributs:**

- `ValueWithUnit` (pint.ValueWithUnit): La quantité avec sa valeur numérique et son unité associée.

**Constructeur:**

```python
ValueWithUnit(value, unit)
```

- `value` (float): La valeur numérique de la quantité.
- `unit` (str): L'unité sous forme de chaîne de caractères (par exemple, `"m/s"`, `"Pa"`, `"kg"`).

**Méthodes:**

- `set_ValueWithUnit(value, unit)`:
  Met à jour la quantité avec une nouvelle valeur et unité.
  - `value` (float): La nouvelle valeur numérique.
  - `unit` (str): La nouvelle unité associée.

- `get_in(target_unit)`:
  Convertit la quantité en une unité spécifiée et retourne sa valeur.
  - `target_unit` (str): L'unité cible pour la conversion (par exemple, `"ft/s"`, `"inch"`, `"psi"`).
  - **Retourne:** `float` : La valeur numérique convertie dans l'unité cible.

- `__repr__()`:
  Retourne une représentation textuelle de la quantité.
  - **Retourne:** `str` : La valeur numérique avec son unité (par exemple, `"10.0 m/s"`).





### `foampilot.utilities.stl_analyzer.STLAnalyzer`

La classe `STLAnalyzer` est conçue pour analyser et traiter les fichiers STL (Stereolithography). Elle fournit des méthodes pour charger des fichiers STL, extraire des propriétés géométriques, effectuer des requêtes spatiales et calculer des paramètres liés au maillage pour les simulations CFD.

**Attributs:**

- `filename` (Path): Objet `Path` pointant vers le fichier STL.
- `mesh` (pv.PolyData): Objet maillage PyVista contenant la géométrie STL chargée.
- `reader` (pyvista.reader): Objet lecteur pour le fichier STL.

**Constructeur:**

```python
STLAnalyzer(filename)
```

- `filename` (Path): Objet `Path` pointant vers le fichier STL à analyser.

**Méthodes:**

- `load()`:
  Charge et retourne le fichier STL en tant que maillage PyVista.
  - **Retourne:** `pv.PolyData` : L'objet maillage chargé.

- `get_info()`:
  Obtient des informations géométriques de base sur le maillage chargé.
  - **Retourne:** `dict` : Dictionnaire contenant les propriétés du maillage, y compris le nombre de points, de cellules, les dimensions, la surface et le volume.
  - **Lève:** `ValueError` si le maillage n'a pas encore été chargé.

- `is_point_inside(point)`:
  Vérifie si un point 3D se trouve à l'intérieur du maillage STL fermé.
  - `point` (tuple): Coordonnées (x, y, z) du point à tester.
  - **Retourne:** `bool` : `True` si le point est à l'intérieur du maillage, `False` sinon.
  - **Lève:** `ValueError` si le maillage n'a pas encore été chargé.

- `get_center_of_mass()`:
  Calcule le centre de masse du maillage STL.
  - **Retourne:** `tuple` : Coordonnées (x, y, z) du centre de masse.
  - **Lève:** `ValueError` si le maillage n'a pas encore été chargé.

- `get_curvature()`:
  Calcule les valeurs de courbure moyenne et gaussienne pour le maillage.
  - **Retourne:** `dict` : Dictionnaire contenant `Mean_Curvature` et `Gaussian_Curvature`.
  - **Lève:** `ValueError` si le maillage n'a pas encore été chargé.

- `get_smallest_curvature()`:
  Trouve la valeur de courbure moyenne minimale dans le maillage.
  - **Retourne:** `float` : La plus petite valeur de courbure moyenne.
  - **Lève:** `ValueError` si le maillage n'a pas encore été chargé.

- `calc_mesh_settings(stlBoundingBox, nu=1e-6, rho=1000., U=1.0, maxCellSize=0.5, sizeFactor=1.0, expansion_ratio=1.5, onGround=False, internalFlow=False, refinement=1, nLayers=5, halfModel=False, thicknessRatio=0.3)`:
  Calcule les paramètres de génération de maillage pour les simulations CFD basées sur la géométrie STL.
  - `stlBoundingBox` (tuple): Boîte englobante du STL (xmin, xmax, ymin, ymax, zmin, zmax).
  - `nu` (float): Viscosité cinématique (par défaut : `1e-6`).
  - `rho` (float): Densité (par défaut : `1000.`).
  - `U` (float): Vitesse caractéristique (par défaut : `1.0`).
  - `maxCellSize` (float): Taille maximale des cellules (par défaut : `0.5`).
  - `sizeFactor` (float): Multiplicateur de la taille du domaine (par défaut : `1.0`).
  - `expansion_ratio` (float): Rapport d'expansion de la couche limite (par défaut : `1.5`).
  - `onGround` (bool): Indique si l'objet est au sol (affecte le domaine) (par défaut : `False`).
  - `internalFlow` (bool): Indique si le flux est interne (affecte le dimensionnement des cellules) (par défaut : `False`).
  - `refinement` (int): Niveau de raffinement (0=grossier, 1=moyen, 2=fin) (par défaut : `1`).
  - `nLayers` (int): Nombre de couches limites (par défaut : `5`).
  - `halfModel` (bool): Indique si un demi-modèle est utilisé (symétrie) (par défaut : `False`).
  - `thicknessRatio` (float): Rapport d'épaisseur de la couche finale (par défaut : `0.3`).
  - **Retourne:** `dict` : Dictionnaire contenant tous les paramètres de maillage calculés, y compris les dimensions du domaine, le nombre de cellules, les paramètres de taille de cellule, les niveaux de raffinement et les paramètres de modélisation de la turbulence.





## Conclusion

Ce document fournit une documentation complète du module `foampilot`, couvrant ses principaux sous-modules et classes. Le module `foampilot` est conçu pour faciliter la configuration, l'exécution et le post-traitement des simulations OpenFOAM, en offrant des outils pour la gestion du maillage, la définition des conditions aux limites, la configuration des solveurs et l'analyse des résultats.

Les améliorations apportées à ce module visent à renforcer sa robustesse, sa convivialité et sa capacité à gérer des cas de simulation complexes. En particulier, l'ajout de validations d'entrée et d'une meilleure gestion des erreurs contribue à une expérience utilisateur plus fiable.

Pour toute question ou contribution, veuillez vous référer au dépôt GitHub original : [https://github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot).
