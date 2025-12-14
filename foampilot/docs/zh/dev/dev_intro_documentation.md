# Documentation de Prise en Main pour Développeurs : Module `foampilot`


## 1. Introduction

Le module `foampilot` est conçu pour simplifier et automatiser la création, la configuration, l'exécution et le post-traitement des cas de simulation basés sur **OpenFOAM**. Il offre une interface orientée objet en Python pour gérer les fichiers de configuration complexes d'OpenFOAM, permettant aux développeurs de se concentrer sur la physique et la géométrie du problème plutôt que sur la syntaxe des dictionnaires.

Cette documentation a pour but de fournir une vue d'ensemble détaillée de la structure du code pour faciliter la compréhension et la contribution au projet.

## 2. Concepts Clés et Architecture

L'architecture de `foampilot` est étroitement calquée sur la structure d'un cas OpenFOAM standard, qui est divisé en trois répertoires principaux : `constant`, `system`, et le répertoire du temps initial (`0`).

### La Classe Centrale : `Solver`

La classe `Solver` (`foampilot.solver.Solver`) est l'orchestrateur central du module. Elle encapsule l'ensemble du cas de simulation et fournit des points d'accès aux configurations des répertoires `constant` et `system`, ainsi qu'à la gestion des conditions aux limites.

Lors de l'initialisation, une instance de `Solver` crée automatiquement les objets nécessaires pour gérer les fichiers de configuration :

*   `solver.constant` : Une instance de `ConstantDirectory` pour gérer les fichiers du répertoire `constant`.
*   `solver.system` : Une instance de `SystemDirectory` pour gérer les fichiers du répertoire `system`.
*   `solver.boundary` : Une instance de `Boundary` pour gérer les conditions aux limites.

## 3. Structure Détaillée du Code

Le cœur de la logique de `foampilot` se trouve dans le répertoire `foampilot/foampilot/src/foampilot`. Ce répertoire est organisé en sous-modules logiques, chacun responsable d'un aspect spécifique de la gestion des cas OpenFOAM.

| Répertoire Source | Description | Classes Clés (Exemples) |
| :--- | :--- | :--- |
| `base` | Classes de base et utilitaires pour la manipulation des fichiers OpenFOAM. | `Meshing`, `OpenFOAMFile` |
| `solver` | Contient la classe principale `Solver` et la logique d'exécution de la simulation. | `Solver`, `BaseSolver` |
| `constant` | Gestion des fichiers de configuration dans le répertoire `constant`. | `ConstantDirectory`, `transportPropertiesFile`, `turbulencePropertiesFile` |
| `system` | Gestion des fichiers de configuration dans le répertoire `system`. | `SystemDirectory`, `controlDictFile`, `fvSchemesFile`, `fvSolutionFile` |
| `boundaries` | Définition et application des conditions aux limites. | `Boundary`, `boundaries_conditions_config` |
| `mesh` | Outils de maillage, y compris l'intégration avec `classy_blocks` et `snappyHexMesh`. | `BlockMeshFile`, `Meshing`, `gmsh_mesher` |
| `utilities` | Fonctions et classes utilitaires non spécifiques à OpenFOAM (unités, propriétés des fluides, etc.). | `Quantity`, `FluidMechanics`, `manageunits` |
| `postprocess` | Classes pour l'analyse des résultats, la visualisation (via `pyvista`) et l'extraction de données. | `FoamPostProcessing`, `ResidualsPost` |

## 4. Mécanismes Internes pour les Développeurs

Pour une contribution efficace, il est crucial de comprendre comment `foampilot` traduit les objets Python en fichiers OpenFOAM et gère la complexité des configurations.

### 4.1. Le Mécanisme d'Écriture des Fichiers (`OpenFOAMFile`)

Le cœur de la sérialisation des données réside dans la classe de base `OpenFOAMFile` (`foampilot/foampilot/src/foampilot/base/openFOAMFile.py`).

*   **Héritage et Attributs :** Chaque fichier de configuration OpenFOAM (comme `transportPropertiesFile` ou `controlDictFile`) hérite de `OpenFOAMFile`. Les paramètres de configuration sont stockés dans l'attribut `self.attributes` de l'instance.
*   **Accès Dynamique :** La surcharge des méthodes magiques `__getattr__` et `__setattr__` permet d'accéder et de modifier les paramètres directement comme des attributs de l'objet (ex: `solver.constant.transportProperties.nu = ...`), même si ces paramètres sont stockés dans le dictionnaire `self.attributes`.
*   **Sérialisation (`write_file`) :** La méthode `write_file` parcourt récursivement le dictionnaire `self.attributes` et utilise la méthode interne `_format_value` pour convertir les types de données Python (booléens, nombres, tuples, et surtout `Quantity`) en la syntaxe spécifique d'OpenFOAM (ex: `true`/`false`, listes entre parenthèses).

### 4.2. Gestion des Unités et des Dimensions (`Quantity`)

La classe `Quantity` (`foampilot/foampilot/src/foampilot/utilities/manageunits.py`) est un *wrapper* autour de la librairie `pint` et est essentielle pour assurer la cohérence physique.

*   **Rôle :** Elle stocke une valeur numérique avec son unité physique (ex: `Quantity(10, "m/s")`).
*   **Conversion Automatique :** Lors de l'écriture dans un fichier OpenFOAM, la méthode `_format_value` de `OpenFOAMFile` vérifie si la valeur est une instance de `Quantity`. Si c'est le cas, elle utilise la méthode `get_in(target_unit)` pour convertir la valeur dans l'unité attendue par OpenFOAM (définie dans `OpenFOAMFile.DEFAULT_UNITS`), garantissant que toutes les valeurs écrites sont dans le système d'unités de base d'OpenFOAM.
*   **Dimensions OpenFOAM :** La méthode `to_openfoam_dimensions()` utilise les capacités de `pint` pour dériver le vecteur de dimensions OpenFOAM (M, L, T, Θ, N, J, A) à partir de l'unité, ce qui est crucial pour la génération des en-têtes de fichiers de champs (ex: `U`, `p`).

### 4.3. Orchestration du Solveur et des Champs (`Solver` et `CaseFieldsManager`)

La classe `Solver` délègue la gestion des champs à la classe `CaseFieldsManager` (`foampilot/foampilot/src/foampilot/base/cases_variables.py`).

*   **Sélection du Solveur :** La classe `Solver` (`foampilot/foampilot/src/foampilot/solver/solver.py`) utilise des propriétés booléennes (ex: `self.compressible`, `self.with_gravity`, `self.is_vof`) pour déterminer le type de simulation. La méthode interne `_update_solver()` sélectionne le solveur OpenFOAM approprié (ex: `incompressibleFluid`, `compressibleVoF`) et met à jour l'instance de `BaseSolver`.
*   **Gestion des Champs :** `CaseFieldsManager` utilise ces mêmes propriétés pour générer dynamiquement la liste des champs physiques nécessaires (ex: `U`, `p`, `k`, `epsilon`, `T`).
    *   Si `self.with_gravity` est vrai, le champ de pression devient `p_rgh`.
    *   Si un modèle de turbulence est défini, les champs associés (`k`, `epsilon`, `omega`, `nut`) sont ajoutés.
    *   Cette liste de champs est ensuite utilisée par la classe `Boundary` pour initialiser les conditions aux limites pour *tous* les champs requis.

### 4.4. Gestion Avancée des Conditions aux Limites (`Boundary`)

La classe `Boundary` (`foampilot/foampilot/src/foampilot/boundaries/boundaries_dict.py`) est responsable de la traduction des conditions physiques en configurations OpenFOAM.

*   **Configuration Centralisée :** Elle utilise un dictionnaire de configuration (`BOUNDARY_CONDITIONS_CONFIG`) qui mappe les types de conditions physiques (ex: `"velocityInlet"`) aux configurations OpenFOAM requises pour chaque champ (U, p, k, etc.), en fonction du modèle de turbulence sélectionné.
*   **Application par *Wildcard* :** La méthode `apply_condition_with_wildcard(pattern, condition_type, **kwargs)` permet d'appliquer une condition à tous les patchs dont le nom correspond à une expression régulière (`pattern`).
*   **Résolution des Conditions :** Pour chaque champ, la méthode `_resolve_field_config` détermine la configuration OpenFOAM finale. Par exemple, pour une condition de paroi (`wall`), elle choisit entre `noSlip` ou `slip` en fonction des arguments fournis, et applique les fonctions de paroi (`wallFunction`) appropriées pour les champs de turbulence (k, epsilon, etc.) en utilisant le dictionnaire `WALL_FUNCTIONS`.
*   **Génération des Fichiers :** La méthode `write_boundary_conditions()` itère sur tous les champs gérés par `CaseFieldsManager` et utilise `OpenFOAMFile.write_boundary_file` pour générer les fichiers de conditions aux limites dans le répertoire `0/`.

## 5. Flux de Travail du Développeur (Basé sur `run_exemple2.py`)

L'exemple d'utilisation illustre le flux de travail typique pour un développeur utilisant `foampilot` :

| Étape | Description | Classes et Méthodes Clés |
| :--- | :--- | :--- |
| **1. Initialisation** | Définir le répertoire de travail et initialiser le solveur. | `Solver(path)`, `FluidMechanics` |
| **2. Configuration Physique** | Déterminer les propriétés du fluide et les appliquer aux fichiers de configuration. | `FluidMechanics.get_fluid_properties()`, `solver.constant.transportProperties.nu = ...` |
| **3. Maillage** | Définir la géométrie et le maillage (souvent via l'intégration `classy_blocks`) et générer le `blockMeshDict`. | `classy_blocks.Cylinder`, `cb.Mesh()`, `Meshing(path, mesher="blockMesh")` |
| **4. Conditions aux Limites** | Initialiser et appliquer les conditions aux limites aux patchs définis par le maillage. | `solver.boundary.initialize_boundary()`, `solver.boundary.apply_condition_with_wildcard()` |
| **5. Écriture des Fichiers** | Générer tous les fichiers de configuration OpenFOAM sur le disque. | `solver.system.write()`, `solver.constant.write()`, `solver.boundary.write_boundary_conditions()` |
| **6. Exécution** | Lancer la simulation OpenFOAM. | `solver.run_simulation()` |
| **7. Post-traitement** | Analyser les résultats, générer des visualisations et des rapports. | `FoamPostProcessing`, `ResidualsPost`, `latex_pdf.LatexDocument` |

Ce flux de travail met en évidence la manière dont les différents modules de `foampilot` s'articulent pour fournir une abstraction complète du processus de simulation OpenFOAM. Pour contribuer, un développeur doit comprendre comment les classes de chaque module interagissent avec l'objet central `Solver` et comment elles traduisent les commandes Python en syntaxe de dictionnaire OpenFOAM.