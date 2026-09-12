# Documentation Utilisateur pour `foampilot`

**Auteur :** Manus AI
**Date :** 4 Décembre 2025

## 1. Réflexion de Fonctionnement Globale de `foampilot`

Le module `foampilot` est conçu comme une surcouche Python pour **OpenFOAM**, visant à simplifier et automatiser le processus de simulation numérique en mécanique des fluides (CFD). Il abstrait la complexité de la structure des fichiers et des commandes OpenFOAM, permettant à l'utilisateur de définir, exécuter et post-traiter une simulation entièrement en Python.

La philosophie de `foampilot` repose sur les principes suivants :

1.  **Définition du Cas en Python :** Au lieu de modifier manuellement des fichiers de configuration (dictionnaires) dans la structure de répertoires OpenFOAM, l'utilisateur interagit avec des objets Python (`Solver`, `Meshing`, `Boundary`, `Constant`, `System`).
2.  **Génération Automatique des Fichiers :** Les objets Python sont responsables de la **génération automatique** des fichiers de configuration OpenFOAM (`controlDict`, `fvSchemes`, `transportProperties`, etc.) dans le répertoire du cas.
3.  **Intégration de l'Écosystème Python :** Le module s'intègre avec des bibliothèques Python puissantes pour des tâches spécifiques :
    *   **`classy_blocks`** pour la génération de maillage structuré (`blockMesh`).
    *   **`pyfluid`** (implicite dans l'exemple) pour la gestion des propriétés des fluides et des constantes physiques.
    *   **`pyvista`** pour le post-traitement et la visualisation avancée.
    *   **`latex_pdf`** pour la génération de rapports structurés.

En résumé, `foampilot` permet de passer d'un flux de travail **manuel et fragmenté** (édition de fichiers, exécution de commandes shell) à un flux de travail **scripté et reproductible** (un seul script Python).

## 2. Choix de la Géométrie et du Maillage

Le choix de la méthode de maillage est crucial en CFD et dépend de la complexité de la géométrie. `foampilot` intègre des outils pour gérer les trois scénarios principaux :

| Méthode de Maillage | Géométrie Cible | Outil `foampilot` / Librairie | Description et Avantages |
| :--- | :--- | :--- | :--- |
| **`blockMesh`** | Géométries simples, extrudées, ou composées de blocs hexaédriques. | `Meshing(..., mesher="blockMesh")` (via `classy_blocks`) | Idéal pour les géométries simples (canaux, cylindres, etc.) ou les domaines de calcul réguliers. Offre un **contrôle total** sur la qualité du maillage et la distribution des cellules. L'exemple `run_exemple2.py` utilise cette méthode pour créer une géométrie complexe par composition de blocs (`Cylinder`, `ExtrudedRing`, `Elbow`). |
| **`gmsh`** | Géométries complexes importées au format **STEP** ou IGES. | `Meshing(..., mesher="gmsh")` | Permet de mailler des géométries CAO complexes avec des maillages non structurés (tétraèdres, prismes). Nécessite l'importation d'un fichier géométrique (ex: `.step`). |
| **`snappyHexMesh`** | Géométries complexes importées au format **STL** (surface triangulée). | `Meshing(..., mesher="snappy")` | Le maillage de référence pour les géométries très complexes (véhicules, bâtiments). Il crée un maillage hexaédrique conforme à la surface STL, avec raffinement automatique des couches limites. ||

### 2.1. Maillage Structuré avec `blockMesh` (via `classy_blocks`)

Pour les géométries qui peuvent être décomposées en blocs hexaédriques (y compris les extrusions), `foampilot` utilise la librairie `classy_blocks`.

**Principe :**
1.  Définir des formes géométriques de base (`cb.Cylinder`, `cb.ExtrudedRing`, `cb.Elbow`).
2.  Utiliser les méthodes de chaînage (`.chain()`, `.expand()`, `.fill()`) pour construire la géométrie complexe.
3.  Définir le maillage sur chaque forme avec les méthodes `.chop_axial()`, `.chop_radial()`, `.chop_tangential()`.
4.  Assigner les **patchs** (surfaces) avec `.set_start_patch()`, `.set_end_patch()`.
5.  Assembler le tout dans un objet `cb.Mesh()` et écrire le `blockMeshDict` :
    \`\`\`python
    # Exemple d'utilisation
    mesh = cb.Mesh()
    # ... ajouter les formes ...
    mesh.set_default_patch("walls", "wall")
    mesh.write(current_path / "system" / "blockMeshDict", current_path /"debug.vtk")
    \`\`\`

### 2.2. Maillage Non Structuré avec `gmsh` (pour STEP)

Pour les géométries CAO au format STEP, l'approche est la suivante :

1.  Assurez-vous que le fichier STEP est disponible (ex: `geometry.step`).
2.  Initialisez l'objet `Meshing` en spécifiant `mesher="gmsh"`.
3.  Exécutez le maillage en passant le chemin du fichier STEP :

    \`\`\`python
    mesh_obj = Meshing(current_path, mesher="gmsh")
    # Le chemin d'accès au fichier STEP est passé à la méthode run
    mesh_obj.mesher.run(current_path / "geometry.step")
    \`\`\`

### 2.3. Maillage de Surface avec `snappyHexMesh` (pour STL)

Pour les géométries complexes au format STL, vous devez d'abord créer un maillage de base (`blockMesh`) qui englobe la géométrie STL, puis utiliser `snappyHexMesh` pour découper et affiner :

1.  Créez un `blockMeshDict` simple (via `classy_blocks` ou manuellement) pour définir le domaine de calcul englobant.
2.  Assurez-vous que le fichier STL est dans le répertoire `constant/triSurface`.
3.  Initialisez l'objet `Meshing` en spécifiant `mesher="snappyHexMesh"`.
4.  Exécutez le maillage. `foampilot` gère la configuration et l'exécution des étapes de `snappyHexMesh` :

    \`\`\`python
    mesh_obj = Meshing(current_path, mesher="snappyHexMesh")
    # snappyHexMesh n'a pas besoin d'un chemin de fichier en argument, il utilise les fichiers de configuration
    mesh_obj.mesher.run()
    \`\`\`

    *Note : La configuration détaillée de `snappyHexMeshDict` (niveaux de raffinement, couches limites) doit être gérée par l'utilisateur, soit en modifiant le dictionnaire généré par défaut, soit en utilisant des fonctions avancées de `foampilot` si elles sont disponibles.*

## 3. Choix du Solveur et Prise en Compte des Physiques

Le choix du solveur est la manière dont `foampilot` prend en compte les physiques de la simulation. Le module utilise la classe `Solver` pour configurer le cas, et le solveur OpenFOAM approprié est sélectionné et exécuté en arrière-plan.

### 3.1. Sélection du Solveur

La sélection du solveur se fait implicitement en configurant les propriétés de l'objet `Solver`.

\`\`\`python
from foampilot.solver import Solver
# Initialisation du solveur pour le répertoire de cas
solver = Solver(current_path)

# Configuration des physiques
solver.compressible = False  # Simulation incompressible
solver.with_gravity = False   # Sans gravité
# ... d'autres propriétés comme turbulence, multiphase, etc.
\`\`\`

En fonction de ces propriétés, `foampilot` configure le `controlDict` et les autres dictionnaires pour utiliser le solveur OpenFOAM le plus pertinent (par exemple, `simpleFoam` ou `pimpleFoam` pour l'incompressible, `rhoSimpleFoam` pour le compressible).

| Physique | Propriété `Solver` | Solveur OpenFOAM Typique |
| :--- | :--- | :--- |
| **Incompressible** | `solver.compressible = False` | `incompressibleFluid` (Solveur interne `foampilot`) |
| **Compressible** | `solver.compressible = True` | `compressibleFluid` (Solveur interne `foampilot`) |
| **Transitoire** | `solver.transient = True` | (Gère les paramètres transitoires) |
| **Turbulence** | `solver.turbulence_model = "kEpsilon"` | (Configure les modèles de turbulence) |
| **Multiphase (VOF)** | `solver.is_vof = True` | `incompressibleVoF` ou `compressibleVoF` (Solveurs internes `foampilot`) |
| **Solide (Déplacement)** | `solver.is_solid = True` | `solidDisplacement` (Solveur interne `foampilot`) |
| **Énergie (Thermique)** | `solver.energy_activated = True` | (Active les champs thermiques) |}],path:

### 3.2. Mise en Place des Conditions Limites

La gestion des conditions limites (CL) est centralisée dans l'objet `solver.boundary`. L'approche est de définir les conditions sur les **patchs** créés lors de l'étape de maillage.

**Étapes :**
1.  **Initialisation :** `solver.boundary.initialize_boundary()` crée les fichiers de conditions limites initiaux (dans le répertoire `0`).
2.  **Application :** La méthode `apply_condition_with_wildcard` permet d'appliquer une condition à un ou plusieurs patchs dont le nom correspond à un motif (wildcard).

\`\`\`python
# Initialisation des conditions limites
solver.boundary.initialize_boundary()

# Condition d'entrée (patch 'inlet')
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05
)

# Condition de sortie (patch 'outlet')
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)

# Condition de paroi (patch 'walls')
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)

# Écriture des fichiers de conditions limites
solver.boundary.write_boundary_conditions()
\`\`\`

**Conditions Limites Disponibles (Exemples Typiques) :**

| `condition_type` | Description | Champs Typiques |
| :--- | :--- | :--- |
| **`fixedValue`** | Valeur fixe imposée (ex: température, concentration). | `T`, `C` |
| **`zeroGradient`** | Gradient normal nul (condition de Neumann). | `p`, `T`, `U` (sortie) |
| **`velocityInlet`** | Vitesse d'entrée avec profil (souvent uniforme) et paramètres de turbulence. | `U`, `k`, `epsilon` |
| **`pressureOutlet`** | Pression de sortie fixe ou à gradient nul. | `p` |
| **`wall`** | Paroi solide (vitesse nulle, flux de chaleur nul par défaut). | `U`, `T` |
| **`symmetryPlane`** | Plan de symétrie. | `U`, `p`, `T` |

### 3.3. Modification d'un Dictionnaire ou Ajout d'un Patch

`foampilot` expose les dictionnaires OpenFOAM comme des objets Python, ce qui permet une modification directe et intuitive.

#### Modification d'un Dictionnaire Existant

Pour modifier ou ajouter une entrée dans un dictionnaire, il suffit d'accéder à l'attribut correspondant :

\`\`\`python
# Modification de la viscosité cinématique dans transportProperties
from foampilot.utilities.manageunits import ValueWithUnit
kinematic_viscosity = ValueWithUnit(1e-6, "m2/s")
solver.constant.transportProperties.nu = kinematic_viscosity

# Modification d'une propriété dans controlDict
solver.system.controlDict.writeInterval = 100 # Écrit les résultats toutes les 100 itérations
solver.system.controlDict.endTime = 1000 # Définit le temps de fin de simulation
\`\`\`

**Fichiers du Répertoire `system` Gérés par `foampilot` :**

L'objet `solver.system` expose plusieurs attributs pour gérer les fichiers de ce répertoire :

| Attribut | Fichier OpenFOAM Correspondant | Description |
| :--- | :--- | :--- |
| `solver.system.controlDict` | `controlDict` | Contrôle de la simulation (temps de début/fin, intervalle d'écriture, solveur). |
| `solver.system.fvSchemes` | `fvSchemes` | Schémas de discrétisation des termes de l'équation (différenciation, interpolation). |
| `solver.system.fvSolution` | `fvSolution` | Paramètres des solveurs linéaires (tolérance, préconditionneur) et des boucles PISO/SIMPLE. |
| `solver.system.decomposeParDict` | `decomposeParDict` | Configuration de la décomposition du domaine pour le calcul parallèle. |
| `solver.system.add_dict_file(...)` | Fichier personnalisé | Permet d'ajouter n'importe quel dictionnaire OpenFOAM supplémentaire. |

**Utilitaires Intégrés :**

`foampilot` permet également d'exécuter des utilitaires OpenFOAM directement via l'objet `solver.system` :

| Méthode | Utilitaire OpenFOAM | Description |
| :--- | :--- | :--- |
| `solver.system.run_topoSet()` | `topoSet` | Crée des ensembles de cellules et de faces pour le post-traitement ou le raffinement. |
| `solver.system.run_createPatch()` | `createPatch` | Modifie ou crée des patchs de conditions limites après le maillage. |
| `solver.system.write_functions_file()` | `functions` | Crée le fichier `system/functions` pour le post-traitement en cours de simulation (ex: `fieldAverage`). |

Après la modification, il est impératif d'appeler la méthode `write()` sur l'objet parent pour générer le fichier OpenFOAM mis à jour :

\`\`\`python
# Génère le fichier constant/transportProperties
solver.constant.write() 
# Génère le fichier system/controlDict
solver.system.write() 
\`\`\`

#### Ajout d'un Patch

L'ajout d'un patch se fait principalement lors de la phase de maillage (voir Section 2).

*   **Avec `blockMesh` (`classy_blocks`) :** Le patch est défini sur une face géométrique :
    \`\`\`python
    shapes[-1].set_end_patch("monNouveauPatch")
    \`\`\`
*   **Avec `gmsh` ou `snappyHexMesh` :** Le patch est défini dans les fichiers de configuration du maillage.

Une fois le maillage généré, le nouveau patch est automatiquement reconnu par OpenFOAM. Il suffit ensuite d'appliquer une condition limite à ce patch via `solver.boundary.apply_condition_with_wildcard(pattern="monNouveauPatch", ...)` comme décrit ci-dessus.
\`\`\`

## 3.4. Conditions aux Limites Temporelles et Spatiales depuis des Fichiers CSV

### 3.4.1. Vue d'ensemble

Le module  fournit une API complete pour appliquer des conditions aux limites variables dans le temps et/ou spatialement distribuees a partir de fichiers CSV ou de dataframes pandas.

### 3.4.2. API de Haut Niveau

####  — Conditions uniformes variables dans le temps

Attache une condition limite uniforme mais variable dans le temps a un patch. Utilise OpenFOAM Function1 table avec format csv.



Pour un champ vecteur, utilise :



Le CSV est copie dans  sans en-tete. Le Function1  lit les colonnes par index. Le  pour un vecteur est .

####  — Distributions spatiales interpolees

Interpole des valeurs depuis un nuage de points CSV sur les centres de faces du patch. Requiert SciPy.



Formats supportes:
- **Point cloud** : colonnes 
- **Long format** : colonnes 
- **Wide format** : une ligne par temps, colonnes spatiales

### 3.4.3. API de Bas Niveau

#### 



#### 

Ecrit le CSV sans en-tete dans :



####  / 

Genetrent le dictionnaire OpenFOAM pour uniformFixedValue:



### 3.4.4. Format des Fichiers CSV

| Format | Colonnes | Exemple |
| :--- | :--- | :--- |
| Scalaire | time, value | 0.0, 350 |
| Vecteur | time, vx, vy, vz | 0.0, 1.0, 0.0, 0.0 |
| Spatial (point cloud) | time, x, y, z, value | 0.0, 0.5, 0.0, 0.05, 350 |

### 3.4.5. Gestion de l'Energie (Temperature Incompressible)



Le solveur reste . Le transport de T est gere par un functionObject  dans :

# 0 "<stdin>"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "<stdin>"

D = nu / Pr.

**fvSchemes automatiques :**

| Entree | Valeur |
| :--- | :--- |
| div(phi,T) | bounded Gauss linearUpwind grad(T) |
| laplacian(DT,T) | Gauss linear corrected |

**fvSolution automatiques :**

| Entree | Description |
| :--- | :--- |
| T solver | smoothSolver, tolerance 1e-6 |
| TFinal | Herite de T avec relTol 0 |
| relaxationFactors.equations.T | 0.7 |

### 3.4.6. Limitations et Bonnes Pratiques

1.  doit etre appele avant .
2. Spatial CSV requiert SciPy: Requirement already satisfied: scipy in /home/steven/venv/lib/python3.10/site-packages (1.15.3)
Requirement already satisfied: numpy<2.5,>=1.23.5 in /home/steven/venv/lib/python3.10/site-packages (from scipy) (1.26.4).
3. Le separateur CSV est automatiquement mis entre guillemets dans le fichier OpenFOAM.
4. Columns pour les vecteurs: format OpenFOAM utilise (0 (1 2 3)).
5. Verifier que le solveur reste incompressibleFluid (ne pas utiliser le solveur functions).

## 3.4. Conditions aux Limites Temporelles et Spatiales depuis des Fichiers CSV

### 3.4.1. Vue d'ensemble

Le module `foampilot.boundaries.csv_boundary_condition` fournit une API complete pour appliquer des conditions aux limites variables dans le temps et/ou spatialement distribuees a partir de fichiers CSV ou de dataframes pandas. Ce module repose sur deux mecanismes OpenFOAM :

1. **Function1 `table` (format CSV)** — pour les valeurs uniformes qui varient dans le temps (ex: temperature d'entree sinusoidale).
2. **Valeurs `nonuniformList`** — pour les distributions spatiales interpolees sur les faces du patch (ex: profil de temperature 2D imposé en entree).

Le transport de l'energie/temperature (`T`) pour les echelles incompressibles s'effectue via un **functionObject `scalarTransport`** dans `system/functions`. Voir la section 5.4 pour le detail de la configuration energetique.

### 3.4.2. API de Haut Niveau

Deux fonctions sont exposees via `foampilot.boundaries` :

#### `set_csv_condition()` — Conditions uniformes variables dans le temps

Cette fonction attache une condition limite uniforme mais variable dans le temps a un patch. Elle utilise OpenFOAM's `Function1::table` avec `format csv`.

**Signature :**

```python
set_csv_condition(
    boundary,          # objet solver.boundary
    patch_name,        # str : nom du patch (ex: "inlet")
    field,             # str : nom du champ (ex: "T")
    data,             # str | Path | pandas.DataFrame
    time_column=0,     # str | int : colonne temps
    value_column=None, # str | int : colonne valeur (scalaire)
    value_columns=None,# list : colonnes valeurs (vecteur, 3 elements)
    header_lines=0,    # int : lignes d'en-tete a ignorer
    separator=",",     # str : separateur CSV
    out_of_bounds="clamp",  # str : "clamp", "error", "warn", "zero", "repeat"
    interpolation_scheme="linear",  # str : "linear" ou "spline"
    default_value=None,  # float | str : valeur par defaut pour l'entree "value"
    csv_filename=None,   # str : nom du fichier dans constant/
)
```

**Exemple — Temperature d'entree scalaire variable dans le temps :**

```python
import pandas as pd
from foampilot.boundaries import set_csv_condition

df = pd.DataFrame({
    "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "T_K": [300, 350, 320, 380, 340, 360],
})

set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data=df,
    time_column="time_s",
    value_column="T_K",
    header_lines=0,
    separator=",",
    out_of_bounds="clamp",
    interpolation_scheme="linear",
    default_value=300,
)
```

Le CSV est copie dans `constant/` sans en-tete. Le Function1 `table` lit les colonnes par index.

**Fichier `0/T` genere :**

```cpp
boundaryField
{
    inlet
    {
        type            uniformFixedValue;
        uniformValue    table
        {
            type            csv;
            nHeaderLine     0;
            columns         (0 1);
            file            "constant/inlet_temperature.csv";
            separator       ",";
            mergeSeparators false;
            interpolationScheme linear;
        }
        value           uniform 300;
    }
}
```

**Exemple — Vitesse d'entree vectorielle variable dans le temps :**

```python
df = pd.DataFrame({
    "time_s": [0.0, 1.0, 2.0],
    "Ux": [1.0, 2.0, 1.5],
    "Uy": [0.0, 0.5, 0.3],
    "Uz": [0.0, 0.0, 0.0],
})

set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="U",
    data=df,
    time_column="time_s",
    value_columns=["Ux", "Uy", "Uz"],
)
```

Le `columns` genere pour un champ vecteur est : `columns (0 (1 2 3));`

---

#### `set_spatial_csv_condition()` — Distributions spatiales interpolees

Cette fonction interpole des valeurs depuis un nuage de points CSV sur les centres de faces du patch OpenFOAM, puis ecrit des listes `nonuniformList` dans les fichiers de champ temporels.

**Signature :**

```python
set_spatial_csv_condition(
    boundary,              # objet solver.boundary
    patch_name,            # str : nom du patch
    field,                 # str : nom du champ (ex: "T")
    data,                  # str | Path | pandas.DataFrame
    time_column=0,         # str | int : colonne temps
    spatial_columns=None,  # list : colonnes [time, x, y, z, value] (point cloud)
    face_id_column=None,   # str | int : colonne ID de face (format long)
    value_column=None,     # str | int : colonne valeur (format long)
    header_lines=0,        # int
    separator=",",         # str
    default_value=None,    # float | str
    interpolation_method="linear",  # "linear", "nearest", "cubic"
)
```

**Formats supportes :**

1. **Format nuage de points (point cloud)** — Colonnes : `time, x, y, z, value`. Les points source sont interpoles sur les centres de faces via `scipy.interpolate.griddata`.

2. **Format long avec IDs de faces** — Colonnes : `time, face_id, value`. Chaque ligne specifie une face et sa valeur a un instant donne.

3. **Format large (wide)** — Une ligne par instant, une colonne par point spatial.

**Exemple — Profil de temperature spatial :**

```python
import pandas as pd
import math

rows = []
for t in [0.0, 0.5, 1.0]:
    for i in range(10):
        x = i * 0.2
        y = 0.5
        temp = 300 + 50 * math.sin(2 * math.pi * x / 2.0) + 20 * t
        rows.append({"time_s": t, "x": x, "y": y, "z": 0.05, "T_K": temp})

df = pd.DataFrame(rows)

set_spatial_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data=df,
    time_column="time_s",
    spatial_columns=["x", "y", "z", "T_K"],
    header_lines=0,
    separator=",",
    default_value=300,
    interpolation_method="nearest",
)
```

**Fichiers generes :**

- `0/T` — contient la distribution non-uniforme au temps initial (copiee depuis le template `0/T`).
- `<time>/T` — un fichier par instant du CSV, avec les valeurs non-uniformes interpolees sur les faces du patch.

---

### 3.4.3. API de Bas Niveau (Helpers)

Pour un controle fin, les fonctions suivantes sont disponibles :

#### `CsvTimeSeries`

Classe utilitaire pour manipuler des series temporelles CSV :

```python
from foampilot.boundaries import CsvTimeSeries

ts = CsvTimeSeries(
    csv_file,          # Path | str | DataFrame
    time_column="time_s",
    value_column="T_K",
    header_lines=1,
    separator=",",
)
ts.get_initial_value()  # -> float : premiere valeur
ts.get_times()          # -> np.ndarray : colonne temps
ts.get_values()         # -> np.ndarray : colonne valeur
ts.get_dataframe()      # -> pd.DataFrame
ts.write_csv_table(destination_path, header_lines=0, separator=",")
```

#### `write_csv_table()`

Ecrit un CSV au format OpenFOAM-compatible dans `constant/<filename>` :

```python
from foampilot.boundaries import write_csv_table

csv_path = write_csv_table(
    case_path=solver.case_path,
    csv_data=df_or_path,
    time_column=0,
    value_columns=[1, 2, 3],  # pour vecteur
    header_lines=0,
    separator=",",
    filename="inlet_data.csv",
)
```

Le CSV est ecrit **sans en-tete** (`header=False, index=False`) car le `table Function1` lit directement les colonnes par index.

#### `make_uniform_fixed_value_bc()` / `make_uniform_fixed_value_vector_bc()`

Generent le dictionnaire OpenFOAM pour `uniformFixedValue` :

```python
from foampilot.boundaries import make_uniform_fixed_value_bc, make_uniform_fixed_value_vector_bc

bc_scalar = make_uniform_fixed_value_bc(
    csv_path="constant/inlet_temperature.csv",
    time_column=0,
    value_column=1,
    header_lines=0,
    separator=",",
    out_of_bounds="clamp",
    interpolation_scheme="linear",
    default_value=300,
)

bc_vector = make_uniform_fixed_value_vector_bc(
    csv_path="constant/inlet_velocity.csv",
    time_column=0,
    value_columns=[1, 2, 3],
)
```

Ces dictionnaires peuvent etre appliques via :

```python
solver.boundary.set_raw_condition("inlet", "T", bc_dict)
```

---

### 3.4.4. Format des Fichiers CSV

Le CSV source doit contenir une **colonne de temps** et **une ou trois colonnes de valeurs** :

| Format | Colonnes | Exemple |
| :--- | :--- | :--- |
| **Scalaire** | `time, value` | `0.0, 350` |
| **Vecteur** | `time, vx, vy, vz` | `0.0, 1.0, 0.0, 0.0` |
| **Spatial (nuage de points)** | `time, x, y, z, value` | `0.0, 0.5, 0.0, 0.05, 350` |

Le fichier est ensuite ecrit dans `constant/` (sans en-tete ni index) pour etre lu par OpenFOAM.

### 3.4.5. Gestion de l'Energie (Temperature Incompressible)

Pour les echelles incompressibles avec transport de la temperature :

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.energy_activated = True
solver.turbulence_model = "laminar"

solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
solver.constant.transportProperties.Pr = 0.85
```

Le solveur reste `incompressibleFluid` — le transport de `T` est gere par un `functionObject scalarTransport` dans `system/functions` :

```cpp
#includeFunc scalarTransport(T, diffusivity=constant, D = 1.76471e-05)
```

`D = nu / Pr` (diffusivite thermique constante).

**Entrees fvSchemes automatiques :**

| Entree | Valeur |
| :--- | :--- |
| `div(phi,T)` | `bounded Gauss linearUpwind grad(T)` |
| `laplacian(DT,T)` | `Gauss linear corrected` |

**Entrees fvSolution automatiques :**

| Entree | Description |
| :--- | :--- |
| `T` solver | `smoothSolver`, tolerance 1e-6 |
| `TFinal` | Herite de `$T` avec `relTol 0` |
| `relaxationFactors.equations.T` | 0.7 |

**Fichier `system/functions` genere :**

```cpp
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      functions;
}

#includeFunc scalarTransport(T, diffusivity=constant, D = 1.76471e-05)
```

### 3.4.6. Limitations et Bonnes Pratiques

1. **Ordre d'appel critique** : `write_boundary_conditions()` doit etre appele **avant** `set_spatial_csv_condition()` pour que le template `0/<field>` existe.
2. **Spatial CSV requiert SciPy** : `pip install scipy`.
3. **Separateur CSV** : automatiquement entoure de guillemets dans le fichier OpenFOAM.
4. **Columns pour les vecteurs** : format OpenFOAM utilise `columns (0 (1 2 3))`.
5. **Energie incompressible** : utiliser `incompressibleFluid` avec `scalarTransport` (le solveur `functions` ne resout pas l'ecoulement).

---

## 4. Mise en Place de `system` et `constant` avec `pyfluid`

Le module `pyfluid` (ou la classe `FluidMechanics` de `foampilot.utilities.fluids_theory`) est essentiel pour définir les propriétés physiques du fluide et les constantes du cas OpenFOAM.

#1. Définition des Constantes Physiques avec `FluidMechanics`

La classe `FluidMechanics` permet de calculer les propriétés d'un fluide (comme la viscosité cinématique $\nu$) en fonction de la température et de la pression, en utilisant des données physiques intégrées.

\`\`\`python
from foampilot.utilities.fluids_theory import FluidMechanics
from foampilot.utilities.manageunits import ValueWithUnit

# 1. Créer une instance de FluidMechanics pour l'eau à 20°C (293.15 K)
fluid_mech = FluidMechanics(
    FluidMechanics.get_available_fluids()['Water'],
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)

# 2. Récupérer les propriétés
properties = fluid_mech.get_fluid_properties()
kinematic_viscosity = properties['kinematic_viscosity']

# 3. Injecter la propriété dans le dictionnaire constant/transportProperties
solver.constant.transportProperties.nu = kinematic_viscosity

# 4. Écrire les répertoires system et constant
solver.system.write()
solver.constant.write()
\`\`\`

Cette approche garantit que les constantes physiques sont définies de manière cohérente et basée sur des données physiques fiables, et qu'elles sont correctement formatées pour le fichier `constant/transportProperties`.

**Fichiers du Répertoire `constant` Gérés par `foampilot` :**

L'objet `solver.constant` expose plusieurs attributs pour gérer les fichiers de ce répertoire :

| Attribut | Fichier OpenFOAM Correspondant | Description |
| :--- | :--- | :--- |
| `solver.constant.transportProperties` | `transportProperties` | Propriétés de transport (viscosité, diffusivité) pour les fluides incompressibles. |
| `solver.constant.physicalProperties` | `physicalProperties` | Propriétés physiques (densité, chaleur spécifique) pour les fluides compressibles. |
| `solver.constant.turbulenceProperties` | `turbulenceProperties` | Configuration du modèle de turbulence. |
| `solver.constant.gravity` | `g` | Vecteur de gravité. Activé via `solver.with_gravity = True`. |
| `solver.constant.pRef` | `pRef` | Point de référence pour la pression (pour les cas compressibles). |
| `solver.constant.radiation` | `radiationProperties` | Propriétés de rayonnement. Activé via `solver.constant.enable_radiation()`. |
| `solver.constant.fvmodels` | `fvModels` | Modèles de volume fini (utilisé pour le rayonnement). |

**Note sur la Gravité :** L'activation de la gravité (`solver.with_gravity = True`) entraîne la création du fichier `constant/g` et, si nécessaire, le remplacement du champ de pression `p` par `p_rgh` dans les fichiers de conditions initiales, conformément aux pratiques OpenFOAM pour les cas avec gravité.

## 5. Lancement du Solveur

Le lancement de la simulation est géré par la méthode `run_simulation()` de l'objet `Solver`.

### 5.1. Lancement Séquentiel

Le lancement par défaut est séquentiel :

\`\`\`python
# Exécute le solveur configuré (par exemple, simpleFoam)
solver.run_simulation()
\`\`\`

### 5.2. Lancement en Parallèle

Pour les maillages volumineux, le calcul parallèle est indispensable. `foampilot` intègre la gestion du parallélisme, qui nécessite généralement deux étapes : la décomposition du domaine et l'exécution parallèle.

\`\`\`python
# 1. Décomposition du domaine (utilise l'utilitaire OpenFOAM decomposePar)
# Le nombre de cœurs (cores) est spécifié ici
solver.decompose_domain(cores=4) 

# 2. Exécution du solveur en parallèle (utilise mpirun)
solver.run_simulation(parallel=True)

# 3. Reconstruction du domaine après la simulation (utilise reconstructPar)
solver.reconstruct_domain()
\`\`\`

**Note :** La méthode `decompose_domain` configure le dictionnaire `system/decomposeParDict` et exécute l'utilitaire. L'utilisateur peut modifier le `decomposeParDict` via `solver.system.decomposeParDict` avant l'exécution pour ajuster la méthode de décomposition (par exemple, `scotch`, `simple`, `hierarchical`).

## 6. Post-Traitement avec `pyvista`

Le post-traitement est géré par la classe `postprocess.FoamPostProcessing`, qui utilise la librairie **`pyvista`** pour la visualisation 3D et l'analyse des données.

### 6.1. Flux de Travail de Post-Traitement

1.  **Conversion des Résultats :** Les résultats OpenFOAM sont d'abord convertis au format VTK (lisible par `pyvista`) via l'utilitaire `foamToVTK`.
2.  **Chargement des Données :** Les données sont chargées dans des objets `pyvista` (`PolyData` ou `UnstructuredGrid`).
3.  **Analyse et Visualisation :** Utilisation des fonctions de `foampilot` ou directement de `pyvista` pour l'analyse.

\`\`\`python
from foampilot import postprocess
import pyvista as pv

# 1. Initialisation et conversion
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()

# 2. Chargement du dernier pas de temps
time_steps = foam_post.get_all_time_steps()
if time_steps:
    latest_time_step = time_steps[-1]
    structure = foam_post.load_time_step(latest_time_step)
    cell_mesh = structure["cell"] # Le maillage des cellules

# 3. Visualisation (Exemple de tracé de contour de pression)
pl_contour = pv.Plotter(off_screen=True)
pl_contour.add_mesh(cell_mesh, scalars='p', show_scalar_bar=True)
foam_post.export_plot(pl_contour, current_path / "contour_plot.png")
\`\`\`

### 6.2. Détail des Possibilités de Post-Traitement

`foampilot` et `pyvista` offrent une gamme étendue de possibilités :

| Fonctionnalité | Description | Méthode `foampilot` / `pyvista` |
| :--- | :--- | :--- |
| **Visualisation de Tranches** | Afficher les champs de données (U, p, T) sur un plan de coupe. | `foam_post.plot_slice(...)` |
| **Tracé de Contours** | Visualiser les lignes ou surfaces d'isovaleur d'un champ (ex: pression). | `pl.add_mesh(..., scalars='p')` |
| **Visualisation de Vecteurs** | Afficher le champ de vitesse (U) sous forme de flèches (glyphs). | `cell_mesh.glyph(orient='U', ...)` |
| **Analyse de Tourbillons** | Calculer le critère Q ou la vorticité pour identifier les structures tourbillonnaires. | `foam_post.calculate_q_criterion(...)`, `foam_post.calculate_vorticity(...)` |
| **Statistiques de Maillage** | Obtenir des métriques de qualité du maillage (non-orthogonalité, aspect ratio). | `foam_post.get_mesh_statistics(...)` |
| **Statistiques de Champ** | Calculer les valeurs min/max/moyenne/écart-type d'un champ sur une région. | `foam_post.get_region_statistics(...)` |
| **Export de Données** | Exporter les données de champ (U, p) vers des formats externes (CSV, JSON). | `foam_post.export_region_data_to_csv(...)` |
| **Animation** | Créer des animations à partir des différents pas de temps. | Fonctions `pyvista` pour l'enregistrement vidéo. |
\`\`\`

## 7. Rapport en LaTeX avec `latex_pdf`

L'une des fonctionnalités les plus puissantes de `foampilot` est sa capacité à générer automatiquement un rapport de simulation structuré au format PDF, en utilisant le module `latex_pdf`.

### 7.1. Fonctionnement du Module `latex_pdf`

Le module `latex_pdf` est une surcouche qui permet de construire un document LaTeX en Python, sans avoir à écrire directement le code LaTeX. Il est particulièrement adapté pour intégrer des résultats de simulation (statistiques, figures, tableaux).

**Étapes :**
1.  **Collecte des Données :** Les données de post-traitement (statistiques, chemins des images générées par `pyvista`) sont collectées et sauvegardées (souvent au format JSON ou CSV).
2.  **Initialisation du Document :** Création de l'objet `LatexDocument`.
3.  **Ajout de Contenu :** Utilisation des méthodes de l'objet pour ajouter des sections, des tableaux, des figures, etc.
4.  **Génération du PDF :** Compilation du document LaTeX en PDF.

### 7.2. Détail des Possibilités de `latex_pdf`

Le module offre des méthodes pour structurer le rapport et intégrer les résultats :

| Méthode | Description | Utilisation Typique |
| :--- | :--- | :--- |
| **`LatexDocument(...)`** | Initialise le document avec titre, auteur et nom de fichier. | Début du script de rapport. |
| **`add_title()`** | Ajoute la page de titre. | Après l'initialisation. |
| **`add_toc()`** | Ajoute la table des matières. | Après la page de titre. |
| **`add_abstract(...)`** | Ajoute un résumé du rapport. | Introduction du rapport. |
| **`add_section(...)`** | Ajoute une nouvelle section ou sous-section. | Structuration du rapport. |
| **`add_table(...)`** | Intègre un tableau de données (ex: statistiques de maillage). | Présentation des données numériques. |
| **`add_figure(...)`** | Intègre une image (ex: visualisation `pyvista`). | Présentation des résultats visuels. |
| **`add_appendix(...)`** | Ajoute une annexe. | Pour les données brutes ou les fichiers exportés. |
| **`generate_document(...)`** | Compile le document LaTeX en PDF. | Fin du script de rapport. |

### 7.3. Exemple d'Intégration de Résultats

L'exemple `run_exemple2.py` montre comment les statistiques et les figures sont intégrées :

\`\`\`python
# Chargement des statistiques JSON
stats_file = current_path / "all_stats.json"
with open(stats_file, "r") as f:
    stats = json.load(f)

doc = latex_pdf.LatexDocument(
    title="Simulation Report: Muffler Flow Case",
    author="Automated Report",
    filename="simulation_report",
    output_dir=current_path
)

# ... ajout des sections ...

# Ajout d'un tableau de statistiques
mesh_stats = stats.get("mesh_stats", {})
mesh_table_data = [[k, v] for k, v in mesh_stats.items()]
doc.add_table(
    mesh_table_data,
    headers=["Statistic", "Value"],
    caption="Mesh Quality Statistics"
)

# Ajout des figures générées par pyvista
for img_name in ["slice_plot.png", "contour_plot.png", "vector_plot.png", "mesh_style_plot.png"]:
    img_path = current_path / img_name
    if img_path.exists():
        doc.add_figure(str(img_path), caption=img_name.replace("_", " ").title(), width="0.7\\textwidth")

# Génération finale
doc.generate_document(output_format="pdf")
\`\`\`

Cette capacité permet de garantir une **traçabilité** et une **reproductibilité** complètes des résultats de simulation, en générant un livrable professionnel directement à partir du script de simulation.
\`\`\`