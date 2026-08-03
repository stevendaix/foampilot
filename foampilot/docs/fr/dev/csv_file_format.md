# Documentation : Gestion des Fichiers CSV pour les Conditions aux Limites

## 1. Vue d'ensemble

Le module `foampilot.boundaries.csv_boundary_condition` gère la lecture, l'écriture et l'interprétation des fichiers CSV pour les conditions aux limites OpenFOAM. Il supporte deux modes d'opération :

| Mode | Mécanisme OpenFOAM | Usage | Champ typique |
| :--- | :--- | :--- | :--- |
| **Uniform temps** | `Function1::table` (format CSV) | Valeur uniforme variant dans le temps | `T`, `U`, `p` |
| **Spatial** | `nonuniformList` directement dans le fichier de champ | Distribution spatiale interpolée sur les faces | `T`, `U` |

---

## 2. Formats de Fichiers CSV Supportés

### 2.1. Format Scalaire (Uniforme dans l'espace, Variable dans le Temps)

Utilisé avec `set_csv_condition()` pour un champ scalaire (`T`, `p`, etc.).

**Structure :** une colonne de temps + une colonne de valeur.

```
0.0,300
0.5,350
1.0,320
1.5,380
2.0,340
```

**Paramètres d'import :**

| Paramètre | Type | Description | Exemple |
| :--- | :--- | :--- | :--- |
| `time_column` | `str \| int` | Colonne temps (0 = première) | `0` ou `"time_s"` |
| `value_column` | `str \| int` | Colonne valeur | `1` ou `"T_K"` |
| `header_lines` | `int` | Nombre de lignes à ignorer | `0` ou `1` |
| `separator` | `str` | Séparateur CSV | `","` |
| `out_of_bounds` | `str` | Comportement en dehors des bornes | `"clamp"`, `"error"`, `"warn"`, `"zero"`, `"repeat"` |

**Code d'utilisation :**

```python
from foampilot.boundaries import set_csv_condition
import pandas as pd

df = pd.DataFrame({
    "time_s": [0.0, 0.5, 1.0, 1.5, 2.0],
    "T_K": [300, 350, 320, 380, 340],
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

---

### 2.2. Format Vecteur (Uniforme dans l'espace, Variable dans le Temps)

Utilisé avec `set_csv_condition()` pour un champ vecteur (`U`, etc.).

**Structure :** une colonne de temps + trois colonnes de valeurs (x, y, z).

```
0.0,1.0,0.0,0.0
0.5,2.0,0.5,0.0
1.0,1.5,0.3,0.0
```

**Paramètres supplémentaires :**

| Paramètre | Type | Description | Exemple |
| :--- | :--- | :--- | :--- |
| `value_columns` | `list[str] \| list[int]` | Colonnes des composantes | `["Ux", "Uy", "Uz"]` ou `[1, 2, 3]` |

```python
df = pd.DataFrame({
    "time_s": [0.0, 0.5, 1.0],
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

**Note :** Le `columns` généré dans le fichier OpenFOAM est `(0 (1 2 3))` — le premier entier est l'indice de la colonne temps, le tuple `(1 2 3)` contient les indices des colonnes vecteur.

---

### 2.3. Format Spatial — Nuage de Points (Point Cloud)

Utilisé avec `set_spatial_csv_condition()` pour une distribution spatiale de valeurs.

**Structure :** colonnes `time, x, y, z, value`. Les points source sont interpolés sur les centres de faces du patch via `scipy.interpolate.griddata`.

```
0.0,0.0,0.5,0.05,300
0.0,0.2,0.5,0.05,310
0.0,0.4,0.5,0.05,320
0.0,0.6,0.5,0.05,330
0.0,0.8,0.5,0.05,340
0.0,1.0,0.5,0.05,350
0.5,0.0,0.5,0.05,280
0.5,0.2,0.5,0.05,290
...
```

**Paramètres :**

| Paramètre | Type | Description |
| :--- | :--- | :--- |
| `spatial_columns` | `list` | Colonnes `[x, y, z, value]` ou `[time, x, y, z, value]` |
| `interpolation_method` | `str` | `"linear"`, `"nearest"`, `"cubic"` |

```python
set_spatial_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data="inlet_temperature_spatial.csv",
    time_column=0,          # colonne temps (indice 0)
    spatial_columns=[1, 2, 3, 4],  # colonnes x, y, z, value
    header_lines=0,
    separator=",",
    default_value=300,
    interpolation_method="nearest",
)
```

**Requis :** SciPy (`pip install scipy`).

---

### 2.4. Format Spatial — Long Format avec IDs de Faces

Alternative au nuage de points. Chaque ligne spécifie le temps, l'ID de face et la valeur.

**Structure :** `time, face_id, value`.

```
0.0,0,300
0.0,1,310
0.0,2,320
0.0,3,330
...
0.5,0,280
0.5,1,290
...
```

**Paramètres :**

| Paramètre | Type | Description |
| :--- | :--- | :--- |
| `face_id_column` | `str \| int` | Colonne ID de face |
| `value_column` | `str \| int` | Colonne valeur |

```python
set_spatial_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data="inlet_face_values.csv",
    time_column="time_s",
    face_id_column="face_id",
    value_column="T_K",
)
```

---

## 3. Écriture des Fichiers CSV par foampilot

### 3.1. `write_csv_table()` — Écriture dans `constant/`

Tous les CSV sont écrits dans le répertoire `constant/` du cas, **sans en-tête ni index** :

```
constant/
├── transportProperties
├── turbulenceProperties
└── inlet_temperature.csv    ← écrit par write_csv_table
```

**Pourquoi `header=False, index=False` :**

Le `Function1::table` d'OpenFOAM lit les colonnes par **indice** (0, 1, 2, ...), pas par nom. Aucun en-tête n'est nécessaire.

```python
from foampilot.boundaries import write_csv_table
from pathlib import Path
import pandas as pd

df = pd.DataFrame({
    "time_s": [0.0, 1.0, 2.0],
    "T_K": [300, 350, 320],
})

csv_path = write_csv_table(
    case_path=Path("/path/to/case"),
    csv_data=df,
    time_column=0,
    value_columns=[1],
    header_lines=0,
    separator=",",
    filename="inlet_temperature.csv",
)
# -> constant/inlet_temperature.csv
```

### 3.2. `CsvTimeSeries` — Gestion des séries temporelles

Classe utilitaire pour lire, manipuler et exporter des CSV :

```python
from foampilot.boundaries import CsvTimeSeries

ts = CsvTimeSeries(
    csv_file="constant/inlet_temperature.csv",
    time_column="time_s",
    value_column="T_K",
    header_lines=0,
    separator=",",
)

# Accès aux données
ts.get_times()           # -> np.ndarray [0.0, 1.0, 2.0, ...]
ts.get_values()          # -> np.ndarray [300, 350, 320, ...]
ts.get_dataframe()        # -> pd.DataFrame

# Export vers OpenFOAM
ts.write_csv_table(
    destination_path=Path("/case/constant/data.csv"),
    header_lines=0,
    separator=",",
)
```

---

## 4. Génération des Entrées OpenFOAM

### 4.1. Function1 `table` (CSV) — Format scalaire

Le fichier `0/T` généré pour une BC scalaire uniforme variable dans le temps :

```cpp
dimensions      [0 0 0 1 0 0 0];
internalField   uniform 0;

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
    outlet
    {
        type            zeroGradient;
    }
    ...
}
```

**Description des entrées :**

| Entrée | Valeur | OpenFOAM |
| :--- | :--- | :--- |
| `type` | `uniformFixedValue` | BC type — valeur uniforme par défaut |
| `uniformValue` | `table { ... }` | Function1 — table CSV |
| `type` (table) | `csv` | Format de lecture |
| `nHeaderLine` | `0` | Nombre de lignes d'en-tête à ignorer |
| `columns` | `(0 1)` | Colonnes (0=temps, 1=valeur) |
| `file` | `"constant/..."`. | Chemin du CSV (guillemis requis) |
| `separator` | `","` | Séparateur CSV (guillemis requis) |
| `mergeSeparators` | `false` | Fusion des séparateurs multiples |
| `interpolationScheme` | `linear` | Interpolation temporelle |
| `value` | `uniform 300` | Valeur initiale (fallback) |

### 4.2. Function1 `table` (CSV) — Format vecteur

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
            columns         (0 (1 2 3));
            file            "constant/inlet_velocity.csv";
            separator       ",";
            mergeSeparators false;
            interpolationScheme linear;
        }
        value           uniform (1 0 0);
    }
}
```

**Différence :** `columns (0 (1 2 3))` — le tuple `(1 2 3)` indique les 3 colonnes vecteur.

### 4.3. `nonuniformList` — Format spatial

Fichiers générés dans les répertoires temporels (`<time>/T`) :

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  13
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0.05";
    object      T;
}

dimensions      [0 0 0 1 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           nonuniform List<scalar>
<number_of_faces>
(
<valeur_face_0>
<valeur_face_1>
...
<valeur_face_N>
);
    }
    outlet
    {
        type            zeroGradient;
    }
    ...
}
```

Pour un champ vecteur (`U`) :

```cpp
        value           nonuniform List<vector>
<number_of_faces>
(
(vx0 vy0 vz0)
(vx1 vy1 vz1)
...
);
```

---

## 5. API Complète

### 5.1. Fonctions de haut niveau

| Fonction | Description | Module |
| :--- | :--- | :--- |
| `set_csv_condition(...)` | BC uniforme variable dans le temps | `foampilot.boundaries` |
| `set_spatial_csv_condition(...)` | BC spatiale interpolée | `foampilot.boundaries` |

### 5.2. Fonctions utilitaires

| Fonction | Description | Module |
| :--- | :--- | :--- |
| `write_csv_table(...)` | Écrit CSV dans `constant/` | `foampilot.boundaries` |
| `make_uniform_fixed_value_bc(...)` | Dict OpenFOAM pour BC scalaire | `foampilot.boundaries.csv_boundary_condition` |
| `make_uniform_fixed_value_vector_bc(...)` | Dict OpenFOAM pour BC vecteur | `foampilot.boundaries.csv_boundary_condition` |

### 5.3. Classes

| Classe | Description | Module |
| :--- | :--- | :--- |
| `CsvTimeSeries` | Gestion de séries temporelles CSV | `foampilot.boundaries` |

### 5.4. Fonctions avancées (bas niveau)

| Fonction | Description | Usage |
| :--- | :--- | :--- |
| `_read_openfoam_mesh(case_path)` | Lit les points et faces du maillage | Interne |
| `_compute_face_centres(...)` | Calcule les centres de faces | Interne |
| `_format_nonuniform_scalar(values)` | Formate une liste non-uniforme scalaire | Interne |
| `_format_nonuniform_vector(values)` | Formate une liste non-uniforme vecteur | Interne |
| `_write_spatial_field_from_template(...)` | Écrit un champ spatial depuis template | Interne |
| `_write_spatial_field(...)` | Écrit un champ spatial simple | Interne |

---

## 6. Workflows Typiques

### 6.1. Scalaire avec énergie (transient)

```python
from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit
from foampilot.boundaries import set_csv_condition
import pandas as pd

solver = Solver(case_path)
solver.compressible = False
solver.energy_activated = True
solver.turbulence_model = "laminar"
solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
solver.constant.transportProperties.Pr = 0.85

# 1. Configurer le solveur et les fichiers système
solver.system.write()
solver.constant.write()

# 2. Maillage
blockmesh.run()

# 3. Conditions aux limites
solver.boundary.initialize_boundary()
solver.boundary.set_raw_condition("inlet", "T", {"type": "fixedValue", "value": "uniform 0"})
solver.boundary.set_raw_condition("inlet", "U", {"type": "fixedValue", "value": "uniform (1 0 0)"})
solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
# ... autres BCs ...

# 4. Appliquer la BC CSV (écrase l'entrée inlet T)
set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data=csv_path,
    time_column=0,
    value_column=1,
    default_value=350,
)

# 5. Écrire les fichiers de conditions aux limites
solver.boundary.write_boundary_conditions()

# 6. Lancer la simulation
solver.run_simulation()
```

### 6.2. Spatial (steady-state)

```python
solver = Solver(case_path)
solver.energy_activated = True
solver.transient = False  # steady-state

solver.system.write()
solver.constant.write()
blockmesh.run()

solver.boundary.initialize_boundary()
solver.boundary.set_raw_condition("inlet", "T", {"type": "fixedValue", "value": "uniform 0"})
solver.boundary.set_raw_condition("inlet", "U", {"type": "fixedValue", "value": "uniform (1 0 0)"})
solver.boundary.set_raw_condition("outlet", "p", {"type": "fixedValue", "value": "uniform 0"})
# ... autres BCs ...

# Écrire d'abord les conditions de base
solver.boundary.write_boundary_conditions()

# Ensuite appliquer la BC spatiale (écrase 0/T et crée <time>/T)
set_spatial_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data=csv_file,
    time_column=0,
    spatial_columns=[1, 2, 3, 4],
    default_value=300,
    interpolation_method="nearest",
)

solver.run_simulation()
```

---

## 7. Limitations et Erreurs Courantes

| Problème | Cause | Solution |
| :--- | :--- | :--- |
| `foamRun` échoue : "invalid first character" | BC `uniformValue` formattée comme un dict Python | Utiliser `set_csv_condition` (pas de manuel) |
| `foamRun` échoue : "expected string, found" | `file` ou `separator` non guiluminés | foampilot ajoute automatiquement les guillemets |
| T n'est pas résolu par OpenFOAM | Pas de `scalarTransport` dans `system/functions` | Activer `energy_activated=True` |
| T a des valeurs uniformes (pas de transport) | `incompressibleFluid` sans energy activé | Vérifier `solver.energy_activated` |
| Fichier `0/T` manquant | `write_boundary_conditions()` non appelé | Appeler avant `set_spatial_csv_condition()` |
| `import scipy` échoue | SciPy non installé | `pip install scipy` |
| `columns` format incorrect pour vecteur | Utilisation de `(0, 1, 2, 3)` au lieu de `(0 (1 2 3))` | Utiliser `value_columns=[...]` |

---

## 8. Référence des Paramètres CSV

### 8.1. `set_csv_condition()` — Tous les paramètres

```
set_csv_condition(
    boundary: Boundary          # [requis] Objet boundary du solver
    patch_name: str             # [requis] Nom du patch
    field: str                   # [requis] Nom du champ ("T", "U", "p")
    data: str|Path|DataFrame    # [requis] Source de données CSV
    time_column: int|str = 0    # Colonne temps (défaut: 0)
    value_column: int|str = None # Colonne valeur scalaire
    value_columns: list = None   # Colonnes valeur vecteur (3 éléments)
    header_lines: int = 0        # Lignes d'en-tête à ignorer
    separator: str = ","         # Séparateur CSV
    out_of_bounds: str = "clamp" # "clamp"|"error"|"warn"|"zero"|"repeat"
    interpolation_scheme: str = "linear"  # "linear"|"spline"
    default_value: float|str = None        # Valeur fallback
    csv_filename: str = None   # Nom du fichier dans constant/
)
```

### 8.2. `set_spatial_csv_condition()` — Tous les paramètres

```
set_spatial_csv_condition(
    boundary: Boundary           # [requis]
    patch_name: str              # [requis]
    field: str                    # [requis]
    data: str|Path|DataFrame      # [requis]
    time_column: int|str = 0      # Colonne temps
    spatial_columns: list = None  # Colonnes [time, x, y, z, value]
    face_id_column: int|str = None  # Colonne ID de face (long format)
    value_column: int|str = None    # Colonne valeur (long format)
    header_lines: int = 0
    separator: str = ","
    default_value: float|str = None
    interpolation_method: str = "linear"  # "linear"|"nearest"|"cubic"
)
```
