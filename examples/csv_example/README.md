# Exemples CSV foampilot

Ce dossier contient des exemples d'utilisation des conditions aux limites
variables dans le temps à partir de fichiers CSV ou DataFrames.

## Exemples

### `run_uniform_scalar.py` — CSV uniforme scalaire

Deux variantes :
- **Stationnaire** (`--steady`) : CSV à une ligne, valeur constante
- **Transitoire** (défaut) : CSV avec pas de temps multiples, interpolation linéaire

```bash
# Stationnaire
python run_uniform_scalar.py --steady

# Transitoire
python run_uniform_scalar.py
```

### `run_uniform_vector.py` — CSV uniforme vectoriel

Cas transitoire avec vitesse d'entrée variable dans le temps.

```bash
python run_uniform_vector.py
```

### `run_spatial.py` — CSV spatial avec interpolation

Condition spatiale interpolée sur le maillage du patch à partir d'un nuage
de points source.

```bash
python run_spatial.py
```

## Fonctionnalités

| Exemple | Champ | Type | Mode |
|---------|-------|------|------|
| `run_uniform_scalar.py --steady` | T | uniforme | stationnaire |
| `run_uniform_scalar.py` | T | uniforme | transitoire |
| `run_uniform_vector.py` | U | uniforme | transitoire |
| `run_spatial.py` | T | spatial | transitoire |

## API utilisée

```python
from foampilot.boundaries import set_csv_condition, set_spatial_csv_condition

# Uniforme scalaire
set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data="inlet_temperature.csv",
    time_column=0,
    value_column=1,
    separator=",",
    default_value=350,
)

# Uniforme vectoriel
set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="U",
    data="inlet_velocity.csv",
    time_column=0,
    value_columns=[1, 2, 3],
    separator=",",
    default_value="(1 0 0)",
)

# Spatial (point cloud interpolé sur le maillage)
set_spatial_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data="inlet_temperature_spatial.csv",
    time_column=0,
    spatial_columns=[1, 2, 3, 4],
    interpolation_method="nearest",
)
```
