# Conditions aux Limites CSV avec foampilot

## Description

Ce tutoriel démontre l'utilisation des conditions aux limites temporelles et spatiales basées sur des fichiers CSV avec le module `csv_boundary_condition` de foampilot.

## Cas d'usage

- **CSV scalaire uniforme** : température d'entrée variable dans le temps (stationnaire et transitoire)
- **CSV vecteur uniforme** : vitesse d'entrée variable dans le temps
- **CSV spatial** : distribution spatiale de température interpolée sur les faces du patch (stationnaire et transitoire)

## Fichiers d'exemple

Les scripts d'exemple se trouvent dans :

```
examples/csv_example/
├── run_uniform_scalar.py      # CSV scalaire (transient + steady)
├── run_uniform_vector.py      # CSV vecteur (transient)
├── run_spatial.py             # CSV spatial (transient)
├── run_spatial_steady.py      # CSV spatial (steady)
├── verify_csv_post.py         # Vérification post-traitement
└── visualize_csv_direct.py    # Visualisation PyVista
```

## Activation de l'énergie (température)

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.energy_activated = True
solver.turbulence_model = "laminar"
solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
solver.constant.transportProperties.Pr = 0.85
```

Le solveur reste `incompressibleFluid`. Le transport de `T` est géré par un `functionObject scalarTransport` dans `system/functions`.

## Utilisation

```bash
cd foampilot/foampilot
PYTHONPATH=src python3 examples/csv_example/run_uniform_scalar.py
PYTHONPATH=src python3 examples/csv_example/run_uniform_scalar.py --steady
PYTHONPATH=src python3 examples/csv_example/run_spatial.py
PYTHONPATH=src python3 examples/csv_example/run_spatial_steady.py
```

## Vérification

```bash
PYTHONPATH=src python3 examples/csv_example/verify_csv_post.py --base-dir examples/csv_example
```
