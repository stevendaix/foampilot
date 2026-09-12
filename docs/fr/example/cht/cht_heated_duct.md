# CHT Heat Exchanger – FoamPilot Tutorial

## Vue d’ensemble

Ce tutoriel présente un **flux de travail complet de conjugate heat transfer (CHT)** en utilisant
**FoamPilot** et **OpenFOAM 13** (`chtMultiRegionFoam`). Il modélise un écoulement d'air compressible,
laminaire, en régime stationnaire dans un conduit chauffé couplé à une paroi solide en cuivre.

Le **module CHT de FoamPilot** (`foampilot.cht`) fournit des classes dédiées :

- `ChtSolver` — solveur multi-région pour le transfert de chaleur conjugé
- `FluidRegion` — domaine fluide (heRhoThermo, compressible)
- `SolidRegion` — domaine solide (heSolidThermo)
- `CoupledInterface` — interface de couplage thermique fluide-solide
- `FixedTemperatureBC`, `CoupledTemperatureBC`, etc. — aides pour les conditions aux limites
- `calc_nusselt_number`, `calc_heat_transfer_coefficient`, `calc_thermal_resistance` — post-traitement

Moteur de rapport FoamPilot

Le `CFDReportGenerator` s'intègre avec :

- `LatexDocument` — rapports LaTeX/PDF via PyLaTeX
- `ScientificDocument` / `TypstRenderer` — documents scientifiques basés sur Typst

Le `CFDReportGenerator` fournit :

- `add_statistic()` — enregistrer des statistiques scalaires (Re, Nu, h, etc.)
- `add_figure()` — enregistrer des images
- `add_table()` — enregistrer des tableaux de données
- `collect_time_series()` — collecter des statistiques de champ au cours des pas de temps
- `collect_region_statistics()` — statistiques de champ par région
- `save_html_report()` — rapport HTML interactif avec Plotly
- `save_latex_report()` — rapport LaTeX avec tableaux et figures
- `save_typst_report()` — document scientifique Typst

📁 **Location**: `foampilot/tutorials/09_CHT_heatedDuct/`

---

## 1. Prérequis

- OpenFOAM 13 installé et accessible
- FoamPilot installé (`pip install -e .`)
- Dépendances Python : `pyvista`, `numpy`, `pandas`, runtime `OpenFOAM`

---

## 2. Physique du cas

- **Géométrie** : Échangeur de chaleur shell-and-tube avec trois participants (fluid-inner, fluid-outer, solid)
  - **Taille du domaine** : -0.649 × 0.649 × (-3.45 à 3.45) m
  - **Fluid-Inner** : Fluide de type eau (ρ₀=1027 kg/m³, Cp=4195 J/kg·K, Pr=2.289, μ=3.645e-4 Pa·s)
  - **Fluid-Outer** : Même propriétés d'eau, température d'entrée différente (353 K vs 283 K)
  - **Solid** : Solveur structural CalculiX pour les parois des tubes
- **Écoulement** : Régime stationnaire, laminaire (Re ≈ 13,000 basé sur le diamètre intérieur du tube 0.025 m)
- **Thermo** : heRhoThermo, hConst, équation d'état perfectFluid
- **Couplage** : preCICE avec mapping nearest-neighbor
  - Interface : Solid-to-Fluid-Inner et Solid-to-Fluid-Outer
  - Échange de données : Sink-Temperature, Heat-Transfer-Coefficient (couplage implicite)

### 2.1 Équations gouvernantes

**Continuity (incompressible, Boussinesq) :**

$$
\nabla \cdot \mathbf{u} = 0
$$

**Momentum (buoyantSimpleFoam) :**

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p_{rgh} + \nabla \cdot \left[ \mu_{eff} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right] + \rho \mathbf{g}
$$

**Energy :**

$$
\frac{\partial (\rho h)}{\partial t} + \nabla \cdot (\rho h \mathbf{u}) = \nabla \cdot \left( \frac{\kappa}{Pr} \nabla h \right)
$$

**Pression modifiée :**

$$
p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}
$$

### 2.2 Conditions aux limites

| Patch | Champ | Condition | Valeur |
|-------|-------|-----------|--------|
| inlet (inner) | U | fixedValue | (0, 0, -0.002) m/s |
| inlet (inner) | T | fixedValue | 283 K |
| inlet (outer) | T | fixedValue | 353 K |
| outlet | T | zeroGradient | — |
| interface | T | mixed | refValue=293 K, frac=0.5 |
| adiabatic | T | zeroGradient | — |

### 2.3 Configuration preCICE

La configuration preCICE utilise un **schéma de couplage implicite** :

- **Données échangées** : Sink-Temperature, Heat-Transfer-Coefficient
- **Mapping** : nearest-neighbor (contrainte cohérente)
- **Convergence** : couplage parallel-explicit (pseudo pas de temps vers l'état stationnaire)

---

## 3. Flux de travail

### 3.1 Configuration du cas avec le solveur CHT

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion, CoupledInterface

fluid_region = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
    thermophysical_model="heRhoThermo",
    equation_of_state="perfectGas",
)

solid_region = SolidRegion(
    name="solid",
    temperature=350.0,
    thermal_conductivity=380.0,   # W/(m·K) — cuivre
    density=8960.0,               # kg/m³
    specific_heat=385.0,          # J/(kg·K)
)

solver = ChtSolver(
    case_path=case_path,
    solver_name="chtMultiRegionFoam",
    regions=[fluid_region, solid_region],
    interfaces=[CoupledInterface(...)],
)

solver.system.controlDict.start_time = 0
solver.system.controlDict.end_time = 1.0
solver.system.controlDict.delta_t = 5e-4
solver.system.controlDict.application = "chtMultiRegionFoam"

solver.setup_case()
solver.write_case()
```

### 3.2 Génération du maillage

Le maillage est généré via une configuration JSON :

```python
from foampilot import Meshing

mesh = Meshing(case_path, mesher="blockMesh")
mesh.mesher.load_from_json(case_path / "block_mesh.json")
mesh.mesher.write(file_path=case_path / "system" / "blockMeshDict")
solver.run_command(["blockMesh"], log_filename="log.blockMesh")
```

### 3.3 Configuration multi-région

```python
solver.run_command(["createZones"], log_filename="log.createZones")
solver.run_command(["splitMeshRegions", "-cellZones", "-defaultRegionName", "fluid"],
                   log_filename="log.splitMeshRegions")
solver.run_command(["foamSetupCHT"], log_filename="log.foamSetupCHT")
```

### 3.4 Exécution de la simulation

```python
solver.run_simulation(nb_proc=1)
```

### 3.5 Conversion VTK

```python
solver.run_command(["foamToVTK", "-region", "fluid", "-latestTime",
                    "-fields", "(T U p k omega)"],
                   log_filename="log.foamToVTK_fluid")
solver.run_command(["foamToVTK", "-region", "solid", "-latestTime",
                    "-fields", "(T)"],
                   log_filename="log.foamToVTK_solid")
```

---

## 4. Post-traitement

Le script de post-traitement (`run_post.py`) utilise les fonctions d'analyse CHT de foampilot :

```python
from foampilot.cht import (
    calc_nusselt_number,
    calc_heat_transfer_coefficient,
    calc_thermal_resistance,
    calc_total_heat_balance,
    calc_temperature_contour,
)
```

### 4.1 Résultats clés

| Metric | Value | Reference |
|--------|-------|-----------|
| Interface T (fluid side) | 293.00 K | preCICE reference |
| Interface T (solid side) | 293.00 K | preCICE reference |
| Heat transfer coefficient h | Variable | Coupled via preCICE |
| Mass flow rate (inner) | ~0.005 | kg/s |
| Mass flow rate (outer) | ~0.15 | kg/s |
| Temperature difference ΔT | 70 K | 353−283 K |

### 4.2 Statistiques de température

| Region | T_min (K) | T_max (K) | T_mean (K) |
|--------|-----------|-----------|------------|
| Fluid-Inner | 283.00 | 353.00 | ~293 |
| Fluid-Outer | 283.00 | 353.00 | ~318 |
| Solid | 283.00 | 353.00 | ~303 |

### 4.3 Statistiques du maillage

| Propriété | Inner Fluid | Outer Fluid |
|----------|-------------|-------------|
| Cells | ~100,000 | ~150,000 |
| Points | 37,894 (inner) | 95,000+ (outer) |
| Faces | ~1,084,000 | ~1,700,000 |
| Patches | inlet, outlet, interface, adiabatic | inlet, outlet, interface, adiabatic |

---

## 5. Génération de rapport

Le `CFDReportGenerator` de FoamPilot automatise la création de rapports. Le tutoriel CHT
génère un rapport complet avec :

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="CHT Heated Duct Report",
    author="FoamPilot",
)

# Add key statistics
report.add_statistic("Nu", 0.2597, "", "Nusselt number")
report.add_statistic("h", 3.38, "W/(m²·K)", "Heat transfer coefficient")
report.add_statistic("R_th", 0.2963, "K/W", "Thermal resistance")
report.add_statistic("T_interface", 350.0, "K", "Interface temperature")

# Add figures
report.add_figure("postProcessing/fluid_temperature_contour.png",
                  "Temperature contour (fluid)")
report.add_figure("postProcessing/solid_temperature_contour.png",
                  "Temperature contour (solid)")
report.add_figure("postProcessing/cht_temperature_contour.png",
                  "CHT temperature overlay")

# Generate LaTeX report
report.save_latex_report(compile_pdf=True)

# Generate interactive HTML report
report.save_html_report()
```

### 5.1 Types de rapports

| Méthode | Sortie | Fonctionnalités |
|--------|--------|-----------------|
| `save_latex_report()` | `.tex` / `.pdf` | LaTeX via PyLaTeX, tableaux, figures |
| `save_typst_report()` | `.typ` | Document scientifique Typst |
| `save_html_report()` | `.html` | Figures interactives Plotly, tableaux intégrés |

---

## 6. Fichiers

| File | Description |
|------|-------------|
| `run.py` | Script principal du tutoriel (configuration CHT, maillage, simulation) |
| `run_post.py` | Post-traitement et analyse |
| `block_mesh.json` | Configuration géométrique pour `BlockMesher` |
| `README.md` | Cette documentation |

---

## 7. Exécution

```bash
cd foampilot/tutorials/09_CHT_heatedDuct
python run.py
python run_post.py
```

---

## 8. Sorties attendues

```
postProcessing/
├── temperature_statistics.csv
├── temperature_profile.csv
├── temperature_profile_combined.csv
├── fluid_temperature_contour.png
├── solid_temperature_contour.png
├── cht_temperature_contour.png
└── CHT_Report.md
```
