# MotorBike External Aerodynamics — FoamPilot Tutorial

## Aperçu

Ce tutoriel simule le **flux externe à grande vitesse autour d'une moto**
en utilisant `simpleFoam` (k-omega SST). Il démontre le maillage résolu au mur
et la prédiction du sillage.

FoamPilot automatise :

- Configuration d'entrée à grande vitesse (30 m/s)
- Conditions aux limites pour parois et sol mobile
- Suivi de la traînée et de la portance

📁 **Location**: `foampilot/tutorials/07_motorBike/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé

---

## 2. Physique du cas

- **Géométrie** : Modèle de moto avec surface routière
- **Écoulement** : Incompressible, turbulent, stationnaire
- **Vitesse** : 30 m/s (108 km/h, vitesse autoroutière)
- **Modèle de turbulence** : k-omega SST
- **Intensité de turbulence** : 5%

### 2.1 Paramètres sans dimension

Nombre de Reynolds basé sur la longueur du véhicule (L = 2.0 m) :

$$
Re_L = \frac{U L}{\nu} = \frac{30 \times 2}{1.5 \times 10^{-5}} = 4 \times 10^6
$$

Coefficient de traînée :

$$
C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}
$$

Où :
- `Fd` — force de traînée
- `A` — surface frontale (~0.7 m² pour une moto)

### 2.2 Prédiction du sillage

En aval de la moto, le sillage présente :

- Déficit de vitesse
- Mélange turbulent
- Récupération de pression

$$
T_{aw} = T_\infty \left[ 1 + r \frac{\gamma - 1}{2} M_\infty^2 \right]
$$

(Formule de température de récupération pour écoulement grande vitesse)

---

## 3. Flux de travail

### 3.1 Initialisation du solveur

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 Conditions aux limites

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(30, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*wheels.*|.*moving.*",
    condition_type="wall",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*road.*",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

Le système de motifs génériques (wildcard) de FoamPilot gère des patches complexes :

- `.*wheels.*` — correspond à tous les patches de roues
- `.*moving.*` — correspond aux surfaces mobiles
- `.*road.*` — correspond au plancher/sol

### 3.3 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Coefficients de forces

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="MotorBike Aerodynamics",
    author="FoamPilot",
)
report.add_statistic("Re_L", 4e6, "", "Reynolds number")
report.add_statistic("Cd_expected", 0.35, "", "Expected drag coefficient")
```

### 4.2 Rapport LaTeX

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="MotorBike External Aerodynamics",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Aerodynamic analysis of motorcycle at 30 m/s.")
doc.add_section("Method", "")
doc.add_equation(r"Re_L = \frac{UL}{\nu}")
doc.add_section("Results", "")
doc.add_table(
    [["Drag coeff", "0.35"], ["Lift coeff", "0.05"]],
    headers=["Coefficient", "Value"],
    caption="Aerodynamic coefficients",
)
for img in ["pressure_contour.png", "velocity_vectors.png"]:
    doc.add_figure(img, caption=img.replace("_", " ").title())
doc.generate_document(output_format="pdf")
```

---

## 5. Résultats attendus

| Quantité | Attendu |
|----------|---------|
| Coefficient de traînée (Cd) | 0.30–0.40 |
| Traînée frontale | ~200–250 N |
| Taille du sillage | ~3–5 longueurs de moto |
| Récupération de pression à la queue | ~70–80% |

---

## 6. Exécution

```bash
cd foampilot/tutorials/07_motorBike
python run.py
python report_generator.py
```
