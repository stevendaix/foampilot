# SimpleCar Turbulent Flow — FoamPilot Tutorial

## Vue d’ensemble

Ce tutoriel présente une simulation de flux turbulent RANS stationnaire autour d'une géométrie simplifiée de voiture en utilisant **FoamPilot** et le solveur `simpleFoam` avec le modèle de turbulence **k-omega SST**.

FoamPilot automatise :

- la configuration des conditions aux limites turbulentes avec l'intensité de turbulence
- les function objects (moyenne de champ, contrôles à l'exécution)
- la surveillance des forces et des coefficients de pression

📁 **Emplacement**: `foampilot/tutorials/02_simpleCar_turbulent/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé
- `classy_blocks` (optionnel, pour la géométrie)

---

## 2. Propriétés du cas

- **Géométrie** : aérodynamique externe d'une voiture simplifiée
- **Écoulement** : incompressible, turbulent, stationnaire
- **Vitesse à l'entrée** : 30 m/s (108 km/h vent de face)
- **Modèle de turbulence** : k-omega SST
- **Intensité de turbulence** : 5%

### 2.1 Équations gouvernantes

RANS avec approximation de Boussinesq :

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nabla \cdot \left[ \nu_{eff} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right]
$$

### 2.2 Modèle k-omega SST

Énergie cinétique turbulente :

$$
\frac{\partial (\rho k)}{\partial t} + \frac{\partial (\rho u_j k)}{\partial x_j} = P_k - \beta^* \rho k \omega
$$

Taux de dissipation spécifique :

$$
\frac{\partial (\rho \omega)}{\partial t} + \frac{\partial (\rho u_j \omega)}{\partial x_j} = \alpha S_\omega
$$

### 2.3 Paramètres sans dimension

Nombre de Reynolds du tunnel (basé sur la longueur de la voiture L = 4.5 m) :

$$
Re_L = \frac{U L}{\nu} = \frac{30 \times 4.5}{1.5 \times 10^{-5}} \approx 9 \times 10^6
$$

---

## 3. Flux de travail

### 3.1 Configuration du solveur

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
```

La `velocityInlet` de FoamPilot avec `turbulence_intensity` calcule automatiquement les valeurs d'entrée de `k` et `omega` :

$$
k = \frac{3}{2} (I \cdot U)^2, \quad \omega = \frac{\sqrt{k}}{L_{ref} \cdot 0.016}
$$

### 3.3 Objets de fonction (function objects)

FoamPilot permet d'ajouter des function objects pour la surveillance :

```python
solver.system.functions.velocity_field_average = {
    "type": "fieldAverage",
    "enabled": True,
    "fields": [("U", "U_mean", "U_rms")],
}
```

### 3.4 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Coefficients de force

FoamPilot surveille les coefficients de traînée et de portance via des function objects :

```
forces {
    type            forces;
    functionObjectLibs ("libforces.so");
    patches          (car body walls);
    rho            rhoInf;  // incompressible
    liftDir        (0 0 1);
    dragDir        (1 0 0);
    CofR           (0 0 0);
}
```

### 4.2 Coefficient de pression

$$
C_p = \frac{p - p_\infty}{\frac{1}{2} \rho U_\infty^2}
$$

### 4.3 Génération de rapport

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="SimpleCar Aerodynamics Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("External aerodynamics simulation of a simplified car.")
doc.add_section("Drag Coefficient", f"Cd = {cd_value:.4f}")
doc.add_section("Pressure Distribution", "")
for img in ["pressure_contour.png", "velocity_contour.png"]:
    doc.add_figure(img, caption=img.replace("_", " ").title())
doc.generate_document(output_format="pdf")
```

---

## 5. Résultats attendus

| Grandeur | Valeur attendue |
|----------|-----------------|
| Drag coefficient (Cd) | 0.25–0.35 |
| Lift coefficient (Cl) | 0.1–0.2 |
| Max. Cp | ~1.2 |
| Reattachment length behind car | ~2–3 car lengths |

---

## 6. Exécution

```bash
cd foampilot/tutorials/02_simpleCar_turbulent
python run.py
python report_generator.py
```
