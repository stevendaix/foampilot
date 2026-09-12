# Flottabilité thermique (buoyantSimpleFoam) — Tutoriel FoamPilot

## Aperçu

Ce tutoriel simule la **convection naturelle** dans une pièce chauffée en utilisant `buoyantSimpleFoam` avec l'**approximation de Boussinesq**. Il illustre le couplage entre l'écoulement et le transfert de chaleur sous l'effet de la gravité.

FoamPilot automatise :

- Activation de la gravité et flottabilité Boussinesq
- Patchs de paroi isothermes
- Configuration de l'équation d'énergie

📁 **Emplacement**: `foampilot/tutorials/08_thermalBuoyancy/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé

---

## 2. Physique du cas

- **Domaine** : Pièce (4 m × 4 m × 3 m)
- **Fluide** : Air (Boussinesq incompressible)
- **Paroi chaude** : 350 K (paroi gauche)
- **Paroi froide** : 300 K (paroi droite)
- **Autres parois** : Adiabatiques (gradient nul)
- **Gravité** : 9.81 m/s² (direction -Z)

### 2.1 Approximation de Boussinesq

La variation de densité est modélisée par :

$$
\rho = \rho_0 [1 - \beta (T - T_0)]
$$

Où :
- `ρ₀` — densité de référence
- `β` — coefficient d'expansion thermique
- `T₀` — température de référence

Le terme de flottabilité dans l'équation de la quantité de mouvement :

$$
\frac{\partial (rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p_{rgh} + \nabla \cdot (\mu_{eff} \nabla \mathbf{u}) + \rho \mathbf{g}
$$

### 2.2 Nombre de Rayleigh

$$
Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}
$$

Pour ce cas (ΔT = 50 K, L = 4 m) :

$$
Ra = \frac{9.81 \times 3.2 \times 10^{-3} \times 50 \times 4^3}{1.5 \times 10^{-5} \times 2.2 \times 10^{-5}} \approx 9.7 \times 10^9
$$

Ceci se situe dans le régime de **convection naturelle turbulente** (Ra > 1e9), confirmant la nécessité d'un modèle de turbulence (k-epsilon).

### 2.3 Équations gouvernantes

Énergie :

$$
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T
$$

Pression (modifiée hydrostatiquement) :

$$
p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}
$$

---

## 3. Flux de travail

### 3.1 Initialisation du solveur

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = True
solver.turbulence_model = "kEpsilon"
```

Définir `solver.with_gravity = True` active :

- le solveur `buoyantSimpleFoam`
- la densité de Boussinesq dans l'équation de quantité de mouvement
- la variable de pression `p_rgh`

### 3.2 Conditions aux limites

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(0.1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)

# Paroi chaude à 350 K
solver.boundary.set_raw_condition("hotWall", "T", {"type": "fixedValue", "value": "350"})
# Paroi froide à 300 K
solver.boundary.set_raw_condition("coldWall", "T", {"type": "fixedValue", "value": "300"})
```

La méthode `set_raw_condition` de FoamPilot permet la spécification directe des dictionnaires OpenFOAM pour des cas complexes.

### 3.3 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Cellules de convection naturelle

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Rapport de flottabilité thermique",
    author="FoamPilot",
)
report.add_statistic("Ra", 9.7e9, "", "Nombre de Rayleigh")
report.add_statistic("T_hot", 350.0, "K", "Température paroi chaude")
report.add_statistic("T_cold", 300.0, "K", "Température paroi froide")
report.save_html_report(filename="buoyancy_report.html")
```

### 4.2 Rapport LaTeX

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Convection naturelle dans une pièce chauffée",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Simulation de flottabilité selon Boussinesq avec buoyantSimpleFoam.")
doc.add_section("Équations gouvernantes", "")
doc.add_equation(
    r"Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}",
    caption="Nombre de Rayleigh",
)
doc.add_equation(
    r"p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}",
    caption="Pression modifiée",
)
doc.add_section("Conditions aux limites", "")
doc.add_table(
    [["hotWall", "350", "K"], ["coldWall", "300", "K"], ["Other walls", "adiabatic", ""]],
    headers=["Patch", "Température", "Condition"],
    caption="Conditions de paroi",
)
doc.generate_document(output_format="pdf")
```

### 4.3 Document scientifique Typst

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Convection naturelle", "FoamPilot")
doc.add_section("Introduction", "Analyse de l'écoulement entraîné par la flottabilité.")
doc.add_equation(
    r"Ra = g \beta \Delta T L^3 / (\nu \alpha)",
    caption="Nombre de Rayleigh",
    label="eq:rayleigh",
)
doc.add_table(
    [["T_hot", "350 K"], ["T_cold", "300 K"], ["g", "9.81 m/s²"]],
    headers=["Paramètre", "Valeur"],
    caption="Paramètres de la simulation",
)
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 5. Résultats attendus

| Quantité | Valeur attendue |
|----------|-----------------|
| Cellules de convection naturelle | 2–4 cellules de convection |
| Vitesse de montée de l'air chaud | ~0.1–0.3 m/s |
| Profil de température au plan médian | Linéaire de 350 K à 300 K |
| Vitesse près de la paroi chaude | Vers le haut (0.05–0.15 m/s) |

---

## 6. Exécution

```bash
cd foampilot/tutorials/08_thermalBuoyancy
python run.py
python report_generator.py
```
