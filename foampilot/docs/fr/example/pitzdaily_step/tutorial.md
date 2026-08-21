# PitzDaily Backward-Facing Step — Tutoriel FoamPilot

## Vue d'ensemble

Ce tutoriel simule l'écoulement turbulent sur un **backward-facing step**
en utilisant **FoamPilot** et `simpleFoam` (k-omega SST). Le cas valide
la zone de recirculation et la longueur de réattachement.

📁 **Emplacement**: `foampilot/tutorials/03_pitzDaily_step/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé

---

## 2. Physique du cas

- **Géométrie** : canal 2D avec marche en retrait (hauteur de la marche H = 0.012 m)
- **Écoulement** : incompressible, turbulent, stationnaire
- **Vitesse d'entrée** : 1 m/s
- **Modèle de turbulence** : k-omega SST
- **Intensité de turbulence** : 5 %

### 2.1 Points clés de la physique

La marche en retrait génère une **zone de recirculation** en aval
de la marche due à la séparation d'écoulement. Un **point de réattachement** se forme lorsque
l'écoulement inversé se réattache à la paroi en aval.

### 2.2 Paramètres sans dimension

$$
Re_H = \frac{U H}{\nu} = \frac{1 \times 0.012}{1.5 \times 10^{-5}} \approx 800
$$

La longueur de la bulle de recirculation pour un écoulement turbulent à ce Re :

$$
L_r \approx 6.5 H \approx 0.078 \text{ m}
$$

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
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)
```

### 3.3 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Analyse de la zone de recirculation

La zone de recirculation est identifiée par une vitesse axiale négative :

$$
u_x < 0 \quad \text{dans la région de recirculation}
$$

La longueur de réattachement est trouvée à l'emplacement sur la paroi où $u_x = 0$
en aval de la marche.

### 4.2 Génération de rapport

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Backward-Facing Step Report",
    author="FoamPilot",
)

report.add_statistic("Re_H", 800, "", "Hydraulic Reynolds number")
report.add_statistic("L_r_expected", 6.5, "H", "Expected reattachment length ratio")

report.save_html_report(filename="step_report.html")
```

### 4.3 Rapports LaTeX/Typst

```python
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

# LaTeX
doc = LatexDocument("Backward-Facing Step", "FoamPilot",
                    output_dir=case_path)
doc.add_title()
doc.add_section("Recirculation Zone", "Length and velocity analysis.")
doc.generate_document(output_format="tex")

# Typst
tdoc = ScientificDocument("BFS Analysis", "FoamPilot")
tdoc.add_equation(r"L_r = 6.5 H", caption="Reattachment length", label="eq:reattachment")
r = TypstRenderer()
r.render(tdoc)
```

---

## 5. Résultats attendus

| Quantité | Attendu |
|----------|---------|
| Longueur de recirculation (L_r/H) | 6.0–7.0 |
| Point de réattachement x/H | 6.5 |
| Récupération de la vitesse | Vers x/H ≈ 20 |

---

## 6. Exécution

```bash
cd foampilot/tutorials/03_pitzDaily_step
python run.py
python report_generator.py
```
