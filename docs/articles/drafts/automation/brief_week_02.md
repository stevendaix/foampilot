# Brief — Semaine 2 : "Automatiser OpenFOAM en 20 lignes de Python"

**Objectif** : Montrer que foampilot permet de créer un cas OpenFOAM complet en quelques lignes de code, avec une validation physique.
**Angle** : Tutoriel pratique, "best practices", comparaison avant/après, cas Poiseuille plan.

---

## Structure proposée

### Introduction — Le réveil à 2h du matin

Storytelling : Vous avez une simulation qui doit tourner avant une réunion. Vous copiez un cas, éditez 7 fichiers, oubliez un point-virgule, et la simulation crash. Promesse : avec foampilot, ce cas se crée en 20 lignes de Python.

**Nouveau** : Lier à l'article précédent (physique) : "Maintenant que vous savez calculer Re et choisir votre modèle de turbulence, voici comment automatiser la création du cas."

### Section 1 — Le problème physique : choisir les bons paramètres

Avant même de penser au code, il faut choisir la physique. Présenter le cas de l'écoulement de Poiseuille plan :

- **Fluide** : air à 20°C (ou eau)
- **Viscosité cinématique** : ν = 1.52×10⁻⁵ m²/s (air) ou ν = 1×10⁻⁶ m²/s (eau)
- **Vitesse d'entrée** : U = 1 m/s
- **Hauteur du canal** : H = 1 m
- **Nombre de Reynolds** : Re = UH/ν = 65 800 (air) ou 1×10⁶ (eau)

**Message** : Ces calculs ne doivent pas être faits à la main dans un tableur. Ils doivent être dans le script.

### Section 2 — Le workflow foampilot avec physique

Présenter le code complet avec physique :

```python
from foampilot import Solver, Meshing, FluidMechanics, ValueWithUnit

# 1. Définir le fluide et calculer ses propriétés
fluid = FluidMechanics(
    fluid_name=FluidsList.Air,
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)
props = fluid.get_fluid_properties()
nu = props['kinematic_viscosity']

# 2. Définir le solveur
solver = Solver(case_path="./poiseuille")
solver.transient = False
solver.turbulence_model = "laminar"  # Re = 65 800 → transition possible

# 3. Injecter la physique
solver.constant.transportProperties.nu = nu

# 4. Conditions aux limites
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    "inlet", "velocityInlet", velocity=(1, 0, 0)
)
solver.boundary.apply_condition_with_wildcard(
    "outlet", "pressureOutlet"
)

# 5. Écrire et lancer
solver.write_case()
solver.boundary.write_boundary_conditions()
solver.run_simulation(nb_proc=2)
```

**Message** : La physique est calculée en Python, pas copiée d'un tableur.

### Section 3 — Validation : comparer à la solution analytique

Présenter la solution analytique de Poiseuille :

```
u(y) = (1 / 2ν) × (dp/dx) × (h² - y²)
```

Montrer comment foampilot valide ses résultats :

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./poiseuille")
mesh = fp.read_direct(time_step=100)

# Extraire le profil de vitesse sur la ligne médiane
centreline = mesh.slice(normal="y", origin=(0, 0, 0))
U_analytical = (1 / (2 * nu)) * dpdx * (H**2 - centreline.points[:, 2]**2)

# Comparaison
error = np.mean(np.abs(centreline.point_data["U"][:, 0] - U_analytical))
print(f"Erreur moyenne : {error:.4f} m/s")
```

**Message** : Avec foampilot, la validation fait partie du workflow, pas une étape séparée.

### Section 4 — Ce qui se passe sous le capot

Montrer ce que `write_case()` génère :
- `system/controlDict`, `fvSchemes`, `fvSolution`
- `0/U`, `0/p`
- `constant/transportProperties` avec ν injecté depuis pyfluids
- Toutes les conditions aux limites

Montrer un extrait de `controlDict` généré et un extrait de `0/U`.

### Section 5 — Étendre le workflow

Montrer comment ajouter le maillage et le rapport :

```python
# Maillage
meshing = Meshing(case_path="./poiseuille", mesher="blockMesh")
meshing.write()

# Rapport PDF avec statistiques physiques
from foampilot.report import CFDReportGenerator
report = CFDReportGenerator(case_path="./poiseuille")
report.add_statistic("Re", Re, "-", "Nombre de Reynolds")
report.add_statistic("nu", nu, "m²/s", "Viscosité cinématique")
report.add_statistic("U_inlet", 1.0, "m/s", "Vitesse d'entrée")
report.save_latex_report(compile_pdf=True)
```

### Conclusion — La CFD reproductible, avec physique

CTA : *"Maintenant que vous savez automatiser un cas complet, je vous emmène dans les coulisses du design de l'API : pourquoi j'ai volontairement répété du code pour rendre foampilot plus prévisible."*

---

## Code à préparer

- [ ] Script complet Poiseuille avec pyfluids (testé)
- [ ] Script de validation analytique (comparaison u(y) numérique vs analytique)
- [ ] Extrait de `controlDict` généré
- [ ] Extrait de `0/U` généré
- [ ] Script de génération de rapport PDF

## Images à préparer

- [ ] Schéma de l'écoulement de Poiseuille plan (2 plaques, vitesse parabolique)
- [ ] Graphique : profil de vitesse numérique vs analytique
- [ ] Capture d'écran du script Python
- [ ] Capture d'écran des fichiers générés (arborescence)
- [ ] Capture d'écran du rapport PDF généré

## Concepts physiques à vérifier

- [ ] Nombre de Reynolds pour le cas Poiseuille
- [ ] Viscosité de l'air à 20°C (pyfluids)
- [ ] Solution analytique de Poiseuille
- [ ] Critères de convergence du solveur

---

## Nouveaux éléments par rapport à la v1

- Intégration de `FluidMechanics` et `pyfluids` dans le tutoriel
- Cas Poiseuille comme exemple concret (pas juste un cas générique)
- Validation analytique intégrée au workflow
- Concepts physiques expliqués avant le code
