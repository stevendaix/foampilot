# Brief — Semaine 4 : "Visualiser OpenFOAM sans foamToVTK"

**Objectif** : Montrer comment foampilot lit les résultats OpenFOAM directement avec PyVista, sans conversion intermédiaire, et extrait des quantités physiques.
**Angle** : Tutoriel technique, visualisation, post-processing moderne, extraction de quantités physiques (y+, contrainte de paroi).

---

## Structure proposée

### Introduction — Le goulot d'étranglement de foamToVTK

Storytelling : Vous avez une simulation qui tourne. Vous voulez visualiser les résultats. Vous lancez `foamToVTK`, attendez 10 minutes, puis ouvrez Paraview. Mais si vous pouviez visualiser directement depuis Python, sans conversion, et extraire les quantités physiques qui vous intéressent ?

**Promesse** : Avec `OpenFOAMDirectReader` et PyVista, c'est possible. Et en plus, vous pouvez calculer y+, la contrainte de paroi, le taux de déformation — directement dans le script.

### Section 1 — Le workflow classique (et ses limites)

1. Lancer la simulation OpenFOAM
2. Lancer `foamToVTK` pour convertir en VTK
3. Ouvrir Paraview
4. Charger le fichier VTK
5. Configurer la visualisation

**Problèmes** :
- Temps de conversion (parfois long pour des cas volumineux)
- Espace disque doublé (OpenFOAM + VTK)
- Workflow non automatisable
- Pas de scriptabilité pour les quantités physiques

### Section 2 — La solution : lecture directe

Présenter le reader direct :

```python
import pyvista as pv
from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader

reader = OpenFOAMDirectReader(case_path="./poiseuille")
mesh = reader.read(time_step=100)

# Visualisation directe
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="U", cmap="viridis")
plotter.show()
```

**Avantages** :
- Pas de conversion
- Accès aux champs directement
- Automatisable dans un script
- Intégrable dans un rapport

### Section 3 — Extraire des quantités physiques

C'est là que ça devient intéressant. foampilot ne se contente pas de lire les champs — il calcule des quantités physiques dérivées :

#### 3.1 y+ (distance adimensionnelle à la paroi)

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./poiseuille")
mesh = fp.read_direct(time_step=100)

# Calcul de y+ sur le patch "walls"
mesh_yp = fp.calc_y_plus(
    mesh, 
    wall_patch_name="walls", 
    velocity_field="U", 
    viscosity=1.52e-5  # viscosité cinématique de l'air
)

print(f"y+ min = {mesh_yp.point_data['y_plus'].min():.2f}")
print(f"y+ max = {mesh_yp.point_data['y_plus'].max():.2f}")
print(f"y+ mean = {mesh_yp.point_data['y_plus'].mean():.2f}")
```

**Physique** : y+ doit être < 1 pour les modèles à couche limite résolue (k-omega SST), ou entre 30 et 300 pour les modèles avec fonctions de paroi (k-epsilon).

#### 3.2 Contrainte de paroi (wall shear stress)

```python
# Contrainte de paroi sur la paroi "walls"
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=1.52e-5, 
    wall_normal=[0, 1, 0]
)

tau_w = mesh_wss.cell_data["wall_shear_stress"]
print(f"τ_w max = {tau_w.max():.4f} Pa")
print(f"τ_w mean = {tau_w.mean():.4f} Pa")

# Validation analytique pour Poiseuille plan
# τ_w = 0.5 × ρ × G × H
rho = 1.204  # kg/m³
G = 5.0      # Pa/m (gradient de pression)
H = 1.0      # m
tau_w_analytical = 0.5 * rho * G * H
print(f"τ_w analytique = {tau_w_analytical:.4f} Pa")
```

**Message** : La CFD sans validation, c'est de la sculpture sur glace. Avec foampilot, la validation fait partie du workflow.

#### 3.3 Taux de déformation (strain rate)

```python
# Taux de déformation dans tout le domaine
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
print(f"Taux de déformation max = {mesh_sr.point_data['strain_rate'].max():.4f} 1/s")
```

### Section 4 — Exemple complet : streamlines et coupes

Montrer des exemples avancés :

```python
# Streamlines
mesh = reader.read(time_step=500)
streamlines = mesh.streamlines(
    vectors="U",
    integration_direction="both",
    initial_step_length=0.1,
    max_step_length=0.5
)
streamlines.plot(cmoap="coolwarm")

# Coupe
slice = mesh.slice(normal="z", origin=(0, 0, 0.5))
slice.plot(scalars="p", cmap="RdBu")
```

### Section 5 — CHT et lectures multi-régions

Présenter `CHTDirectReader` :

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)

for name, region_mesh in regions.items():
    print(f"Region {name}: {region_mesh.n_points} points")
    if "T" in region_mesh.point_data:
        print(f"  T_max = {region_mesh.point_data['T'].max():.1f} K")
```

### Section 6 — Automatiser le post-processing

Montrer comment intégrer dans un rapport :

```python
from foampilot.report import CFDReportGenerator

report = CFDReportGenerator(case_path="./poiseuille")

# Ajouter des statistiques physiques
report.add_statistic("Re", 65800, "-", "Nombre de Reynolds")
report.add_statistic("nu", 1.52e-5, "m²/s", "Viscosité cinématique")
report.add_statistic("y+_max", mesh_yp.point_data['y_plus'].max(), "-", "y+ maximum")
report.add_statistic("tau_w", tau_w.mean(), "Pa", "Contrainte de paroi moyenne")

# Générer le PDF
report.save_latex_report(compile_pdf=True)
```

### Conclusion — Le workflow moderne

CTA : *"La visualisation et le post-processing ne sont que des étapes. La semaine prochaine, on s'attaque à un cas complet de CHT multi-région où toutes ces techniques se combinent."*

---

## Code à préparer

- [ ] Script de lecture directe simple
- [ ] Script de calcul de y+ avec validation
- [ ] Script de calcul de contrainte de paroi avec validation analytique
- [ ] Script de visualisation avec streamlines
- [ ] Script CHTDirectReader
- [ ] Script d'intégration dans un rapport

## Images à préparer

- [ ] Schéma du workflow classique (OpenFOAM → foamToVTK → Paraview)
- [ ] Schéma du workflow direct (OpenFOAM → PyVista)
- [ ] Graphique : profil de vitesse dans le canal Poiseuille
- [ ] Capture d'écran de y+ sur les parois (heatmap)
- [ ] Capture d'écran de streamlines dans PyVista
- [ ] Capture d'écran du rapport PDF généré

## Concepts physiques à vérifier

- [ ] Calcul de y+ : formules, limitations
- [ ] Contrainte de paroi : τ_w = μ × du/dy
- [ ] Taux de déformation : γ̇ = √(2×S:S) où S est le tenseur de déformation
- [ ] Comparaison avec solution analytique Poiseuille

## Performance

- [ ] Temps de lecture d'un cas 1M de mailles (à mesurer)
- [ ] Comparaison mémoire foamToVTK vs direct
