# Visualiser OpenFOAM directement avec PyVista : fini foamToVTK

*Comment lire les résultats OpenFOAM directement depuis Python, extraire les quantités physiques (y+, contrainte de paroi, taux de déformation), et automatiser le post-processing sans conversion intermédiaire.*

---

## Introduction : Le goulot d'étranglement de foamToVTK

Vous avez une simulation qui tourne. Vous voulez visualiser les résultats. Vous lancez `foamToVTK`, attendez 10 minutes, puis ouvrez Paraview. Mais si vous pouviez visualiser directement depuis Python, sans conversion, et extraire les quantités physiques qui vous intéressent ?

**Promesse** : Avec `OpenFOAMDirectReader` et PyVista, c'est possible. Et en plus, vous pouvez calculer y+, la contrainte de paroi, le taux de déformation — directement dans le script.

---

## Le workflow classique (et ses limites)

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

---

## La solution : lecture directe

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

---

## Extraire des quantités physiques

C'est là que ça devient intéressant. foampilot ne se contente pas de lire les champs — il calcule des quantités physiques dérivées.

### y+ (distance adimensionnelle à la paroi)

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

### Contrainte de paroi (wall shear stress)

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

### Taux de déformation (strain rate)

```python
# Taux de déformation dans tout le domaine
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
print(f"Taux de déformation max = {mesh_sr.point_data['strain_rate'].max():.4f} 1/s")
```

---

## Exemple complet : streamlines et coupes

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

---

## CHT et lectures multi-régions

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)

for name, region_mesh in regions.items():
    print(f"Region {name}: {region_mesh.n_points} points")
    if "T" in region_mesh.point_data:
        print(f"  T_max = {region_mesh.point_data['T'].max():.1f} K")
```

---

## Automatiser le post-processing

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

---

## Conclusion : Le workflow moderne

La visualisation et le post-processing ne sont que des étapes. Dans le prochain article, on s'attaque à un cas complet de CHT multi-région où toutes ces techniques se combinent.

**Ressources :**
- Dépôt : [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article en cours d'amélioration — version 1.0*
