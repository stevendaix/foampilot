# Lire OpenFOAM directement dans PyVista : architecture du reader et extraction de quantités physiques

*Comment foampilot lit les résultats OpenFOAM directement depuis Python, sans conversion intermédiaire, et extrait des quantités physiques essentielles comme y+, la contrainte de paroi et le taux de déformation.*

---

## Introduction : Le goulot d'étranglement foamToVTK

Vous avez une simulation qui tourne. Vous voulez visualiser les résultats. Vous lancez `foamToVTK`, attendez 10 minutes, puis ouvrez Paraview. Mais si vous pouviez visualiser directement depuis Python, sans conversion, et extraire les quantités physiques qui vous intéressent ?

**Promesse** : Avec `OpenFOAMDirectReader` et PyVista, c'est possible. Et en plus, vous pouvez calculer y+, la contrainte de paroi, le taux de déformation — directement dans le script.

---

## Partie 1 : Le workflow classique (et ses limites)

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

## Partie 2 : La solution : lecture directe

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

## Partie 3 : Sous le capot — architecture du reader

### 3.1 Lecture des fichiers de champ OpenFOAM

Le reader parse les fichiers `volVectorField` et `volScalarField` :

```python
# Extraction du header FoamFile
# Parsing du boundaryField
# Reconstruction du champ interne
```

### 3.2 Reconstruction du maillage polyMesh

```python
# Lecture de points, faces, owner, neighbour
# Construction de la connectivité cellules
# Extraction des patches
```

### 3.3 Gestion des pas de temps

```python
# Listing des répertoires de temps
# Sélection du pas de temps spécifique
# Chargement lazy pour efficacité mémoire
```

---

## Partie 4 : Extraire des quantités physiques

### 4.1 y+ (distance adimensionnelle à la paroi)

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./poiseuille")
mesh = fp.read_direct(time_step=100)

mesh_yp = fp.calc_y_plus(
    mesh, 
    wall_patch_name="walls", 
    velocity_field="U", 
    viscosity=1.52e-5
)

print(f"y+ min = {mesh_yp.point_data['y_plus'].min():.2f}")
print(f"y+ max = {mesh_yp.point_data['y_plus'].max():.2f}")
```

**Physique** : y+ doit être < 1 pour les modèles à couche limite résolue (k-omega SST), ou entre 30 et 300 pour les modèles avec fonctions de paroi (k-epsilon).

### 4.2 Contrainte de paroi (wall shear stress)

```python
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=1.52e-5, 
    wall_normal=[0, 1, 0]
)

tau_w = mesh_wss.cell_data["wall_shear_stress"]
print(f"τ_w max = {tau_w.max():.4f} Pa")

# Validation analytique pour Poiseuille plan
tau_w_analytical = 0.5 * rho * G * H
print(f"τ_w analytique = {tau_w_analytical:.4f} Pa")
```

### 4.3 Taux de déformation (strain rate)

```python
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
print(f"Taux de déformation max = {mesh_sr.point_data['strain_rate'].max():.4f} 1/s")
```

---

## Partie 5 : CHT multi-région

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)

for name, region_mesh in regions.items():
    if "T" in region_mesh.point_data:
        print(f"{name}: T_max = {region_mesh.point_data['T'].max():.1f} K")
```

---

## Partie 6 : Performance

- Temps de lecture pour 1M de cellules : ~1.5s
- Mémoire : ~500 MB pour un cas 1M cellules
- Comparaison avec foamToVTK + VTK reader
- Stratégie de chargement lazy

---

## Conclusion : Du maillage aux résultats en un pipeline

La visualisation et le post-processing ne sont que des étapes. Dans le prochain article, on s'attaque à un cas complet de CHT multi-région où toutes ces techniques se combinent.

**Ressources :**
- Dépôt : [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article en cours d'amélioration — version 1.0*
