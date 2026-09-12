# CFD et physique : calculer Reynolds, viscosité et couche limite avec foampilot

*Avant de lancer une simulation, il faut comprendre les nombres qui la gouvernent. Voici comment foampilot intègre la physique des fluides directement dans le workflow Python.*

---

## Introduction : Pourquoi la physique d'abord ?

En CFD, le code est secondaire. Ce qui compte, c'est la **physique**. Un maillage parfait, un solveur puissant, des schémas numériques sophistiqués — si les conditions physiques sont fausses, le résultat est faux.

Avant de lancer `simpleFoam` ou `pimpleFoam`, l'ingénieur doit répondre à trois questions :

1. **Quel fluide ?** (eau, air, huile, sang, gaz carbonique…)
2. **Quelles propriétés ?** (viscosité, densité, conductivité thermique…)
3. **Quel régime ?** (laminaire, transitoire, turbulent) et à quel **nombre de Reynolds** ?

Traditionnellement, ces calculs se font dans un tableur ou un code séparé, puis les valeurs sont copiées manuellement dans les dictionnaires OpenFOAM. foampilot change la donne en intégrant ces calculs directement dans le workflow Python.

Dans cet article, je vous montre comment :
- Calculer les propriétés de fluides réels (air, eau, huiles) avec **pyfluids**
- Déterminer le régime d'écoulement via le **nombre de Reynolds**
- Valider une simulation contre la solution analytique de **Poiseuille plan**
- Extraire des quantités physiques dérivées : **y+**, **contrainte de paroi**, **taux de déformation**

---

## Partie 1 : Les propriétés des fluides — de l'air à l'eau

### 1.1 Pourquoi la viscosité est le paramètre le plus critique

La viscosité détermine tout en CFD :

- Le **nombre de Reynolds** : `Re = ρvL/μ = vL/ν`
- L'épaisseur de la **couche limite** : `δ ≈ L / √Re`
- Le type d'écoulement : laminaire, transitoire, turbulent
- La taille de la première cellule du maillage (`y+`)

Une erreur sur la viscosité fausse tous les calculs. Si vous utilisez `ν = 1e-5 m²/s` pour de l'eau à 20°C au lieu de `ν ≈ 1e-6 m²/s`, votre simulation apparaîtra 10 fois plus visqueuse qu'elle ne l'est.

### 1.2 Propriétés de fluides courants

Voici les valeurs de référence à 20°C (293.15 K) :

| Fluide | Densité ρ (kg/m³) | Viscosité dynamique μ (Pa·s) | Viscosité cinématique ν (m²/s) |
|--------|-------------------|-----------------------------|-------------------------------|
| Air | 1.204 | 1.81×10⁻⁵ | 1.52×10⁻⁵ |
| Eau | 998.2 | 1.002×10⁻³ | 1.004×10⁻⁶ |
| Huile de moteur | 888 | 0.290 | 3.27×10⁻⁴ |
| Sang (35°C) | 1060 | 3.5×10⁻³ | 3.3×10⁻⁶ |
| CO₂ | 1.842 | 1.48×10⁻⁵ | 8.03×10⁻⁶ |

### 1.3 Calcul automatique avec pyfluids

foampilot intègre **pyfluids** pour calculer ces propriétés automatiquement :

```python
from foampilot import FluidMechanics, ValueWithUnit
from pyfluids import FluidsList

# Définir le fluide et ses conditions
fluid = FluidMechanics(
    fluid_name=FluidsList.Air,
    temperature=ValueWithUnit(293.15, "K"),   # 20°C
    pressure=ValueWithUnit(101325, "Pa")      # Pression atmosphérique
)

# Récupérer toutes les propriétés disponibles
props = fluid.get_fluid_properties()

print(f"Viscosité cinématique : {props['kinematic_viscosity']}")
print(f"Viscosité dynamique : {props['dynamic_viscosity']}")
print(f"Densité : {props['density']}")
print(f"Conductivité thermique : {props['thermal_conductivity']}")
```

**Output :**
```
Viscosité cinématique : 1.520e-05 m²/s
Viscosité dynamique : 1.813e-05 Pa·s
Densité : 1.204 kg/m³
Conductivité thermique : 0.0257 W/(m·K)
```

---

## Partie 2 : Le nombre de Reynolds — le nombre le plus important en CFD

### 2.1 Définition physique

Le nombre de Reynolds est le rapport entre les forces inertielles et les forces visqueuses :

```
Re = (ρ × v × L) / μ = (v × L) / ν
```

Où :
- `ρ` = masse volumique (kg/m³)
- `v` = vitesse caractéristique (m/s)
- `L` = longueur caractéristique (m)
- `μ` = viscosité dynamique (Pa·s)
- `ν` = viscosité cinématique (m²/s)

### 2.2 Interprétation des régimes

| Nombre de Reynolds | Régime | Observation physique |
|-------------------|--------|---------------------|
| Re < 1 | **Stokes** | Écoulement très visqueux (miel, sirop) |
| 1 < Re < 2300 | **Laminaire** | Écoulement fluide, sans turbulence |
| 2300 < Re < 4000 | **Transitoire** | Apparition de turbulences intermittentes |
| Re > 4000 | **Turbulent** | Écoulement chaotique, mélange intense |

### 2.3 Cas concrets et valeurs de référence

| Configuration | Re | Régime |
|---------------|-----|--------|
| Micro-organisme dans l'eau | 1×10⁻⁶ | Stokes |
| Écoulement dans un microcanal (eau) | 100 | Laminaire |
| Écoulement dans une pipe (eau, 1 m/s, D=0.1m) | 100 000 | Turbulent |
| **Voiture à 120 km/h** | **7×10⁶** | **Turbulent** |
| **Avion de ligne (croisière)** | **10⁸–10⁹** | **Turbulent** |
| Écoulement de Poiseuille plan (cas test) | 100–1000 | Laminaire |

### 2.4 Calcul avec foampilot

```python
from foampilot import FluidMechanics, ValueWithUnit
from pyfluids import FluidsList

# Cas : écoulement autour d'une voiture
fluid = FluidMechanics(
    fluid_name=FluidsList.Air,
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa"),
    velocity=ValueWithUnit(30, "m/s"),           # 108 km/h
    characteristic_length=ValueWithUnit(4.5, "m") # Longueur de la voiture
)

Re = fluid.calculate_reynolds()
print(f"Reynolds = {Re:.2e}")
# Output : Reynolds = 9.03e+06
```

**Interprétation** : `Re ≈ 9×10⁶` signifie un écoulement **profondément turbulent**. Le modèle de turbulence `kEpsilon` ou `kOmegaSST` est nécessaire. Un modèle laminaire donnerait des résultats absurdes.

---

## Partie 3 : Le cas test de Poiseuille — validation physique

L'écoulement de Poiseuille plan est le **test de validation fondamental** en CFD. C'est un écoulement laminaire entre deux plaques parallèles, avec une pression imposée qui génère un profil de vitesse parabolique.

### 3.1 La solution analytique

Pour un écoulement de Poiseuille plan incompressible, la solution analytique de la vitesse est :

```
u(y) = (1 / 2ν) × (dp/dx) × (h² - y²)
```

Où :
- `dp/dx` = gradient de pression imposé (Pa/m)
- `h` = demi-hauteur du canal (m)
- `ν` = viscosité cinématique (m²/s)
- `y` = distance depuis la ligne médiane (m)

### 3.2 Contrainte de paroi analytique

La contrainte de paroi `τ_w` pour Poiseuille plan s'obtient par :

```
τ_w = (h/2) × |dp/dx|
```

Ou encore :
```
τ_w = ρ × ν × (du/dy)|_wall
```

### 3.3 Implémentation dans foampilot

```python
# Paramètres du cas test Poiseuille (voir test_cfd_methods.py)
nu = 0.1          # Viscosité cinématique (m²/s)
rho = 1.0         # Densité (kg/m³)
G = 5.0           # Gradient de pression (Pa/m)
H = 1.0           # Demi-hauteur du canal (m)

# Vitesse maximale au centre — solution analytique : u_max = G·h² / 2ν
U_max = G * H**2 / (2 * nu)
print(f"U_max = {U_max:.1f} m/s")  # Output : U_max = 25.0 m/s

# Vitesse moyenne (U_avg = U_max / 2 pour Poiseuille)
U_avg = U_max / 2

# Nombre de Reynolds
Re = U_avg * (2 * H) / nu
print(f"Re = {Re:.1f}")  # Output : Re = 250.0

# Vérification du régime
if Re < 2300:
    print("Régime laminaire — solution analytique valide")
else:
    print("Régime turbulent — il faut un modèle de turbulence")
```

### 3.4 Validation quantitative

Dans `test_cfd_methods.py`, foampilot valide ses calculs post-traitement contre la solution analytique. La méthode `get_structure` lit les fichiers VTK générés par `foamToVTK` — un `foamToVTK -case ./planarPoiseuille` est requis avant de lire les résultats :

```python
import numpy as np
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./planarPoiseuille")
mesh = fp.get_structure(time_step=25)["cell"]

# Contrainte de paroi analytique
G = 5.0      # Pa/m (gradient de pression)
H = 1.0      # m (demi-hauteur)
rho = 1.0    # kg/m³
tau_w_analytical = 0.5 * rho * G * H
print(f"τ_w analytique = {tau_w_analytical:.4f} Pa")

# Contrainte de paroi calculée par foampilot
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=0.1,  # viscosité utilisée dans le cas test
    wall_normal=[0, 1, 0]
)

wss_max = np.max(mesh_wss.point_data["wall_shear_stress"])
error = abs(wss_max - tau_w_analytical) / tau_w_analytical
print(f"Erreur relative : {error:.2%}")
# Output attendu : Erreur relative < 5%
```

**Résultat attendu** : une erreur inférieure à 5% pour un maillage suffisamment raffiné (≥ 50 cellules dans la direction normale).

---

## Partie 4 : La couche limite et y+

### 4.1 Qu'est-ce que la couche limite ?

Lorsqu'un fluide s'écoule le long d'une paroi, la vitesse passe de 0 à la vitesse libre sur une courte distance : la **couche limite visqueuse**. C'est dans cette région que se forment les tourbillons et que la turbulence naît.

### 4.2 Le paramètre y+

En turbulence, la première cellule du maillage doit être placée dans la couche limite. Le paramètre clé est **y+** :

```
y+ = (u_τ × y) / ν
```

Où `u_τ = √(τ_w / ρ)` est la vitesse de frottement.

### 4.3 Règles de maillage selon le modèle de turbulence

| Modèle de turbulence | y+ cible | Position première cellule |
|---------------------|----------|---------------------------|
| k-epsilon standard | 30–300 | Loi de paroi intégrale |
| k-omega SST | y+ < 1 | Résolu dans la couche limite |
| Spalart-Allmaras | 1–20 | Partiellement résolu |

### 4.4 Calcul automatique avec foampilot

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

fp = FoamPostProcessing(case_path="./planarPoiseuille")
mesh = fp.get_structure(time_step=25)["cell"]

# Calcul de y+ sur le patch "walls"
mesh_yp = fp.calc_y_plus(
    mesh, 
    wall_patch_name="walls", 
    velocity_field="U", 
    viscosity=0.1  # viscosité du cas test
)

print(f"y+ min = {mesh_yp.point_data['y_plus'].min():.2f}")
print(f"y+ max = {mesh_yp.point_data['y_plus'].max():.2f}")
print(f"y+ mean = {mesh_yp.point_data['y_plus'].mean():.2f}")
```

---

## Partie 5 : Quantités physiques dérivées

### 5.1 Contrainte de paroi (wall shear stress)

La contrainte de paroi est la force tangentielle exercée par le fluide sur la paroi :

```
τ_w = μ × (du/dy)|_wall
```

```python
# Calcul de la contrainte de paroi
mesh_wss = fp.calc_wall_shear_stress(
    mesh, 
    velocity_field="U", 
    viscosity=0.1, 
    wall_normal=[0, 1, 0]
)

tau_w = mesh_wss.point_data["wall_shear_stress"]
print(f"τ_w max = {tau_w.max():.4f} Pa")
print(f"τ_w mean = {tau_w.mean():.4f} Pa")

# Validation analytique pour Poiseuille plan
tau_w_analytical = 0.5 * rho * G * H
print(f"τ_w analytique = {tau_w_analytical:.4f} Pa")
```

### 5.2 Taux de déformation (strain rate)

Le taux de déformation mesure comment le fluide se déforme localement :

```
γ̇ = √(2 × S : S)
```

Où `S` est le tenseur de déformation : `S = 0.5 × (∇u + ∇uᵀ)`

```python
# Calcul du taux de déformation
mesh_sr = fp.calc_strain_rate(mesh, velocity_field="U")
strain_rate = mesh_sr.point_data["strain_rate"]
print(f"Taux de déformation max = {strain_rate.max():.4f} 1/s")
print(f"Taux de déformation moyen = {strain_rate.mean():.4f} 1/s")
```

### 5.3 Profil de vitesse analytique vs numérique

```python
import numpy as np

# Paramètres du cas test (identiques à Partie 3)
nu = 0.1      # Viscosité cinématique (m²/s)
G = 5.0       # Gradient de pression (Pa/m)
H = 1.0       # Demi-hauteur du canal (m)

# Extraire le profil de vitesse sur la ligne médiane
centreline = mesh.slice(normal="y", origin=(0, 0, 0))
y_points = centreline.points[:, 2]  # Coordonnée Z

# Solution analytique de Poiseuille
U_analytical = (1 / (2 * nu)) * (-G) * (H**2 - y_points**2)

# Comparaison
U_numerical = centreline.point_data["U"][:, 0]
error = np.mean(np.abs(U_numerical - U_analytical))
print(f"Erreur moyenne sur le profil : {error:.4f} m/s")
```

---

## Partie 6 : Du fluide à la simulation — exemple complet

Voici comment intégrer ces calculs physiques dans un workflow foampilot complet :

```python
from foampilot import Solver, Meshing, FluidMechanics, ValueWithUnit
from pyfluids import FluidsList

# 1. Choisir le fluide et calculer ses propriétés
fluid = FluidMechanics(
    fluid_name=FluidsList.Water,
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa"),
    velocity=ValueWithUnit(2.0, "m/s"),           # Vitesse d'entrée
    characteristic_length=ValueWithUnit(0.05, "m")  # Diamètre de tuyau
)
props = fluid.get_fluid_properties()

# 2. Nombre de Reynolds
Re = fluid.calculate_reynolds()
print(f"Re = {Re:.0f}")  # Ex: Re = 99 800 (turbulent)

# 3. Configuration du solveur
solver = Solver(case_path="./pipe_flow")
solver.transient = False
solver.turbulence_model = "kOmegaSST"  # Choisi selon Re

# 4. Injection des propriétés physiques
solver.constant.transportProperties.nu = props['kinematic_viscosity']

# 5. Maillage et conditions aux limites
meshing = Meshing(case_path="./pipe_flow", mesher="blockMesh")
meshing.write()

solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    "inlet", "velocityInlet",
    velocity=(ValueWithUnit(2.0, "m/s"),
              ValueWithUnit(0, "m/s"),
              ValueWithUnit(0, "m/s"))
)
solver.boundary.write_boundary_conditions()

# 6. Écriture et lancement
solver.write_case()
solver.run_simulation(nb_proc=2)
```

---

## Conclusion : La physique comme fondation

En CFD, le code est un outil. La physique est le fondement. foampilot ne se contente pas de générer des fichiers — il intègre les **calculs physiques** (propriétés de fluides, Reynolds, y+, contrainte de paroi) directement dans le workflow Python.

Dans le prochain article, je vous montre comment automatiser la création complète d'un cas OpenFOAM avec ces concepts physiques intégrés.

**Ressources :**
- Cas test Poiseuille : [github.com/stevendaix/foampilot/tree/main/planarPoiseuille](https://github.com/stevendaix/foampilot/tree/main/planarPoiseuille)
- Documentation FluidMechanics : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)
- pyfluids : [pypi.org/project/pyfluids](https://pypi.org/project/pyfluids)

---

*Article en cours d'amélioration — version 1.1*
