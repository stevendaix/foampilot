# Brief — Semaine 5 : "CHT multi-région avec foampilot"

**Objectif** : Montrer comment configurer une simulation de conduction couplée (CHT) multi-région avec foampilot.
**Angle** : Cas d'usage avancé, CHT, Gmsh, interfaces, `chtMultiRegionFoam`.

---

## Structure proposée

### Introduction — Le défi de la CHT

Storytelling : La Conjugate Heat Transfer (CHT) est l'un des cas les plus courants en CFD industrielle. Un dissipateur en cuivre refroidi par de l'air. Le cuivre conduit la chaleur, l'air la convecte. Deux fluides, deux régions, une interface couplée.

**Le problème** : OpenFOAM gère ça via `chtMultiRegionFoam`, mais la configuration est infernale : `constant/` régionalisé, `0/` régionalisé, interfaces, thermophysicalProperties… 

**Promesse** : Avec foampilot, on définit deux régions en Python, et le reste est généré.

### Section 1 — Le workflow manuel (pourquoi c'est dur)

Lister les étapes :
1. Créer `constant/fluid/`, `constant/solid/`
2. Créer `0/fluid/`, `0/solid/`
3. Définir les interfaces dans `constant/regionInterfaces/`
4. Configurer `controlDict` avec `regionSolvers`
5. Éditer les propriétés thermophysiques par région
6. Vérifier la cohérence des champs entre régions

**Message** : Une erreur dans une interface = crash silencieux de `chtMultiRegionFoam`.

### Section 2 — La solution foampilot : définir, pas configurer

Présenter le code :

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion, CoupledInterface

# Définir les régions
fluid = FluidRegion(
    name="air",
    temperature=300,
    turbulence_model="kOmegaSST"
)
solid = SolidRegion(
    name="copper",
    temperature=350,
    thermal_conductivity=400
)

# Définir l'interface
interface = CoupledInterface(
    name="interface_fluid_solid",
    region_1="air",
    region_2="copper"
)

# Configurer le solveur
solver = ChtSolver(
    case_path="./cht_case",
    regions=[fluid, solid],
    interfaces=[interface]
)

solver.setup_case()
solver.run_simulation(nb_proc=4)
```

**Message** : 15 lignes remplacent des heures de configuration manuelle.

### Section 3 — Le maillage Gmsh → CHT

Montrer comment exporter le maillage multi-région :

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("cht_case")

# Créer le volume fluide (air)
air = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gid_air = gmsh.model.addPhysicalGroup(3, [air], name="air")
gmsh.model.setPhysicalName(3, gid_air, "air")

# Créer le volume solide (cuivre)
copper = gmsh.model.occ.addBox(0.4, 0.4, 0, 0.2, 0.2, 0.1)
gmsh.model.occ.synchronize()
gid_cu = gmsh.model.addPhysicalGroup(3, [copper], name="copper")
gmsh.model.setPhysicalName(3, gid_cu, "copper")

# Mailler et exporter
gmsh.model.mesh.generate(3)
exporter = DirectOpenFOAMExporter("./cht_case")
exporter.export_multi_region()
gmsh.finalize()
```

### Section 4 — Ce qui se passe sous le capot

Expliquer les étapes automatiques :
1. Création de `constant/air/`, `constant/copper/`
2. Écriture des `thermophysicalProperties` par région
3. Création des `0/air/T`, `0/copper/T`, etc.
4. Génération des interfaces dans `constant/regionInterfaces/`
5. Injection de `regionSolvers` dans `controlDict`

### Section 5 — Post-processing multi-région

Montrer comment lire les résultats :

```python
from foampilot.postprocess.openfoam_direct import CHTDirectReader

reader = CHTDirectReader(case_path="./cht_case")
regions = reader.read_all_regions(time_step=1000)

# Comparer les températures
for name, mesh in regions.items():
    print(f"{name}: T_max = {mesh.point_data['T'].max():.1f} K")
```

### Conclusion — La CHT accessible

CTA : *"La CHT n'est qu'un exemple. La semaine prochaine, je vous explique comment présenter un tel projet pour qu'il soit adopté par la communauté."*

---

## Code à préparer

- [ ] Script CHT complet (testé)
- [ ] Script Gmsh multi-région
- [ ] Script de post-processing multi-région
- [ ] Extrait de fichier généré (`controlDict` avec regionSolvers)

## Images à préparer

- [ ] Schéma de la géométrie CHT (air + cuivre)
- [ ] Capture d'écran du maillage Gmsh
- [ ] Capture d'écran des répertoires générés (`constant/air/`, `constant/copper/`)
- [ ] Visualisation des températures (PyVista)
- [ ] Capture d'écran du rapport PDF

## Performance

- [ ] Temps de setup_case() vs configuration manuelle
- [ ] Nombre de fichiers générés automatiquement
