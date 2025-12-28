
# Explication de l'exemple OpenFOAM Python

Ce document explique en détail un script Python qui configure, exécute et post-traite un cas CFD incompressible en utilisant le framework **FoamPilot**. L'exemple est basé sur le tutoriel OpenFOAM suivant :

[Tutoriel SimpleCar (OpenFOAM)](https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar)

---

## 1. Définition du chemin du cas

Le répertoire du cas est défini à l'aide de `Path` :

```python
current_path = Path.cwd() / "cases"
```

Cela garantit que tous les fichiers et dossiers générés sont organisés dans `cases` dans le répertoire courant.

---

## 2. Propriétés du fluide

Nous définissons les propriétés du fluide avec l'API `FluidMechanics` :

```python
available_fluids = FluidMechanics.get_available_fluids()
fluid = FluidMechanics(
    available_fluids["Air"],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)
nu = fluid.get_fluid_properties()["kinematic_viscosity"]
```

- `get_available_fluids()` fournit la liste des fluides prédéfinis.  
- `FluidMechanics` permet de définir la température et la pression.  
- La viscosité cinématique (`nu`) est extraite pour la configuration du solveur.

---

## 3. Initialisation du solveur

```python
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False
solver.constant.transportProperties.nu = nu
solver.system.write()
solver.constant.write()
```

- Le solveur est configuré en mode incompressible.  
- La gravité est désactivée.  
- La viscosité est définie dans `transportProperties`.  
- Les dossiers `system` et `constant` sont créés et écrits.

---

## 4. Génération du maillage (blockMesh JSON)

```python
data_path = Path.cwd() / "block_mesh.json"
mesh = Meshing(current_path, mesher="blockMesh")
mesh.mesher.load_from_json(data_path)
mesh.mesher.write(file_path=current_path / "system" / "blockMeshDict")
mesh.mesher.run()
```

- L'objet `Meshing` gère la génération du maillage.  
- `load_from_json()` charge la configuration du maillage (blocs, sommets, arêtes) à partir d’un fichier JSON.  
- `write()` crée le fichier `blockMeshDict`.  
- `run()` exécute `blockMesh` pour générer le maillage.  

Cette approche permet de versionner et reproduire le maillage sans modification manuelle des dictionnaires OpenFOAM.

---

## 5. Function Objects

### FieldAverage

```python
name_field, field_average_dict = utilities.Functions.field_average("fieldAverage")
utilities.Functions.write_function_field_average(name_field, field_average_dict, base_path=current_path, folder='system')
```

- Calcule la moyenne d’un champ (par ex. vitesse) sur un patch défini.  
- Ajoute la fonction à `system/controlDict`.

### Reference Pressure

```python
name_field_ref, reference_dict = utilities.Functions.reference_pressure("referencePressure")
utilities.Functions.write_function_reference_pressure(name_field_ref, reference_dict, base_path=current_path, folder='system')
```

- Définit une pression de référence pour les simulations incompressibles.  
- Assure que la pression est unique.

### RunTimeControl

```python
conditions1 = {...}  # critères d'arrêt
name_field_rt1, rt1_dict = utilities.Functions.run_time_control("runTimeControl", conditions=conditions1)
utilities.Functions.write_function_run_time_control(name_field=name_field_rt1, name_condition="runTimeControl1", function_dict=rt1_dict, base_path=current_path, folder='system')
```

- Surveille la convergence ou d'autres critères.  
- Interrompt la simulation lorsque les conditions sont atteintes.

Toutes les fonctions sont finalement écrites dans `system/functions`.

---

## 6. Manipulation des dictionnaires : Patches & topoSet

### createPatchDict

```python
patch_names = ["airIntake"]
patches_dict = utilities.dictonnary.dict_tools.create_patches_dict(patch_names)
create_patch_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='createPatchDict', **patches_dict)
create_patch_dict_file.write("createPatchDict", current_path)
```

- Définit de nouveaux patches à partir de la géométrie existante.  
- Écrit dans `createPatchDict` pour exécution ultérieure.

### topoSetDict

```python
actions = [
    utilities.dictonnary.dict_tools.create_action(...),
    ...
]
actions_dict = utilities.dictonnary.dict_tools.create_actions_dict(actions)
create_topo_set_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='topoSetDict', **actions_dict)
create_topo_set_dict_file.write("topoSetDict", current_path)
solver.system.run_topoSet()
solver.system.run_createPatch()
```

- Crée des `cellSets`, `faceSets` ou `cellZoneSets`.  
- `topoSet` permet de sélectionner des parties du maillage selon des boîtes ou des patches.  
- `createPatch` ajoute des faces aux patches.  
- Automatisation de la préparation du maillage pour des zones spécialisées (ex. zones poreuses).

---

## 7. Conditions aux limites

```python
solver.boundary.initialize_boundary()

solver.boundary.apply_condition_with_wildcard(pattern="inlet", condition_type="velocityInlet", velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)

solver.boundary.apply_condition_with_wildcard(pattern="outlet", condition_type="pressureOutlet")

solver.boundary.apply_condition_with_wildcard(pattern="airIntake", condition_type="velocityInlet", velocity=(Quantity(1.2, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)

solver.boundary.apply_condition_with_wildcard(pattern="walls", condition_type="wall")

solver.boundary.write_boundary_conditions()
```

- Les jokers permettent d’appliquer une condition à plusieurs patches.  
- Gère les entrées de vitesse, sorties de pression et parois.  
- L’écriture génère les fichiers `0/` nécessaires à OpenFOAM.

---

## 8. Exécution de la simulation

```python
solver.run_simulation()
```

- Exécute le solveur (ex. `simpleFoam`) avec tous les réglages et le maillage appliqué.

---

## 9. Post-traitement

```python
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
```

- Traite les résidus pour l’analyse de convergence.  
- Exporte les données au format CSV, JSON, PNG et HTML pour reporting.

---

**Fin de l’exemple**
