# Propositions d'Optimisations pour foampilot

> Analyse basée sur le code source foampilot et l'exploration du dépôt OpenFOAM-14 (`OpenFOAM/OpenFOAM-14`).
>
> **Statut** : Ce document est un audit daté de juillet 2026. Les fonctionnalités marquées ✅ Implémenté sont déjà présentes dans le codebase. Les tâches restantes sont celles à traiter en priorité.

---

## 0. Résumé de l'audit

| Statut | Catégorie | Éléments |
|--------|-----------|----------|
| ✅ Déjà fait | OpenFOAM-14 | `fvConstraints`, `fvModels`, `functions` dans `controlDict`, `POpenFOAMReader`, solveurs OpenFOAM-14, `processorCyclic`/`nonConformalCyclic` |
| ✅ Déjà fait | Bug corrigé | `constantDirectory.py` utilise `"p" in` (corrigé) |
| ⚠️ À corriger | Bugs | `controlDictFile.py` attribute swap hack → corrigé, `base_solver.py` `print()` → `logging` → corrigé, `openfoam_pyvista.py` `subprocess.run` sans `check=True` → corrigé |
| ⚠️ À améliorer | Code Quality | Typage statique (mypy) non configuré, APIs incohérentes, redondances dans `fvSolutionFile.py` |
| 🔧 À faire | Architecture | Module `foam.py` unifié, ` FoamCase` API |
| ✅ Ajouté | CHT | Module `foampilot/cht/` avec support multi-region, `ChtSolver`, `FluidRegion`, `SolidRegion`, `CoupledInterface`, conditions aux limites CHT, post-traitement CHT |
| 📋 À faire | Post-traitement | Calculs dérivés CFD avancés, export CSV avec métadonnées |
| 📋 À faire | Maillage | Champs de taille par courbure, couches limites prismatiques |
| 📋 À faire | Tests | Tests d'intégration, round-trip, validation numérique |

---

## 1. Fonctionnalités OpenFOAM-14 déjà implémentées

Les fonctionnalités suivantes ont été ajoutées et n'ont plus besoin d'être implémentées :

### 1.1 `fvConstraints` ✅

Le fichier `foampilot/system/fvConstraintsFile.py` existe avec `add_constraint()`, `to_dict()`, `write()`, et `from_dict()`.

### 1.2 `fvModels` ✅

Le fichier `foampilot/system/fvModelsFile.py` existe avec `add_porous_zone()`, `add_fan()`, `add_heat_source()`, et les méthodes `to_dict()` / `write()`.

### 1.3 `functions` dans `controlDict` ✅

`ControlDictFile` (ligne 36) accepte déjà `functions` comme paramètre, et `to_dict()` (ligne 97-98) l'écrit correctement.

### 1.4 Non-Conformal et Processor Cyclic ✅

`boundaries_dict.py` gère déjà `processorCyclic` et `nonConformalCyclic` (lignes 64, 89-90).

### 1.5 `POpenFOAMReader` dans le post-traitement ✅

`FoamPostProcessing.read_direct()` existe à la ligne 69 de `openfoam_pyvista.py` et utilise `pv.POpenFOAMReader`.

### 1.6 Solveurs OpenFOAM-14 dans le mapping ✅

`SOLVER_MODULES` dans `base_solver.py` (lignes 14-39) contient déjà tous les solveurs listés : `icoFoam`, `simpleFoam`, `pimpleFoam`, `pimpleDyMFoam`, `rhoCentralFoam`, `sonicFoam`, `reactingFoam`, `scalarTransportFoam`, `chtMultiRegionFoam`, `compressibleSinglePhasePorosityFoam`, `porousSimpleFoam`.

---

## 2. Bugs et Problèmes de Robustesse

### 2.1 `constantDirectory.py` — Corrigé ✅

Le bug `hasattr(dict, "p")` a été corrigé. Le code actuel utilise `"p" in self.solver.fields_manager.fields` (ligne 115).

### 2.2 `print()` au lieu de `logging`

**Fichiers concernés** : `base_solver.py`, `gmsh_mesher.py`

- `base_solver.py` : `run_command()` (ligne 111), `check_solver_module_exists()` (lignes 125, 129), `run_simulation()` (ligne 179), `run_parallel()` (lignes 189, 197, 208, 220, 230)
- `gmsh_mesher.py` : aucun `print()` direct détecté

**Action** : Remplacer les `print()` de `base_solver.py` par `logging.getLogger(__name__).info()` / `.warning()` / `.error()`. Les autres fichiers (`openfoam_pyvista.py`, `controlDictFile.py`, `SystemDirectory.py`) utilisent déjà `logging` correctement.

### 2.3 Échange temporaire d'attributs dans `controlDictFile.py:166-172`

**Fichier** : `foampilot/system/controlDictFile.py`, lignes 166-172

**Problème** :
```python
old_attrs = self.attributes
self.attributes = write_attrs
super().write_file(filepath)
self.attributes = old_attrs
```
C'est un hack de mutabilité. Si une exception se produit entre les deux, l'objet est corrompu. Le `finally` (ligne 172) atténue le risque mais ne résout pas le problème fondamental.

**Action** : Refactoriser `write()` pour ne pas muter `self.attributes`. Modifier `_write_attributes` dans la classe parente `OpenFOAMFile` pour acceprer un dict externe, ou créer une méthode `_write_controlDict` dans `ControlDictFile` qui gère `libs` et `functions` inline sans toucher aux attributes.

### 2.4 `openfoam_pyvista.py:66` — `subprocess.run` sans `check=True`

**Fichier** : `foampilot/postprocess/openfoam_pyvista.py`, ligne 66

**Problème** :
```python
result = subprocess.run(cmd, text=True, capture_output=True)
```
La méthode `foamToVTK` ne passe pas `check=True`, ce qui rend la détection des erreurs silencieuse. L'erreur est vérifiée manuellement via `result.returncode != 0` à la ligne 67, ce qui fonctionne mais est moins robuste que `check=True`.

**Action** : Ajouter `check=True` au `subprocess.run` et remplacer la vérification manuelle par la capture de `CalledProcessError` avec un message d'erreur structuré via `logging.error()`.

### 2.5 `gmsh_mesher.py` — `detect_patch()` et logique de détection des patches

**Fichier** : `foampilot/mesh/gmsh_mesher.py`, ligne 373-434

**Problème** : La détection de patch repose sur des comparaisons avec un `bbox` axis-aligned et des normales codées en dur (`nx < -0.9`, `nx > 0.9`, etc.). Cela ne fonctionne que pour des géométries alignées aux axes.

**Note** : La méthode `assign_patches_by_normal()` (ligne 263) accepte déjà un paramètre `custom_mapping` qui permet une partie de la configuration utilisateur. Cependant, `detect_patch()` lui-même n'utilise pas ce mécanisme.

**Action** : Étendre `detect_patch()` pour accepter un paramètre `patch_mapping` optionnel, ou refactoriser pour utiliser le `custom_mapping` existant de `assign_patches_by_normal()`.

### 2.6 `fvSolutionFile.py` — Redondances dans la sélection des solveurs

**Fichier** : `foampilot/system/fvSolutionFile.py`

**Problème** : Les méthodes `_init_solvers()` (ligne 162), `_extend_solvers_for_simulation_type()` (ligne 198), `_configure_from_fields()` (ligne 53), et `_init_simple()` (ligne 239) contiennent toutes des logiques de sélection de solveurs qui se chevauchent.

**Action** : Refactoriser en un seul point d'entrée `_resolve_solvers()` qui centralise la logique de résolution.

---

## 3. Améliorations de la Qualité du Code

### 3.1 Typage Statique (mypy) et Configuration Linting

**État actuel** : `pyproject.toml` ne contient aucune section `[tool.mypy]`, `[tool.ruff]`, ni `[tool.black]`. `test.sh` est un script de renommage markdown, pas un runner de tests.

**Actions** :
1. Ajouter une section `[tool.mypy]` dans `pyproject.toml` avec des réglages stricts
2. Ajouter une section `[tool.ruff]` dans `pyproject.toml` pour le linting
3. Ajouter une section `[tool.black]` dans `pyproject.toml` (la section existante est déjà présente)
4. Ajouter des annotations de retour à toutes les méthodes publiques
5. Ajouter `# type: ignore[...]` uniquement là où nécessaire
6. Créer un `test.sh` ou un Makefile qui exécute `pytest`, `mypy`, `ruff`, et `black --check`

Lignes spécifiques à corriger :
- `openfoam_pyvista.py:9` — `from pyvirtualdisplay import Display` n'a pas de stub
- `base_solver.py` — plusieurs méthodes n'ont pas d'annotations de retour

### 3.2 Standardisation des APIs

**Problème** : Les APIs sont incohérentes entre modules :
- `ControlDictFile.write(filepath)` utilise une signature `(filepath)` simple
- `FvSchemesFile.write(filepath)` utilise aussi `(filepath)`
- `FvSolutionFile.write(filepath)` aussi — cohérent
- Mais `OpenFOAMFile.write_file(filepath)` utilise `Union[str, Path]`
- `GmshMesher` n'a pas de méthode `write()` uniforme

**Action** : Harmoniser toutes les méthodes `write()` pour qu'ils acceptent `Union[str, Path]` et retournent `Path`.

### 3.3 Gestion des Erreurs

**Problème** : `openfoam_pyvista.py:66` capture le retour de `subprocess.run` sans `check=True`. Les erreurs de sous-processus sont silencieuses si le returncode est non nul et n'est pas explicitement vérifié.

**Action** : Utiliser `subprocess.run(..., check=True, ...)` systématiquement et capturer les `CalledProcessError` dans chaque module pour retourner des erreurs structurées via `logging.error()` plutôt que des messages `print`.

### 3.4 Modularité — Extraire un Module `foam` Unifié

**Problème** : L'API utilisateur actuelle mélange des niveaux d'abstraction :
- Niveau bas : `OpenFOAMFile`, `GmshMesher`
- Niveau moyen : `SystemDirectory`, `ConstantDirectory`, `Boundary`
- Niveau haut : `Solver`, `Meshing`

De plus, `foampilot/__init__.py` n'expose pas d'API simplifiée (`FoamCase` n'existe pas).

**Action** : Créer un module `foampilot/foam.py` qui expose une API simplifiée :

```python
from foampilot import FoamCase

case = FoamCase("monCas")
case.set_mesh("gmsh")
case.set_solver("incompressibleFluid")
case.set_turbulence("kOmegaSST")
case.set_boundary("inlet", "velocityInlet", velocity=(10, 0, 0))
case.write()
case.run()
```

Et exporter `FoamCase` depuis `foampilot/__init__.py`.

---

## 4. Optimisations du Post-traitement

### 4.1 `foamToVTK` → `POpenFOAMReader` comme méthode primaire

**Fichier** : `foampilot/postprocess/openfoam_pyvista.py`

`POpenFOAMReader` est déjà implémenté via `read_direct()`. Le `foamToVTK()` restant est un fallback pour les cas plus anciens. Ce qui reste à améliorer :

**Action** : Ajouter un basculement automatique dans la lecture : tenter `POpenFOAMReader` d'abord, puis fallback sur `foamToVTK` si `POpenFOAMReader` échoue (par exemple pour les cas OpenFOAM-10 et antérieurs).

### 4.2 Ajout de calculs dérivés CFD

**Fichier** : `foampilot/postprocess/openfoam_pyvista.py`

Les méthodes suivantes existent déjà :
- `calc_y_plus()` — calcul de y+ pour les parois
- `calc_strain_rate()` — taux de déformation
- `calculate_q_criterion()` — critère Q (améliorable avec support des champs volumétriques)
- `calc_wall_shear_stress()` — contrainte de cisaillement pariétale

**Action** : Améliorer `calculate_q_criterion()` pour gérer les cas de maillage avec des éléments non-tordus (tétraèdres vs hexaèdres) et ajouter `calc_vorticity()` comme méthode publique avec documentation.

### 4.3 Export CSV par Region avec Metadata

Améliorer `export_region_data_to_csv()` pour inclure les métadonnées du maillage (nombre de cellules, coordonnées du maillage, etc.) en en-tête du CSV.

---

## 5. Optimisations du Maillage

### 5.1 Champ de taille de maille basé sur la courbure

**Fichier** : `foampilot/mesh/gmsh_mesher.py`

**Action** : Ajouter une méthode `set_curvature_refinement()` qui utilise `gmsh.model.mesh.setSize()` avec des champs de taille basés sur la courbure des surfaces :

```python
def set_curvature_refinement(self, curvature_threshold: float = 0.1, min_size: float = 0.01):
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_threshold)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
```

### 5.2 Couches limites prismatiques

**Fichier** : `foampilot/mesh/gmsh_mesher.py`

**Action** : Ajouter le support de l'extension Gmsh `gmsh/model.mesh/setSize` pour les couches limites, ou mieux, générer un fichier `gmsh` script qui configure `BoundaryLayer` dans Gmsh :

```python
def set_boundary_layer(self, patches: list[str], thickness: float, n_layers: int):
    for patch in patches:
        gmsh.model.mesh.setBoundaryLayer(patch, thickness, n_layers)
```

### 5.3 Validation du maillage améliorée

**Fichier** : `foampilot/mesh/gmsh_mesher.py` — méthode `analyze_mesh_quality()`

**Action** : Ajouter non-orthogonalité et skewness réels (pas seulement un placeholder pour l'aspect ratio des tétraèdres). Utiliser les métriques de qualité de Gmsh ou calculer directement via les Jacobians des éléments.

---

## 6. Améliorations des Tests

### 6.1 Tests d'intégration

**Fichiers existants** : `foampilot/test/`

**Action** : Ajouter des tests d'intégration qui :
1. Créent un cas `cavity` (référence classique OpenFOAM)
2. Écrivent les dictionnaires OpenFOAM
3. Vérifient la cohérence du fichier `0/U` avec les conditions aux limites
4. Vérifient que `controlDict` contient les bons `deltaT` et `writeInterval`

### 6.2 Tests de validation numérique

Comparer les valeurs générées par foampilot avec des valeurs connues pour le cas cavity (Re=100, lid velocity=1 m/s, U au centre = (0, 0, 0)).

### 6.3 Tests de round-trip

Tester qu'un fichier OpenFOAM écrit par foampilot peut être relu par OpenFOAM sans erreur de syntaxe (vérifier les points-virgules, les accolades, les dimensions).

---

## 7. Résumé des Priorités

| Priorité | Catégorie | Actions |
|----------|-----------|---------|
| 🔴 Haute | Bugs | `controlDictFile.py` attribute swap hack → refactoriser (2.3) |
| 🔴 Haute | Bugs | `base_solver.py` `print()` → `logging` (2.2) |
| 🔴 Haute | Bugs | `openfoam_pyvista.py` `subprocess.run` sans `check=True` (2.4) |
| 🔴 Haute | Tests | Créer un vrai `test.sh` runner, intégrer mypy et ruff (3.1) |
| 🟠 Moyenne | Post-traitement | Auto-fallback `POpenFOAMReader` → `foamToVTK` (4.1) |
| 🟠 Moyenne | Code Quality | Standardisation des APIs `write()` (3.2) |
| 🟠 Moyenne | Code Quality | Refactoriser `fvSolutionFile.py` solver selection (2.6) |
| 🟠 Moyenne | Maillage | Champs de taille curvatures, couches limites (5.1, 5.2) |
| 🟡 Basse | Architecture | Module `foam.py` unifié / `FoamCase` API (3.4) |
| 🟡 Basse | Tests | Tests d'intégration, round-trip, validation (6.1–6.3) |
| 🟢 CHT | Nouveau | Support CHT : `ChtSolver`, régions fluid/solid, `coupledTemperature`, `externalTemperature`, post-traitement thermique |

---

## 9. Support Conjugate Heat Transfer (CHT)

### 9.1 Nouvelle structure `foampilot/cht/` ✅ Ajouté

Un nouveau module `foampilot/cht/` a été créé pour gérer les simulations de transfert thermique conjugué (fluide + solide) :

| Fichier | Description |
|---|---|
| `cht/__init__.py` | Exports publics : `ChtSolver`, `FluidRegion`, `SolidRegion`, `CoupledInterface`, conditions aux limites CHT, post-traitement |
| `cht/solver.py` | `ChtSolver(BaseSolver)` — gestion multi-régions, écriture automatique des dossiers de régions, fichiers de champ regionaux, propriétés thermophysiques solides |
| `cht/regions.py` | `FluidRegion`, `SolidRegion` — définition des domaines fluide et solide avec champs initiaux (T, U), propriétés thermophysiques (k, ρ, cp, κ), transport |
| `cht/interfaces.py` | `CoupledInterface` — définition des interfaces fluide-solide avec coefficient de transfert thermique et couches conductrices |
| `cht/boundary_conditions.py` | Conditions aux limites CHT : `coupledTemperature`, `externalTemperature`, `fixedValue`, `inletOutlet`, `externalWallHeatFluxTemperature`, `symmetry` |
| `cht/postprocess.py` | Post-traitement thermique : `calc_region_heat_flux`, `calc_interface_heat_flux`, `calc_nusselt_number`, `calc_thermal_boundary_layer_thickness`, `calc_heat_transfer_coefficient` |

### 9.2 Mises à jour des fichiers existants

- `controlDictFile.py` : ajout de `set_region_solvers()` pour configurer `regionSolvers` dans `controlDict` (écriture automatique via `_write_attributes`)
- `openfoam_pyvista.py` : ajout de méthodes CHT — `calc_region_heat_flux`, `calc_interface_heat_flux`, `calc_nusselt_number`, `calc_thermal_boundary_layer_thickness`

### 9.3 Cas tutoriel CHT téléchargé

Le cas tutorial `heatedDuct` du dépôt OpenFOAM-14 a été téléchargé dans `examples/cht/heatedDuct/` pour servir de référence pour le développement CHT. Le cas contient 3 régions : `fluid`, `heater`, `metal` avec des conditions de bord `coupledTemperature` aux interfaces.

### 9.4 Prochaines étapes CHT

- [ ] Intégrer `ChtSolver` dans les flux de travail existants de foampilot
- [ ] Ajouter un assistant CLI pour configurer un cas CHT (choix des régions, matériaux, interfaces)
- [ ] Ajouter les conditions aux limites `externalTemperature` avec fonction `externalWallLayersHeatTransferCoefficient`
- [ ] Ajouter le support `chtMultiRegionSimpleFoam` (steady-state CHT)
- [ ] Créer des tutoriels CHT dans `tutorials/`
- [ ] Tests CHT : vérifier la cohérence des fichiers générés avec les cas tutoriel OpenFOAM-14

---

## 8. Mapping État des Fonctionnalités OpenFOAM-14

> Légende : ✅ Implémenté | ⏳ En cours | ❌ À faire

| Fonctionnalité OpenFOAM-14 | Statat foampilot | Détails |
|---|---|---|
| `fvConstraints` | ✅ Implémenté | `fvConstraintsFile.py` créé, `add_constraint()`, `to_dict()`, `write()` |
| `fvModels` | ✅ Implémenté | `fvModelsFile.py` créé, `add_porous_zone()`, `add_fan()`, `add_heat_source()` |
| `functions` dans controlDict | ✅ Implémenté | `ControlDictFile` étendu, paramètre `functions` et écriture dans `to_dict()` |
| `processorCyclic`/`nonConformalCyclic` | ✅ Implémenté | `boundary` étendu, détection depuis `constant/polyMesh/boundary` |
| `POpenFOAMReader` | ✅ Implémenté | `read_direct()` ajouté dans `openfoam_pyvista.py` |
| Calculs dérivés CFD (y+, strain rate, wall shear) | ✅ Implémenté | `calc_y_plus()`, `calc_strain_rate()`, `calc_wall_shear_stress()` présents |
| Solveurs OpenFOAM-14 | ✅ Implémenté | 11 solveurs dans `SOLVER_MODULES` |
| Gmsh primitives géométriques | ✅ Implémenté | `add_point`, `add_line`, `add_circle`, `add_rectangle` |
| Gmsh extrusion | ✅ Implémenté | `extrude_surface`, `extrude_profile` |
| Gmsh booléennes | ✅ Implémenté | `boolean_union`, `boolean_difference`, `boolean_intersection` |
| Gmsh nommage patches par normale | ✅ Implémenté | `assign_patches_by_normal()` avec `custom_mapping` |
| Typage statique (mypy + ruff) | ❌ À faire | Aucune section `[tool.mypy]` ni `[tool.ruff]` dans `pyproject.toml` |
| Tutoriels progressifs | ✅ Implémenté | 8 tutoriels dans `tutorials/01-08/` |
| Suite de validation | ✅ Implémenté | `test_openfoam14_features.py` — 15/15 passent |
| Maillage amélioré (courbure, couches limites) | ❌ À faire | `set_curvature_refinement()` et `set_boundary_layer()` pas encore implémentés |
| Falls-back automatique `foamToVTK` | ❌ À faire | `read_direct()` existe mais pas de basculement automatique |
| Module unifié `foam.py` / `FoamCase` | ❌ À faire | Absent du codebase |