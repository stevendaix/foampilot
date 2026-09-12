# Plan d'Implémentation CHT (Conjugate Heat Transfer) pour foampilot

> Date : 2026-08-02
> Objectif : Ajouter un support complet CHT (fluide + solide) à foampilot

---

## Étape 1 : Module `foampilot/cht/` — Structure de base

Fichiers créés / modifiés :
- `cht/__init__.py` — Exports publics (toutes les classes et fonctions)
- `cht/solver.py` — `ChtSolver(BaseSolver)` avec gestion multi-régions
- `cht/regions.py` — `FluidRegion`, `SolidRegion`
- `cht/interfaces.py` — `CoupledInterface`
- `cht/boundary_conditions.py` — Conditions aux limites CHT (classes + fonctions)
- `cht/postprocess.py` — Post-traitement thermique

### Actions Étape 1
- [x] Vérifier que `cht/__init__.py` exporte correctement tous les symboles
- [x] Ajouter des docstrings complètes à chaque classe et méthode
- [x] Ajouter des tests unitaires dans `foampilot/test/test_cht.py`

---

## Étape 2 : Support `regionSolvers` dans `ControlDictFile`

### Actions Étape 2
- [x] Méthode `set_region_solvers()` ajoutée à `controlDictFile.py`
- [x] Vérifier que `_write_attributes` écrit correctement `regionSolvers` comme sous-dictionnaire OpenFOAM
- [x] Tester l'écriture/lecture round-trip de `regionSolvers`
- [x] Ajouter `regionSolvers` au `from_dict()` pour la désérialisation
- [x] Ajouter `region_solvers` au `to_dict()` pour la sérialisation

---

## Étape 3 : Support multi-régions dans `CaseFieldsManager`

### Actions Étape 3
- [x] Modifier `cases_variables.py` pour supporter des champs par région (ex: `0/fluid/T`, `0/heater/T`, `0/metal/T`)
- [x] Ajouter un paramètre `regions` à `CaseFieldsManager.__init__()`
- [x] Générer automatiquement les dossiers `0/<region>/` avec les champs appropriés
- [x] Ajouter le support `is_solid` par région (le solide n'a pas de champ `U`)

---

## Étape 4 : Support `chtMultiRegionFoam` et `chtMultiRegionSimpleFoam`

### Actions Étape 4
- [x] `chtMultiRegionFoam` dans `SOLVER_MODULES` (base_solver.py)
- [x] Ajouter `chtMultiRegionSimpleFoam` à `SOLVER_MODULES`
- [x] `BaseSolver` gère correctement le lancement via `foamRun -solver`
- [x] `ChtSolver.setup_case()` appelle `splitMeshRegions` via region directories

---

## Étape 5 : Conditions aux limites CHT avancantes

### Actions Étape 5
- [x] `coupledTemperature` — classe `CoupledTemperatureBC` + fonction
- [x] `externalTemperature` — classe `ExternalTemperatureBC` + fonction
- [x] `externalWallHeatFluxTemperature` — classe `HeatFluxBC` + fonction
- [x] `fixedValue` pour température fixe — classe `FixedTemperatureBC` + fonction
- [x] `inletOutlet` pour température aux entrées/sorties — classe `InletOutletTemperatureBC` + fonction
- [x] `symmetry` pour les plans de symétrie — classe `SymmetryBC` + fonction
- [x] `totalTemperature` pour les conditions d'entrée compressibles — classe `TotalTemperatureBC` + fonction
- [x] `heatFlux` pour les conditions de flux thermique prescrit — classe `HeatFluxBC` (via externalWallHeatFluxTemperature)
- [x] `radiationCoupledTemperature` pour les interfaces avec rayonnement — classe `RadiationCoupledTemperatureBC` + fonction

---

## Étape 6 : Propriétés thermophysiques par région

### Actions Étape 6
- [x] `SolidRegion.get_thermophysical_properties()` dans `regions.py`
- [x] `FluidRegion.get_transport_properties()` dans `regions.py`
- [x] Support de `thermophysicalProperties` pour les fluides (heRhoThermo, hConst, etc.)
- [x] Support de `transportProperties` pour les fluides (mu, nu, Pr)
- [x] Support de `transportProperties` pour les solides (const, mu=0, nu=0)
- [x] Support configurable des modèles (const, polynomial, etc.)

---

## Étape 7 : Post-traitement CHT

### Actions Étape 7
- [x] `calc_region_heat_flux()` dans `cht/postprocess.py`
- [x] `calc_interface_heat_flux()` dans `cht/postprocess.py`
- [x] `calc_nusselt_number()` dans `cht/postprocess.py`
- [x] `calc_thermal_boundary_layer_thickness()` dans `cht/postprocess.py`
- [x] `calc_heat_transfer_coefficient()` dans `cht/postprocess.py`
- [x] `calc_total_heat_balance()` — vérification conservation de l'énergie
- [x] `calc_temperature_contour()` — isolignes de température
- [x] `calc_thermal_resistance()` — résistance thermique entre régions

---

## Étape 8 : Intégration avec les rapports et présentations

### Actions Étape 8
- [x] `CFDReportGenerator` (report_generator.py)
- [x] `CFDDashboard` (web_presentation.py)
- [x] Sections CHT dans les rapports (via fonctions de post-traitement)
- [x] Export de tableaux de résultats CHT

---

## Étape 9 : Tutoriels CHT complets

### Actions Étape 9
- [x] Tutoriel `examples/cht/heatedDuct/run.py` — cas heatedDuct (chtMultiRegionSimpleFoam)
- [x] Tutoriel `examples/cht/shellAndTube/run.py` — cas shellAndTubeHeatExchanger (chtMultiRegionFoam)
- [x] Tutoriels testés et fonctionnels (génération de fichiers)

---

## Étape 10 : Tests et validation

### Actions Étape 10
- [x] `foampilot/test/test_cht.py` avec 35 tests unitaires couvrant :
  - Import du module CHT
  - Vérification syntaxique des fichiers CHT
  - `FluidRegion` / `SolidRegion` (champs T, U, propriétés)
  - `ChtSolver` (création, validation, setup_case, round-trip)
  - `CoupledInterface` (BCs fluide/solide)
  - Conditions aux limites CHT (8 classes)
  - `ControlDictFile` regionSolvers (to_dict, from_dict, round-trip, write)
  - `CaseFieldsManager` multi-région (backward compat)
  - Fonctions de post-traitement (8 fonctions + cas limites)
- [x] Tests de génération des fichiers de champ pour chaque région
- [x] Tests de génération des propriétés thermophysiques
- [x] Tests des conditions aux limites CHT
- [x] Tests du post-traitement CHT avec des données synthétiques
- [x] Test du round-trip des fichiers `controlDict` avec `regionSolvers`
- [x] Tous les tests existants (test_openfoam14_features.py, test_roundtrip.py) continuent de passer

---

## Résumé de l'avancement

| Étape | Statut |
|-------|--------|
| 1. Module `foampilot/cht/` | ✅ Fait |
| 2. `regionSolvers` dans `ControlDictFile` | ✅ Fait |
| 3. Multi-régions dans `CaseFieldsManager` | ✅ Fait |
| 4. `chtMultiRegionSimpleFoam` dans SOLVER_MODULES | ✅ Fait |
| 5. Conditions aux limites CHT avancées | ✅ Fait |
| 6. Propriétés thermophysiques par région | ✅ Fait |
| 7. Post-traitement CHT | ✅ Fait (11 fonctions) |
| 8. Intégration rapports/présentations | ✅ Fait |
| 9. Tutoriels CHT | ✅ Fait (2 tutoriels) |
| 10. Tests et validation | ✅ Fait (35 tests, tous passants) |
