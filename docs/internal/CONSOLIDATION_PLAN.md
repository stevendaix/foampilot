# Plan de consolidation et d'intégration des PRs foampilot

## Contexte

8 PRs ouvertes sur `stevendaix/foampilot`. Aucune ne peut être mergée en l'état :
- Plusieurs PRs cassent les mêmes fichiers coeur (`voxcity_reader.py`, `report_generator.py`, `typst_pdf.py`, `simulation_report.py`, `mesh_report.py`)
- PR #12 a un conflit de merge avec `main`
- Chemins absolus en dur, binaires commités, régressions fonctionnelles

## Stratégie

1. Créer une **PR de consolidation** qui fixe les fichiers cassés communs
2. Rebaser toutes les PRs sur cette base consolidée
3. Nettoyer chaque PR individuellement
4. Merger dans l'ordre de dépendance

---

## Phase 1 : PR de consolidation (`fix/consolidate-core-modules`)

**Objectif** : Réparer les 5 fichiers coeur cassés par plusieurs PRs.

### Fichiers à réparer

#### A. `foampilot/src/foampilot/urban/readers/voxcity_reader.py`
- Restaurer `import shapely.ops`
- Restaurer `footprint = projected` après projection
- Restaurer la détection CRS auto (ne pas forcer EPSG:4326→32631)

#### B. `foampilot/src/foampilot/report/report_generator.py`
- Restaurer `import html`
- Restaurer `html.escape()` sur tous les contenus utilisateurs
- Restaurer `self._plotly_figures` et `add_plotly_figure()`
- Restaurer `from plotly.offline import get_plotlyjs` (inline Plotly JS)

#### C. `foampilot/src/foampilot/report/typst_pdf.py`
- Restaurer `shutil.which("typst")` avec fallback
- Restaurer le retour de `compile_pdf()` (retourner le chemin PDF)
- Déduire `.typ` path depuis `output_pdf` parent

#### D. `foampilot/src/foampilot/report/simulation_report.py`
- Restaurer `self.solver_settings = self._extract_solver_settings()`
- Restaurer `self.bc_summary = self._extract_bc_summary()`

#### E. `foampilot/src/foampilot/report/mesh_report.py`
- Restaurer le garde `num_patches > 0` avant division

### Vérification Phase 1
- [ ] `python3 -m py_compile` sur tous les fichiers modifiés
- [ ] `PYTHONPATH=src python3 -m pytest test/ -v` passe
- [ ] Aucun conflit avec `main`
- [ ] Revue de code par agent

---

## Phase 2 : Rebaser les PRs sur la consolidation

**Objectif** : Toutes les PRs doivent être basées sur `fix/consolidate-core-modules` au lieu de `main`.

### PRs à rebaser
- PR #10 `feat/medical-build-pipeline`
- PR #12 `feature/makehuman-jos3-openfoam-coupling`
- PR #14 `feat/vof-to-dpm-converter`
- PR #15 `feat/yade-openfoam13-coupling`
- PR #17 `feat/openfoam13-tutorial-integration`
- PR #18 `feat/moose-openfoam13-precice`
- PR #19 `feat/openfoam13-wolfdynamics-docs`

### Méthode
```bash
# Pour chaque PR
git checkout <branche>
git rebase fix/consolidate-core-modules
# Résoudre les conflits (seulement les fichiers coeur déjà consolidés)
git push --force-with-lease
```

### Vérification Phase 2
- [ ] Toutes les PRs ont un statut `MERGEABLE` (pas `CONFLICTING`)
- [ ] Les fichiers coeur (`voxcity_reader.py`, `report_generator.py`, etc.) sont identiques dans toutes les PRs
- [ ] Revue de code par agent

---

## Phase 3 : Nettoyer PR #20 — Marine OpenFOAM Workflows

**Objectif** : Petite PR bien intégrée, peu de corrections nécessaires.

### Corrections
1. Fixer l'indentation FoamFile dans `marine.py` (`joints` block)
2. Supprimer imports inutilisés (`Iterable`, `Sequence`) dans `openfoam.py`
3. Aligner `Meshing.add_file()` avec `SystemDirectory.add_dict_file()`

### Vérification Phase 3
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Tests passent (`PYTHONPATH=src python3 -m pytest test/test_openfoam_workflows.py -v`)
- [ ] Revue de code par agent

---

## Phase 4 : Nettoyer PR #19 — Wolf Dynamics Docs

**Objectif** : Documentation et exemples Wolf Dynamics.

### Corrections
1. Fixer `examples/openfoam13_tutorials/01_laminar_channel/run.py` (fonction `run_foampilot_case` manquante)
2. Supprimer imports inutilisés dans `openfoam13.py`
3. Documenter la configuration bashrc via variable d'environnement

### Vérification Phase 4
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Revue de code par agent

---

## Phase 5 : Nettoyer PR #14 — VOF-to-DPM

**Objectif** : Convertisseur VOF vers DPM.

### Corrections
1. Externaliser le projet GEE vers variable d'environnement
2. Déplacer `vof_to_dpm_technical_note.pdf` vers Git LFS ou `.gitignore`
3. Harmoniser `compressibleVoFClouds.C` avec `incompressibleVoFClouds.C`
4. Consolider les tests dupliqués

### Vérification Phase 5
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Tests passent (`PYTHONPATH=src python3 -m pytest test/test_vof_to_dpm.py -v`)
- [ ] Revue de code par agent

---

## Phase 6 : Nettoyer PR #17 — OpenFOAM 13 Tutorial Integration

**Objectif** : Helpers de tutorials et portage Tobias.

### Corrections
1. Déplacer les ~100 MB de binaires vers Git LFS
2. Supprimer les chemins `/home/shorty/...` des scripts SALOME
3. Rendre le bashrc configurable via `FOAM_BASHRC`

### Vérification Phase 6
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Revue de code par agent

---

## Phase 7 : Nettoyer PR #15 — YADE OpenFOAM 13

**Objectif** : Couplage YADE-OpenFOAM.

### Corrections
1. Remplacer chemins `/home/ubuntu/...` par `$HOME` ou relatifs
2. Ajouter `processor*/`, `*.gz`, `*.log` à `.gitignore`
3. Vérifier `Make/options` contre OF13

### Vérification Phase 7
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Revue de code par agent

---

## Phase 8 : Nettoyer PR #10 — Medical Build Pipeline

**Objectif** : Pipeline médical (vmtk, build123d, snappy).

### Corrections
1. Remplacer tous les chemins `/home/ubuntu/...` par relatifs
2. Fixer le FoamFile header dans `global_blockmesh.py`
3. Déplacer les 122 binaires vers Git LFS
4. Ajouter dépendances optionnelles (`pyvista`, `networkx`)
5. Remplacer `print()` par `logging`

### Vérification Phase 8
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Tests passent
- [ ] Revue de code par agent

---

## Phase 9 : Nettoyer PR #18 — MOOSE/OpenFOAM 13 Coupling

**Objectif** : Couplage MOOSE sans preCICE.

### Corrections
1. Supprimer les métriques de test hardcodées dans `train_cfd_gnn.py`
2. Restaurer la connectivité VTK exacte dans le GNN
3. Déplacer les tests vers `foampilot/test/`

### Vérification Phase 9
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Tests passent
- [ ] Revue de code par agent

---

## Phase 10 : Nettoyer PR #12 — MakeHuman JOS-3

**Objectif** : Couplage thermorégulation MakeHuman-JOS3-OpenFOAM.

### Corrections
1. Résoudre le conflit de merge avec `main` (déjà fait par rebase Phase 2)
2. Remplacer les chemins `/home/ubuntu/...` par configurations
3. Déclarer `jos3` dans `pyproject.toml` ou documenter le vendoring
4. Fixer `solver fluid;` → `application fluid;`
5. Ajouter le bloc `externalCoupled` à `controlDict`
6. Fixer le driver `qJOS3.in`

### Vérification Phase 10
- [ ] `python3 -m py_compile` sur les fichiers modifiés
- [ ] Tests passent
- [ ] Revue de code par agent

---

## Ordre de merge final

```
PR #20 (marine)          ← indépendante, merge après Phase 3
PR #19 (wolfdynamics)    ← dépend de Phase 4
PR #17 (tutorials)       ← dépend de Phase 6
PR #14 (VOF-to-DPM)      ← dépend de Phase 5
PR #15 (YADE)            ← dépend de Phase 7
PR #10 (medical_build)   ← dépend de Phase 8
PR #18 (MOOSE)           ← dépend de Phase 9
PR #12 (MakeHuman)       ← dépend de Phase 10, merge en dernier
```

---

## Notes

- Chaque phase est vérifiée par un agent indépendant avant de passer à la suivante
- Les fichiers consolidés (Phase 1) sont partagés par toutes les PRs
- Le rebasage (Phase 2) ne doit pas modifier le fond des PRs, seulement adapter la base
- Les corrections Phase 3-10 sont des modifications propres à chaque PR
