# Changelog — Améliorations des articles

## 2026-08-05 — Publication semaine 1 + amélioration semaine 0

### Article : week_01_project_structure.md
**Version** : queue → publié
**Publication** : Medium, CFD Online, Reddit (r/Python + r/CFD)
**Leçons** :
- Article d'accroche publié, file de publication désormais vide
- Prochain remplissage priorisé : Week 0 (physique, plus avancé)

### Article : week_00_physics_cfd.md
**Version** : v1 → v1.1
**Améliorations** :
- Partie 3 : remplacement des paramètres pyfluids (air, Re=131 579) par les paramètres du cas test Poiseuille (nu=0.1, Re=250, laminaire) pour cohérence avec `test_cfd_methods.py`
- Partie 5.1 : correction `cell_data["wall_shear_stress"]` → `point_data["wall_shear_stress"]` (le method `calc_wall_shear_stress` stocke dans `point_data`)
- Partie 5.3 : code autonome — les variables `nu`, `G`, `H` sont définies localement au lieu de dépendre de la Partie 3
- Partie 6 : ajout de `velocity` et `characteristic_length` au constructeur `FluidMechanics` (requis par `calculate_reynolds()`); suppression de `physicalProperties.rho0` (solveur incodable → incompressible)
- Correction typo : "contrainte de parai" → "contrainte de paroi"
- Note ajoutée : `get_structure()` nécessite `foamToVTK` exécuté au préalable

### Code source : fluids_theory.py
**Bug fix** : `calculate_reynolds()`, `calculate_prandtl()`, `calculate_rayleigh()` appelaient `get_fluid_properties(arg)` mais la méthode n'accepte aucun argument → `TypeError`. Corrigé en appelant `get_fluid_properties()` sans arguments.

---

## 2026-08-03 — Publication semaine 1 (première version)

### Article : week_01_design_opensource.md
**Version** : v1 → publié
**Leçons** :
- Les exemples de code doivent être plus courts (max 10 lignes)
- Les titres de section doivent être plus évocateurs
- Le CTA doit être plus direct

---

## Prochaine session de révision

**Date cible** : 2026-08-12
**Articles à revoir** :
- Week 0 (physique) — version 1.1
- Week 2 (automatisation) — v1 rédigé
- Week 3 (design patterns) — v1 rédigé (FR + EN)
