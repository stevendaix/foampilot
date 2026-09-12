# Calendrier Éditorial — Publication Hebdomadaire

**Objectif** : Publier un article par semaine sur Medium, pendant 6 semaines minimum.
**Public** : Ingénieurs CFD / développeurs Python + chercheurs en open-source scientifique.
**Angle global** : Automatiser OpenFOAM avec Python, design d'outils scientifiques, reproductibilité.

---

## Semaine 0 : La physique — "CFD et physique avec foampilot"

**Statut** : 🔧 En cours d'amélioration (v1.1) — `drafts/physics/week_00_physics_cfd.md` (+ `_en.md`)
**Corrections v1.1** : paramètres Poiseuille cohérents avec le test case, `cell_data`→`point_data` pour wall shear stress, code Partie 5.3 autonome, Partie 6 corrigée, typo "parai"→"paroi", note foamToVTK, bug `calculate_reynolds()` fixé dans le code source.
**Titre** : *CFD et physique : calculer Reynolds, viscosité et couche limite avec foampilot*
**Angle** : Concepts physiques fondamentaux, intégration pyfluids, validation Poiseuille
**Mots-clés** : CFD, Reynolds, viscosité, pyfluids, Poiseuille, physique des fluides
**CTA** : Vers la semaine 2 (automatisation)

---

## Semaine 1 : Architecture et design — "J'ai construit un wrapper Python pour OpenFOam"

**Statut** : ✅ Publié (Medium, CFD Online, Reddit r/Python + r/CFD) — `published/week_01_project_structure.md` + `published/week_01_project_structure_en.md`
**Titre FR** : *J'ai construit un wrapper Python pour OpenFOam : architecture, design et leçons d'un projet open-source*
**Titre EN** : *I Built a Python Wrapper for OpenFOAM: Architecture, Design, and Lessons from an Open-Source Project*
**Angle** : Architecture logicielle, retour d'expérience, présentation de projet
**Mots-clés** : OpenFOAM, Python, CFD, architecture, design logiciel, open-source
**CTA** : Vers la semaine 0 (physique)

---

## Semaine 2 : Le tutoriel — "Automatiser OpenFOAM en 20 lignes de Python"

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_02_automation_tutorial.md` (Français)
- `drafts/v1/week_02_automation_tutorial_en.md` (English)
**Titre** : *Automatiser OpenFOAM avec Python : guide pas-à-pas pour des workflows reproductibles*
**Angle** : Tutoriel pratique, "best practices", cas Poiseuille avec validation physique
**Mots-clés** : OpenFOAM, Python, automatisation, tutorial, Reynolds, Poiseuille, pyfluids
**CTA** : Vers la semaine 3 (design patterns)

---

## Semaine 3 : Le design pattern — "La répétition intentionnelle dans les APIs scientifiques"

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_03_api_design_patterns.md` (Français)
- `drafts/v1/week_03_api_design_patterns_en.md` (English)
**Titre** : *Pourquoi j'ai volontairement répété du code dans foampilot : la répétition intentionnelle comme vertu API*
**Angle** : Deep-dive technique sur le design d'API, philosophie DRY vs prévisibilité
**Mots-clés** : API design, Python, OpenFOAM, design patterns, UX développeur
**CTA** : Vers la semaine 4 (post-processing)

---

## Semaine 4 : Le post-processing — "Visualiser OpenFOAM sans foamToVTK"

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_04_visualization.md` (Français)
- `drafts/v1/week_04_visualization_en.md` (English)
**Titre** : *Visualiser OpenFOAM directement avec PyVista : fini foamToVTK*
**Angle** : Tutoriel technique sur le lecteur direct, extraction de quantités physiques (y+, τ_w, strain rate)
**Mots-clés** : OpenFOAM, PyVista, visualisation, post-processing, CFD, y+, contrainte de paroi
**CTA** : Vers la semaine 5 (CHT)

---

## Semaine 5 : Le cas avancé — "CHT multi-région avec foampilot"

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_05_cht_multiregion.md` (Français)
- `drafts/v1/week_05_cht_multiregion_en.md` (English)
**Titre** : *Conjugate Heat Transfer avec OpenFOAM : du Gmsh au chtMultiRegionFoam en Python*
**Angle** : Cas d'usage avancé, multi-région, interfaces couplées, Gmsh direct export
**Mots-clés** : CHT, OpenFOAM, Gmsh, conjugate heat transfer, multi-region
**CTA** : Vers la semaine 6 (communauté)

---

## Semaine 6 : La communauté — "Comment rendre un outil CFD adoptable"

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_06_community_adoption.md` (Français)
- `drafts/v1/week_06_community_adoption_en.md` (English)
**Titre** : *Rendre un outil CFD adoptable : leçons tirées de 6 mois de foampilot sur GitHub*
**Angle** : Communication, open-source, adoption par la communauté, documentation
**Mots-clés** : open-source, CFD, communauté, documentation, GitHub
**CTA** : Boucle vers la semaine 0, annonce des prochains sujets

---

## Semaine 7 : Export direct Gmsh → OpenFOAM

**Statut** : 📝 Brief prêt (`drafts/brief_week_07_gmsh_direct_export.md`)
**Titre** : *Export direct Gmsh vers OpenFOAM : écrire polyMesh sans gmshToFoam*
**Angle** : Deep-dive technique, algorithmes d'export, suppression d'un maillon faible
**Mots-clés** : Gmsh, OpenFOAM, polyMesh, export direct, maillage, algorithmes
**CTA** : Vers la semaine 8 (lecture PyVista)

---

## Semaine 8 : Reader direct OpenFOAM → PyVista

**Statut** : 📝 Brief prêt (`drafts/brief_week_08_pyvista_reader.md`)
**Titre** : *Lire OpenFOAM directement dans PyVista : architecture du reader direct et extraction de quantités physiques*
**Angle** : Architecture logicielle, performance, extraction y+/τ_w/strain rate
**Mots-clés** : PyVista, OpenFOAM, reader, post-processing, y+, contrainte de paroi, visualisation
**CTA** : Vers la semaine 9 (rapports LaTeX vs Typst)

---

## Semaine 9 : LaTeX vs Typst pour rapports CFD

**Statut** : 📝 Brief prêt (`drafts/brief_week_09_latex_vs_typst.md`)
**Titre** : *Générer des rapports CFD : comparaison LaTeX vs Typst avec foampilot*
**Angle** : Comparaison technique, tutoriel, choix d'outil pour notes de calcul
**Mots-clés** : LaTeX, Typst, rapport, PDF, CFD, PyLaTeX, note de calcul
**CTA** : Boucle vers la semaine 0, annonce des prochains sujets

---

## Semaine 7 : Export direct Gmsh → OpenFOAM

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_07_gmsh_direct_export.md` (Français)
- `drafts/v1/week_07_gmsh_direct_export_en.md` (English)
**Titre** : *Export direct Gmsh vers OpenFOAM : écrire polyMesh sans gmshToFoam*
**Angle** : Deep-dive technique, algorithmes d'export, suppression d'un maillon faible
**Mots-clés** : Gmsh, OpenFOAM, polyMesh, export direct, maillage, algorithmes
**CTA** : Vers la semaine 8 (lecture PyVista)

---

## Semaine 8 : Reader direct OpenFOAM → PyVista

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_08_pyvista_reader.md` (Français)
- `drafts/v1/week_08_pyvista_reader_en.md` (English)
**Titre** : *Lire OpenFOAM directement dans PyVista : architecture du reader et extraction de quantités physiques*
**Angle** : Architecture logicielle, performance, extraction y+/τ_w/strain rate
**Mots-clés** : PyVista, OpenFOAM, reader, post-processing, y+, contrainte de paroi, visualisation
**CTA** : Vers la semaine 9 (rapports LaTeX vs Typst)

---

## Semaine 9 : LaTeX vs Typst pour rapports CFD

**Statut** : 📝 v1 rédigé (FR + EN)
**Fichiers** :
- `drafts/v1/week_09_latex_vs_typst.md` (Français)
- `drafts/v1/week_09_latex_vs_typst_en.md` (English)
**Titre** : *Générer des rapports CFD : comparaison LaTeX vs Typst avec foampilot*
**Angle** : Comparaison technique, tutoriel, choix d'outil pour notes de calcul
**Mots-clés** : LaTeX, Typst, rapport, PDF, CFD, PyLaTeX, note de calcul
**CTA** : Boucle vers la semaine 0, annonce des prochains sujets

---

## Structure bilingue

Chaque article est rédigé en **deux versions** :
- **Français** (`week_XX_slug.md`) : pour le public francophone
- **English** (`week_XX_slug_en.md`) : pour le public international

Les deux versions suivent la même structure et les mêmes exemples de code. Seul le texte change.

---

## Récurrence

- **Lundi** : Session de révision bi-mensuelle (si applicable)
- **Mardi** : Rédaction du brouillon (drafts/v1/)
- **Mercredi** : Revue et ajustements
- **Jeudi** : Intégration des exemples de code, captures d'écran
- **Vendredi** : Publication sur Medium (si article dans queue/)
- **Week-end** : Partage sur réseaux, réponse aux commentaires

---

## Idées pour les semaines 10+

1. **Études paramétriques** : *"Lancer 100 simulations OpenFOAM en 10 minutes avec Python"*
2. **Tests et CI** : *"Tester des cas OpenFOAM automatiquement avec pytest"*
3. **Comparaison solveurs** : *"simpleFoam vs pimpleFoam : automatiser la comparaison avec foampilot"*
4. **Optimisation** : *"Optimiser un maillage snappyHexMesh par algorithme génétique en Python"*
5. **Turbulence avancée** : *"Comprendre et configurer les modèles de turbulence avec foampilot"*
6. **Méthodes numériques** : *"Schémas upwind vs central differencing : impact sur la précision"*
7. **Gmsh géométries complexes** : *"Générer des géométries complexes pour CFD avec Gmsh et Python"*
8. **Post-processing avancé** : *"Analyse de convergence et validation de maillage avec PyVista"*
