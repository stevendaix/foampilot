# Brief — Semaine 0 : "CFD et physique avec foampilot"

**Objectif** : Introduire les concepts physiques fondamentaux (Reynolds, viscosité, Poiseuille) et leur intégration dans foampilot.
**Angle** : Physique appliquée, validation, cas test Poiseuille.

---

## Structure proposée

### Introduction — Pourquoi la physique d'abord ?

Storytelling : En CFD, le code est secondaire. Ce qui compte, c'est la physique. Un maillage parfait, un solveur puissant — si les conditions physiques sont fausses, le résultat est faux.

**Promesse** : Dans cet article, je vous montre comment foampilot intègre la physique des fluides directement dans le workflow Python.

### Section 1 — Les propriétés des fluides

- Viscosité cinématique vs dynamique
- Tableau des fluides courants (air, eau, huile, sang)
- Calcul automatique avec pyfluids

### Section 2 — Le nombre de Reynolds

- Définition : Re = ρvL/μ
- Tableau des régimes (Stokes, laminaire, transitoire, turbulent)
- Cas concrets (voiture, avion, microcanal)

### Section 3 — Le cas test de Poiseuille

- Solution analytique : u(y) = (1/2ν) × (dp/dx) × (h² - y²)
- Validation quantitative
- Comparaison numérique vs analytique

### Section 4 — La couche limite et y+

- Définition de la couche limite
- Calcul de y+ avec foampilot
- Règles de maillage par modèle de turbulence

### Section 5 — Quantités physiques dérivées

- Contrainte de paroi τ_w
- Taux de déformation γ̇
- Profil de vitesse analytique vs numérique

### Section 6 — Du fluide à la simulation

- Workflow complet avec FluidMechanics
- Intégration dans Solver
- Génération de rapport PDF

### Conclusion — La physique comme fondation

CTA : *"Maintenant que vous maîtrisez la physique, je vous montre comment automatiser la création complète d'un cas OpenFOAM."*

---

## Code à préparer

- [ ] Script FluidMechanics + pyfluids
- [ ] Script calcul de Reynolds
- [ ] Script validation Poiseuille
- [ ] Script calcul y+, τ_w, strain rate
- [ ] Script workflow complet

## Images à préparer

- [ ] Tableau viscosités fluides
- [ ] Graphique régimes d'écoulement
- [ ] Schéma écoulement Poiseuille
- [ ] Graphique profil de vitesse numérique vs analytique
- [ ] Tableau règles de maillage
