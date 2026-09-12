# Brief — Semaine 9 : "LaTeX vs Typst : générer des rapports CFD professionnels"

**Objectif** : Comparer les deux moteurs de rapport de foampilot (LaTeX via PyLaTeX et Typst) pour les notes de calcul CFD.
**Angle** : Comparaison technique, tutoriel, choix d'outil.

---

## Structure proposée

### Introduction — Le rapport comme produit fini

Storytelling : Vous avez terminé une simulation. Vous devez maintenant produire une note de calcul pour votre responsable ou pour un client. Vous pourriez faire des captures d'écran Paraview et les coller dans Word… mais ce serait oublier que la CFD mérite mieux.

**Promesse** : Avec foampilot, vous générez des rapports PDF professionnels directement depuis Python. Deux moteurs s'offrent à vous : LaTeX (via PyLaTeX) et Typst. Voici comment choisir.

### Section 1 — Les besoins d'un rapport CFD

Un rapport CFD contient :
- **Méta-informations** : titre, auteur, date, numéro de version
- **Statistiques** : Reynolds, nombres sans dimension, valeurs aux parois
- **Tableaux** : résumé des paramètres, convergence
- **Figures** : streamlines, coupes, isosurfaces, maillage
- **Équations** : nombre de Reynolds, contrainte de paroi, etc.
- **Références** : littérature, normes

### Section 2 — La solution LaTeX avec PyLaTeX

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Écoulement de Poiseuille — Validation",
    author="Ingénieur CFD",
    filename="rapport_poiseuille"
)

doc.add_section("Résumé", "Validation de la simulation...")
doc.add_table(
    data=[["Re", "65800", "-", "Nombre de Reynolds"]],
    headers=["Paramètre", "Valeur", "Unité", "Description"],
    caption="Paramètres physiques"
)
doc.add_figure("streamlines.png", "Streamlines de vitesse")
doc.add_math(r"\tau_w = \mu \frac{du}{dy}\bigg|_{y=0}")
doc.generate_pdf()
```

**Avantages LaTeX** :
- Matûrité : 40 ans d'existence, communauté immense
- Packages spécialisés : `siunitx`, `booktabs`, `pgfplots`
- Compatibilité universelle
- Contrôle total sur la typographie

**Inconvénients LaTeX** :
- Courbe d'apprentissage abrupte
- Compilation lente (surtout avec figures)
- Gestion des erreurs de compilation difficile
- Pas de reflow dynamique

### Section 3 — La solution Typst

```python
from foampilot.report.typst_pdf import ScientificDocument

doc = ScientificDocument(
    title="Écoulement de Poiseuille — Validation",
    author="Ingénieur CFD"
)

doc.add_section("Résumé", "Validation de la simulation...", level=1)
doc.add_table(
    [["Paramètre", "Valeur", "Unité", "Description"],
     ["Re", "65800", "-", "Nombre de Reynolds"]],
    caption="Paramètres physiques",
    label="tab:params"
)
doc.add_figure("streamlines.png", caption="Streamlines de vitesse")
# Typst supporte le HTML-like pour les équations
doc.render(doc)
```

**Avantages Typst** :
- Moderne : compilé en Rust, rapide
- Syntaxe plus simple (HTML-like)
- Reflow dynamique
- Intégration native des fonts modernes
- Meilleur support des SVG

**Inconvénients Typst** :
- Écosystème plus jeune
- Moins de packages spécialisés
- Communauté plus petite
- Support des équations complexes en cours de maturation

### Section 4 — Comparaison détaillée

| Critère | LaTeX | Typst |
|---------|-------|-------|
| Courbe d'apprentissage | Steep | Douce |
| Vitesse de compilation | Moyenne | Rapide |
| Gestion des erreurs | Cryptique | Claire |
| Équations | ✅ Excellent | ✅ Bon |
| Tableaux | ✅ Excellent | ✅ Bon |
| Figures | ✅ Excellent | ✅ Bon |
| Bibliographie | BibTeX natif | En maturation |
| Communauté | Immense | Grandissante |
| Intégration Python | PyLaTeX mature | API naissante |

### Section 5 — Cas d'usage recommandés

**Choisir LaTeX si** :
- Vous avez déjà une équipe LaTeX
- Vous avez besoin de packages spécialisés (`siunitx`, `chemfig`, etc.)
- Vous publiez dans des revues qui requièrent LaTeX
- Vous maîtrisez déjà LaTeX

**Choisir Typst si** :
- Vous démarrez un nouveau projet
- Vous voulez une courbe d'apprentissage douce
- Vous priorisez la vitesse de compilation
- Vous aimez la syntaxe moderne
- Vous voulez du reflow dynamique

### Section 6 — Intégration dans le workflow foampilot

```python
from foampilot.report import CFDReportGenerator

report = CFDReportGenerator(
    case_path="./poiseuille",
    title="Écoulement de Poiseuille — Validation"
)

# Ajouter des statistiques
report.add_statistic("Re", 65800, "-", "Nombre de Reynolds")
report.add_statistic("nu", 1.52e-5, "m²/s", "Viscosité cinématique")

# Générer les deux versions
report.save_latex_report(compile_pdf=True)      # LaTeX
report.save_typst_report()                       # Typst
```

### Conclusion — Le meilleur des deux mondes

CTA : *"Dans le prochain article, je vous montre comment automatiser la génération de figures et l'intégration dans vos rapports."*

---

## Code à préparer

- [ ] Script rapport LaTeX complet
- [ ] Script rapport Typst complet
- [ ] Script de comparaison côte à côte
- [ ] Benchmark compilation (temps, taille PDF)

## Images à préparer

- [ ] Capture d'écran PDF LaTeX
- [ ] Capture d'écran PDF Typst
- [ ] Graphique de temps de compilation
- [ ] Comparaison visuelle côte à côte
