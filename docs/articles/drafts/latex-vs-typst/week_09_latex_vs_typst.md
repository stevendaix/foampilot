# Générer des rapports CFD : comparaison LaTeX vs Typst avec foampilot

*Comment choisir entre LaTeX (PyLaTeX) et Typst pour générer des notes de calcul PDF professionnelles depuis Python, avec foampilot.*

---

## Introduction : Le rapport comme produit fini

Vous avez terminé une simulation. Vous devez maintenant produire une note de calcul pour votre responsable ou pour un client. Vous pourriez faire des captures d'écran Paraview et les coller dans Word… mais ce serait oublier que la CFD mérite mieux.

**Promesse** : Avec foampilot, vous générez des rapports PDF professionnels directement depuis Python. Deux moteurs s'offrent à vous : LaTeX (via PyLaTeX) et Typst. Voici comment choisir.

---

## Partie 1 : Les besoins d'un rapport CFD

Un rapport CFD contient :
- **Méta-informations** : titre, auteur, date, numéro de version
- **Statistiques** : Reynolds, nombres sans dimension, valeurs aux parois
- **Tableaux** : résumé des paramètres, convergence
- **Figures** : streamlines, coupes, isosurfaces, maillage
- **Équations** : nombre de Reynolds, contrainte de paroi, etc.
- **Références** : littérature, normes

---

## Partie 2 : La solution LaTeX avec PyLaTeX

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

### Avantages LaTeX

- **Matûrité** : 40 ans d'existence, communauté immense
- **Packages spécialisés** : `siunitx`, `booktabs`, `pgfplots`
- **Compatibilité universelle** : accepté par toutes les revues
- **Contrôle total** sur la typographie et la mise en page

### Inconvénients LaTeX

- **Courbe d'apprentissage abrupte** : syntaxe parfois cryptique
- **Compilation lente** : surtout avec beaucoup de figures
- **Gestion des erreurs** : messages d'erreur difficiles à interpréter
- **Pas de reflow dynamique** : la mise en page est figée

---

## Partie 3 : La solution Typst

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
doc.render(doc)
```

### Avantages Typst

- **Moderne** : compilé en Rust, très rapide
- **Syntaxe simple** : HTML-like, plus intuitive que LaTeX
- **Reflow dynamique** : la mise en page s'adapte au contenu
- **Fonts modernes** : intégration native de fonts modernes
- **Meilleur support SVG** : rendu vectoriel de haute qualité

### Inconvénients Typst

- **Écosystème jeune** : moins de packages que LaTeX
- **Communauté plus petite** : moins de ressources en ligne
- **Équations complexes** : support en cours de maturation
- **Intégration Python** : API encore naissante

---

## Partie 4 : Comparaison détaillée

| Critère | LaTeX | Typst |
|---------|-------|-------|
| Courbe d'apprentissage | Steep | Douce |
| Vitesse de compilation | Moyenne (2-5s) | Rapide (<1s) |
| Gestion des erreurs | Cryptique | Claire |
| Équations | ✅ Excellent | ✅ Bon |
| Tableaux | ✅ Excellent | ✅ Bon |
| Figures | ✅ Excellent | ✅ Bon |
| Bibliographie | BibTeX natif | En maturation |
| Communauté | Immense | Grandissante |
| Intégration Python | PyLaTeX mature | API naissante |
| Taille du PDF | ~200 KB | ~150 KB |

---

## Partie 5 : Cas d'usage recommandés

### Choisir LaTeX si :

- Vous avez déjà une équipe LaTeX
- Vous avez besoin de packages spécialisés (`siunitx`, `chemfig`, `tikz`)
- Vous publiez dans des revues qui requièrent LaTeX
- Vous maîtrisez déjà LaTeX

### Choisir Typst si :

- Vous démarrage un nouveau projet
- Vous voulez une courbe d'apprentissage douce
- Vous priorisez la vitesse de compilation
- Vous aimez la syntaxe moderne
- Vous voulez du reflow dynamique

---

## Partie 6 : Intégration dans le workflow foampilot

```python
from foampilot.report import CFDReportGenerator

report = CFDReportGenerator(
    case_path="./poiseuille",
    title="Écoulement de Poiseuille — Validation"
)

# Ajouter des statistiques
report.add_statistic("Re", 65800, "-", "Nombre de Reynolds")
report.add_statistic("nu", 1.52e-5, "m²/s", "Viscosité cinématique")
report.add_statistic("y+_max", 0.8, "-", "y+ maximum")

# Générer les deux versions
report.save_latex_report(compile_pdf=True)      # LaTeX
report.save_typst_report()                       # Typst
```

### Comparaison côte à côte

```python
import time

# LaTeX
start = time.time()
report.save_latex_report(compile_pdf=True)
latex_time = time.time() - start

# Typst
start = time.time()
report.save_typst_report()
typst_time = time.time() - start

print(f"LaTeX: {latex_time:.2f}s")
print(f"Typst: {typst_time:.2f}s")
```

---

## Conclusion : Le meilleur des deux mondes

LaTeX et Typst ne sont pas en compétition — ils répondent à des besoins différents. foampilot vous laisse choisir selon votre contexte.

**Mon conseil personnel** : si vous débutez, commencez par Typst. Si vous avez déjà une base LaTeX, restez sur LaTeX. Dans les deux cas, foampilot génère le rapport pour vous — il ne vous reste plus qu'à choisir le moteur.

Dans le prochain article, je vous montre comment automatiser la génération de figures et l'intégration dans vos rapports.

**Ressources :**
- PyLaTeX : https://pylatex.readthedocs.io
- Typst : https://typst.app
- foampilot report module : https://stevendaix.github.io/foampilot/

---

*Article en cours d'amélioration — version 1.0*
