# Articles — Système de Publication Agile

**Principe** : Préparer plusieurs articles en avance, les améliorer entre versions, et ne publier que quand ils sont prêts.
**Rythme** : Pas de calendrier fixe. Sessions de révision bi-mensuelles + publication quand l'article est bon.
**Langues** : Chaque article est rédigé en **français** et en **anglais**.

---

## Philosophie

> *"Un article publié imparfait vaut mieux qu'un article parfait qui dort dans un dossier."*

Mais : **un article amélioré vaut mieux qu'un article publié vite.**

Ce système permet de :
- **Construire une réserve** d'articles prêts ou presque prêts
- **Améliorer la qualité** au fil des versions (v1 → v2 → v3)
- **Publier au bon moment** (quand l'article est bon, pas quand le calendrier l'impose)
- **Apprendre de chaque publication** pour améliorer les suivants

---

## Structure

```
articles/
├── README.md                 # Ce fichier — vue d'ensemble du workflow
├── template.md               # Template d'article standardisé
├── checklist.md              # Checklist pré-publication
├── calendar.md               # Calendrier éditorial (vision long terme)
├── published/                # Articles publiés
│   ├── week_01_design_opensource.md
│   ├── week_01_project_structure.md
│   └── week_01_project_structure_en.md
├── project/              # Semaine 1 — architecture, design, présentation
│   ├── brief_week_01_project_structure.md
│   └── week_01_project_structure.md
├── drafts/                   # Brouillons en cours (non publiés)
│   ├── physics/              # Semaine 0 — physique, Reynolds, Poiseuille
│   │   ├── brief_week_00.md
│   │   ├── week_00_physics_cfd.md
│   │   └── week_00_physics_cfd_en.md
│   ├── automation/           # Semaine 2 — automatisation OpenFOAM
│   │   ├── brief_week_02.md
│   │   └── week_02_automation_tutorial.md
│   ├── api-design/           # Semaine 3 — design patterns, répétition intentionnelle
│   │   ├── brief_week_03.md
│   │   ├── week_03_api_design_patterns.md
│   │   └── week_03_api_design_patterns_en.md
│   ├── visualization/        # Semaine 4 — visualisation PyVista
│   │   ├── brief_week_04.md
│   │   └── week_04_visualization.md
│   ├── cht/                  # Semaine 5 — CHT multi-région
│   │   ├── brief_week_05.md
│   │   └── week_05_cht_multiregion.md
│   ├── community/            # Semaine 6 — adoption open-source
│   │   ├── brief_week_06.md
│   │   ├── week_06_community_adoption.md
│   │   └── week_06_community_adoption_en.md
│   ├── gmsh-export/          # Semaine 7 — export direct Gmsh → OpenFOAM
│   │   ├── brief_week_07_gmsh_direct_export.md
│   │   ├── week_07_gmsh_direct_export.md
│   │   └── week_07_gmsh_direct_export_en.md
│   ├── pyvista-reader/       # Semaine 8 — reader direct OpenFOAM → PyVista
│   │   ├── brief_week_08_pyvista_reader.md
│   │   ├── brief_week_08_pyvista_reader_en.md
│   │   ├── week_08_pyvista_reader.md
│   │   └── week_08_pyvista_reader_en.md
│   └── latex-vs-typst/       # Semaine 9 — rapports LaTeX vs Typst
│       ├── brief_week_09_latex_vs_typst.md
│       ├── brief_week_09_latex_vs_typst_en.md
│       ├── week_09_latex_vs_typst.md
│       └── week_09_latex_vs_typst_en.md
├── queue/                    # File de publication (articles prêts)
│   └── (vide)                # Article publié — en attente du prochain
├── reviews/                  # Sessions de révision
│   ├── session_2026-08-10.md # Exemple de session
│   └── feedback_template.md  # Template de feedback
└── changelog.md              # Historique global des améliorations
```

---

## Workflow en 5 étapes

### Étape 1 : Rédaction (drafts/v1/)

1. **Choisir un sujet** dans `calendar.md` ou proposer un nouveau
2. **Créer le fichier FR** dans `drafts/v1/week_XX_slug.md`
3. **Créer le fichier EN** dans `drafts/v1/week_XX_slug_en.md`
4. **Rédiger** selon `template.md`
5. **Ajouter le code** et les visuels
6. **Auto-relecture** : passer la `checklist.md`

**Objectif** : Produire une version "complète" mais perfectible. Pas de publication ici.

### Étape 2 : Révision collective (reviews/)

1. **Planifier une session** de révision (tous les 2 lundis par exemple)
2. **Lire** chaque article de `drafts/v1/` ou `drafts/v2/`
3. **Remplir** le `feedback_template.md` pour chaque article
4. **Noter** l'article sur 5 critères :
   - Clarté du titre et de l'intro
   - Qualité des exemples de code
   - Structure logique
   - Ton et style
   - Actionnabilité du CTA

5. **Décider** : publier directement, améliorer (v2), ou abandonner

**Output** : Sessions documentées dans `reviews/session_DATE.md`

### Étape 3 : Amélioration (drafts/v2/, drafts/v3/)

1. **Prendre les feedbacks** de la session
2. **Améliorer** l'article dans `drafts/v2/slug.md`
3. **Itérer** jusqu'à `drafts/v3/` si nécessaire
4. **Valider** avec la checklist

**Principe** : Chaque version traite un feedback spécifique. On garde trace des changements.

### Étape 4 : Publication (queue/)

1. **Déplacer** l'article validé vers `queue/slug.md`
2. **Ajouter** les métadonnées de publication (date cible, tags, etc.)
3. **Publier** sur Medium quand c'est prêt
4. **Déplacer** vers `published/` après publication

**Règle d'or** : Pas de retour en arrière après publication. Si l'article est médiocre, on écrit un meilleur article sur le même sujet.

### Étape 5 : Apprentissage (changelog.md)

1. **Noter** les statistiques de l'article publié (vues, claps, commentaires)
2. **Identifier** ce qui a marché / pas marché
3. **Mettre à jour** `changelog.md` avec les leçons
4. **Ajuster** le `template.md` si nécessaire

---

## Système de versions

Chaque article suit un cycle de versions :

```
drafts/v1/ → drafts/v2/ → drafts/v3/ → queue/ → published/
   brouillon   amélioré      final        prêt       publié
```

**Quand créer une nouvelle version ?**
- v1 → v2 : après une session de révision avec des feedbacks majeurs
- v2 → v3 : après des feedbacks mineurs (orthographe, reformulation)
- v3 → queue : quand l'article est considéré "bon" par au moins 2 personnes

**Nom de fichier** :
- `week_XX_slug_v1.md` (FR)
- `week_XX_slug_v1_en.md` (EN)
- `week_XX_slug_v2.md` / `week_XX_slug_v2_en.md`
- `week_XX_slug.md` / `week_XX_slug_en.md` (dans queue/ ou published/)

---

## File de publication (queue/)

La file contient les articles **prêts à être publiés**, classés par priorité.

La file est actuellement **vide**. Le prochain article depuis `drafts/` doit être déplacé vers `queue/`.

### Priorité pour le remplissage
1. **Week 0** — `drafts/physics/week_00_physics_cfd.md` (+ `_en.md`) — plus avancé, en amélioration
2. **Week 2** — `drafts/automation/week_02_automation_tutorial.md` — v1 rédigé
3. **Week 3** — `drafts/api-design/week_03_api_design_patterns.md` — v1 rédigé

### Règle de la file
- Maximum **3 articles** dans la file à tout moment
- Quand on publie un article, on le remplace par un nouveau depuis `drafts/v3/`
- Si la file est vide, on fait une session de révision pour la remplir

---

## Sessions de révision

### Fréquence
- **Tous les 2 lundis** (ou toutes les semaines si rythme soutenu)
- Durée : 1h30 par session

### Format
1. **Tour de table** (10 min) : chaque personne présente 1 article qu'elle a rédigé
2. **Lecture silencieuse** (30 min) : chacun lit les articles et prend des notes
3. **Discussion** (40 min) : feedback structuré par article
4. **Décisions** (10 min) : publier / améliorer / abandonner

### Feedback template

```markdown
# Feedback — week_XX_slug_v1.md

## Résumé
**Article** : Titre
**Auteur** : Nom
**Date de révision** : Date

## Forces (3 points)
1. 
2. 
3. 

## Axes d'amélioration (3 points)
1. 
2. 
3. 

## Recommandation
- [ ] Publier directement
- [ ] Améliorer (v2)
- [ ] Abandonner

## Commentaires détaillés
...
```

---

## Changelog global

`changelog.md` contient l'historique de toutes les versions et les leçons apprises.

```markdown
# Changelog — Améliorations des articles

## 2026-08-03 — Session de révision #1

### Article : week_01_design_opensource.md
**Version** : v1 → publié
**Leçons** :
- Les exemples de code doivent être plus courts (max 10 lignes)
- Les titres de section doivent être plus évocateurs
- Le CTA doit être plus direct

### Article : week_02_automation.md
**Version** : v1 → v2
**Améliorations** :
- Ajout d'une section "Ce qui se passe sous le capot"
- Réduction du code de 30 à 15 lignes
- Ajout de 2 schémas de workflow

## 2026-08-10 — Session de révision #2
...
```

---

## Règles d'or

1. **Pas de pression de calendrier** : on publie quand c'est bon, pas quand c'est prévu
2. **Amélioration continue** : chaque article peut être révisé entre versions
3. **Feedback constructif** : on critique le contenu, pas la personne
4. **Code testé** : tout exemple de code doit être exécuté avant inclusion
5. **Visuels soignés** : un schéma vaut mieux qu'un paragraphe technique
6. **Bilingue systématique** : chaque article est rédigé en FR et EN en parallèle

---

## Métriques de succès (par article)

| Phase | Critère de passage |
|-------|---------------------|
| v1 → v2 | Feedback positif de 2 relecteurs |
| v2 → v3 | Tous les feedbacks majeurs traités |
| v3 → queue | Checklist pré-publication validée à 100% |
| queue → published | Décision conjointe de publication |

---

## Articles en cours

| Semaine | Titre | Statut | Fichiers |
|---------|-------|--------|----------|
| 0 | CFD et physique avec foampilot | 🔧 En cours d'amélioration (v1.1) | `drafts/physics/week_00_physics_cfd.md` + `_en.md` |
| 1 | J'ai construit un wrapper Python pour OpenFOam | ✅ Publié (Medium, CFD Online, Reddit) | `published/week_01_project_structure.md` + `_en.md` |
| 2 | Automatiser OpenFOAM en 20 lignes | 📝 v1 rédigé | `drafts/automation/week_02_automation_tutorial.md` |
| 3 | La répétition intentionnelle comme vertu API | 📝 v1 rédigé | `drafts/api-design/week_03_api_design_patterns.md` + `_en.md` |
| 4 | Visualiser OpenFOAM sans foamToVTK | 📝 v1 rédigé | `drafts/visualization/week_04_visualization.md` + `_en.md` |
| 5 | CHT multi-région avec foampilot | 📝 v1 rédigé | `drafts/cht/week_05_cht_multiregion.md` + `_en.md` |
| 6 | Rendre un outil CFD adoptable | 📝 v1 rédigé | `drafts/community/week_06_community_adoption.md` + `_en.md` |
| 7 | Export direct Gmsh → OpenFOAM | 📝 v1 rédigé | `drafts/gmsh-export/week_07_gmsh_direct_export.md` + `_en.md` |
| 8 | Reader direct OpenFOAM → PyVista | 📝 v1 rédigé | `drafts/pyvista-reader/week_08_pyvista_reader.md` + `_en.md` |
| 9 | LaTeX vs Typst pour rapports CFD | 📝 v1 rédigé | `drafts/latex-vs-typst/week_09_latex_vs_typst.md` + `_en.md` |

---

## Outils recommandés

- **Rédaction** : Obsidian / VS Code
- **Schémas** : Excalidraw / Draw.io
- **Review** : GitHub PRs sur les drafts, ou Google Docs
- **Suivi** : Ce dossier + `changelog.md`

---

*Dernière mise à jour : 2026-08-05*
