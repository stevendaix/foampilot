# Comparaison des méthodes P01-P24 — Détection inlet/outlet patient58

## Tableau récapitulatif

| Test | Méthode | # Inlet | # Outlet | # Wall/Other | Confiance | Convention utilisée | Image |
|------|---------|---------|----------|--------------|-----------|---------------------|-------|
| P01 | Boundary loops | 0 | 0 | 1704 boucles | - | - | ✅ |
| P02 | Centerline PCA | 1 | 1 | - | Élevée | s_min=inlet, s_max=outlet | ✅ |
| P03 | Angle normale/tangente | 1 | 1 | 1 | Moyenne | Patch OpenFOAM | ✅ |
| P04 | Caps plans | 1 | 1 | 0 | Élevée | s_min=inlet, s_max=outlet | ✅ |
| P05 | Paroi cylindrique | 0 | 0 | 1057 paroi, 9536 candidates | - | - | ✅ |
| P06 | Courbure | 0 | 0 | 5131 cap, 6877 wall | - | Seuils | ✅ |
| P07 | Region growing | - | - | 430 régions | - | - | ✅ |
| P08 | Clustering | - | - | 3 clusters | - | - | ✅ |
| P09 | PCA | 0 | 0 | - | - | - | ✅ |
| P10 | Squelette | 1 | 1 | - | Élevée | s_min=inlet, s_max=outlet | ✅ |
| P11 | Distance field | 1 | 1 | 1 | Élevée | Patch OpenFOAM | ✅ |
| P12 | Slicing | - | - | - | - | - | ✅ |
| P13 | Graphe topologique | 1 | 1 | 1 | Élevée | opening_0/1 | ✅ |
| P14 | Distance géodésique | 1 | 1 | - | Élevée | Extrémités géodésiques | ✅ |
| P15 | Formes primitives | - | - | - | - | - | ✅ |
| P16 | ML | - | - | - | 100% RF | - | ✅ |
| P17 | Interactive | 1 | 1 | 0 | Élevée | Sélection manuelle | ✅ |
| P18 | Convention | 1 | 1 | 1 | Élevée | s_min=inlet, s_max=outlet | ✅ |
| P19 | Vote hybride | 1 | 1 | 9074 uncertain | 0.79 | Seuils | ✅ |
| P20 | Pipeline VTK/VMTK | 0 | 1 | - | - | Convention | ✅ |
| P21 | Stratégie robuste | - | - | - | - | - | ✅ |
| P22 | Critères numériques | 0 | 0 | 0 | - | - | ✅ |
| P23 | Tests prioritaires | 1 | 1 | 1 | Élevée | Multi-méthodes | ✅ |
| P24 | Conclusion | 1 | 1 | 1 | Élevée | Convention finale | ✅ |

## Analyse

### Méthodes avec détection fiable inlet/outlet
- **P02** : Centerline PCA — 2 extrémités claires
- **P04** : Caps plans — 2 caps détectés
- **P11** : Patch OpenFOAM — 3 patches explicites
- **P13** : Graphe topologique — 2 openings
- **P17** : Interactive — sélection manuelle
- **P18** : Convention s_min/s_max
- **P23/P24** : Combinaison multi-méthodes

### Méthodes sans détection directe
- P01, P05, P06, P07, P08, P09, P10, P12, P14, P15, P16, P21, P22
  → Ces méthodes fournissent des informations utiles mais ne classificient pas directement inlet/outlet.

### Comparaison des conventions
- **s_min → inlet, s_max → outlet** : P02, P04, P18, P23, P24
- **Angle normale/tangente** : P03, P19, P24
- **Patch OpenFOAM** : P03, P11
- **Interactive** : P17

### Recommandation
La méthode la plus robuste est la **combinaison P13 + P18 + P23/P24** :
1. Détecter les openings avec P13 (graphe topologique)
2. Appliquer la convention P18 (s_min=inlet, s_max=outlet)
3. Valider avec P23 (tests prioritaires)
