# Suivi — Détection inlet/outlet patient58

## Contexte
- Cas OpenFOAM : `patient58_cfd_example`
- Surface totale : **10 593 faces** (INLET: 376, OUTLET: 2512, WALL: 7705)
- Tests P01–P24 déjà exécutés avec images et résultats.
- Objectif : comparer si les méthodes fiables sélectionnent **exactement les mêmes zones inlet/outlet**.

## Méthodes fiables retenues
- **P02** — Centerline PCA
- **P04** — Caps plans / région croissante
- **P11** — Patches OpenFOAM existants
- **P13** — Graphe topologique (lit les patches)
- **P18** — Convention s_min/s_max (lit les patches)
- **P23** — Tests prioritaires (même détection que P04)
- **P24** — Conclusion / pipeline combiné (même détection que P04)

## Résultats officiels des tests originaux

### P02 — Centerline PCA
- **Inlet** : extrémité s_min = [0.2679, 0.2834, 0.0180]
- **Outlet** : extrémité s_max = [0.2599, 0.2440, 0.0040]
- Nombre de faces : **0** (seulement des points centerline)

### P04 — Détection de caps plans
- **Cap 1 (outlet)** : 50 faces, aire=0.0040, centre=[0.2264, 0.1809, 0.0466]
- **Cap 2 (inlet)** : 12 faces, aire=0.0024, centre=[0.3101, 0.3362, 0.0353]
- **Inlet** : **12 faces**
- **Outlet** : **50 faces**
- Méthode : région croissante sur normales + adjacence

### P11 — Patches OpenFOAM
- **INLET** : **376 faces**
- **OUTLET** : **2512 faces**
- **WALL** : 7705 faces
- Méthode : lecture des patches existants (pas de détection)

### P13 — Graphe topologique
- Lit les patches OpenFOAM existants
- **INLET** : 376 faces, **OUTLET** : 2512 faces
- Pas de détection propre

### P18 — Convention s_min/s_max
- Lit les patches OpenFOAM existants
- **INLET** : 376 faces, **OUTLET** : 2512 faces
- s_min = -0.015209, s_max = 0.021388

### P23 — Tests prioritaires
- Même détection que P04
- **Inlet** : **12 faces**
- **Outlet** : **50 faces**

### P24 — Conclusion
- Même détection que P04
- **Inlet** : **12 faces**
- **Outlet** : **50 faces**

## Comparaison des méthodes

### Par nombre de faces inlet/outlet
| Méthode | # Inlet | # Outlet | Total | Type |
|---------|---------|----------|-------|------|
| P02 | 0 | 0 | 0 | Extrémités centerline |
| P04 | 12 | 50 | 62 | Région croissante |
| P11 | 376 | 2512 | 2888 | Patches existants |
| P13 | 376 | 2512 | 2888 | Patches existants |
| P18 | 376 | 2512 | 2888 | Patches existants |
| P23 | 12 | 50 | 62 | Région croissante |
| P24 | 12 | 50 | 62 | Région croissante |

### Analyse
- **P04, P23, P24** sont cohérents : mêmes caps (12 inlet, 50 outlet)
- **P11, P13, P18** sont cohérents : mêmes patches existants (376 inlet, 2512 outlet)
- **P02** ne donne pas de faces, seulement des extrémités

### Les deux groupes ne sélectionnent PAS les mêmes zones
- Groupe 1 (P04/P23/P24) : sélectionne uniquement les caps plans détectés
- Groupe 2 (P11/P13/P18) : lit tous les patches OpenFOAM existants

## Conclusion
- Les méthodes **P04, P23, P24** forment un groupe cohérent avec une sélection précise (12 + 50 faces).
- Les méthodes **P11, P13, P18** forment un autre groupe cohérent basé sur les patches existants (376 + 2512 faces).
- Ces deux groupes ne sélectionnent pas les mêmes zones.
- Pour une détection automatique fiable, privilégier le groupe P04/P23/P24.

## Fichiers produits
- `method_images_true/` — visualisations corrigées avec les bons comptages
- `suivi_detection_inlet_outlet.md` — ce fichier
