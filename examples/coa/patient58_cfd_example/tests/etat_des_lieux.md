# État des lieux — Détection inlet/outlet patient58

## Contexte
- Cas OpenFOAM : `patient58_cfd_example`
- Tests P01–P24 déjà exécutés avec images et résultats.
- Objectif : comparer les méthodes fiables et vérifier si elles sélectionnent les mêmes zones inlet/outlet.

## Surface du maillage
- **Total faces de bord** : **10 593**
- **INLET** : 376 faces
- **OUTLET** : 2512 faces
- **WALL** : 7705 faces

## Méthodes fiables étudiées
| Méthode | Type | Résultat |
|---------|------|----------|
| **P02** | Centerline PCA | 0 faces (extrémités centerline uniquement) |
| **P04** | Caps plans / région croissante | **12 faces inlet**, **50 faces outlet** |
| **P11** | Patches OpenFOAM existants | 376 faces inlet, 2512 faces outlet |
| **P13** | Graphe topologique | 376 faces inlet, 2512 faces outlet (lit les patches) |
| **P18** | Convention s_min/s_max | 376 faces inlet, 2512 faces outlet (lit les patches) |
| **P23** | Tests prioritaires | 12 faces inlet, 50 faces outlet (identique P04) |
| **P24** | Conclusion / pipeline | 12 faces inlet, 50 faces outlet (identique P04) |

## Constat important
- **P04, P23, P24** forment un groupe cohérent : détection automatique par région croissante → **12 inlet + 50 outlet = 62 faces**
- **P11, P13, P18** forment un autre groupe : lecture des patches OpenFOAM existants → **376 inlet + 2512 outlet = 2888 faces**
- Ces deux groupes ne sélectionnent **pas les mêmes zones**
- **P02** ne donne pas de faces, seulement 2 points centerline

## Visualisations
- Dossier `method_images_true/` : images corrigées avec les bons comptages
- Les images montrent clairement la différence entre les deux groupes
- **Attention** : certaines images précédentes étaient vides ou incorrectes à cause d'un mauvais mapping des face_id

## Calcul OpenFOAM
- Un calcul `simpleFoam` a été lancé avec les conditions originales
- **Résultat** : calcul terminé, mais pression nulle partout (`p_max = 0`)
- **Diagnostic** : le fichier `0/p` a été supprimé par erreur, puis recréé
- Le calcul a été relancé avec les conditions originales (INLET 376, OUTLET 2512)
- **Problème rencontré** : erreur dans `system/fvSolution` (keyword `smoother` non supporté dans cette version)
- **Solution partielle** : rétablissement de la configuration originale, puis modification minimale (`nCorrectors 1`, `nNonOrthogonalCorrectors 1`)

## Fichiers importants
- `tests/method_images_true/` — visualisations corrigées
- `tests/suivi_detection_inlet_outlet.md` — fichier de suivi
- `tests/run_p04_only.py` — script de détection P04
- `tests/analyze_results.py` — analyse des résultats OpenFOAM
- `log.simpleFoam` — log du calcul OpenFOAM

## Problèmes en cours
1. **fvSolution** : erreur de syntaxe sur le keyword `smoother` pour le solveur `p`
2. **0/p** : fichier recréé mais vérifier sa cohérence
3. **Boundary** : les caps P04 ne sont pas des blocs consécutifs dans le maillage OpenFOAM, donc on ne peut pas directement les utiliser comme patches sans restructurer le maillage

## Prochaines étapes recommandées
1. **Corriger fvSolution** : supprimer le block `smoother` pour le solveur `p` ou utiliser une syntaxe compatible OpenFOAM 13
2. **Vérifier 0/p** : s'assurer que les conditions initiales sont cohérentes
3. **Relancer simpleFoam** avec les fichiers corrigés
4. **Analyser les résultats** une fois le calcul convergé
5. **Optionnel** : si on veut utiliser P04 comme inlet/outlet, il faut créer un nouveau cas OpenFOam avec des patches dédiés, car les faces détectées ne sont pas consécutives dans le maillage actuel

## Conclusion
Les méthodes P04/P23/P24 détectent bien des caps plans cohérents (12 + 50 faces), mais ils ne correspondent pas aux patches OpenFOAM existants. Le calcul OpenFOAM avec les conditions originales est en cours de débogage à cause d'une erreur de configuration dans `fvSolution`.
