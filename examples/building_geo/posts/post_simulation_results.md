# Validation du cas VoxCity : checkMesh et foamRun

## Maillage
- **Points** : 2688
- **Cellules** : 11645 tétraèdres
- **Patches** : 7 (inlet, outlet, top, ground, side_left, side_right, buildings)
- **Non-orthogonalité max** : ~53 %
- **Skewness max** : ~1.03
- `checkMesh` : OK

## Résultats de simulation
- `foamRun -solver incompressibleFluid` lancé sans erreur.
- Résidus faibles sur `U`, `p`, `k`, `epsilon`.
- Erreur de continuité cumulative très faible (~1.5e-5).
- Champs écrits jusqu'à `t = 2000 s`.

## Outillage
- Tout est piloté depuis `voxcity_vector_example.py`.
- `Solver.run_simulation()` remplace l'appel manuel à `foamRun`.

## Ce qui reste à faire
- Visualisation des champs (PyVista off-screen à fiabiliser).
- Extraction de profils et de coefficients de pression sur les façades.
- Comparaison avec des données expérimentales si disponibles.
