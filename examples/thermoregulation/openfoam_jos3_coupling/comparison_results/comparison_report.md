# Rapport de comparaison JOS-3

L'exemple officiel `JOS-3/example/example_v2.py` a été repris sans modifier ses paramètres : morphologie, conditions 28/30 °C, phase à 20 °C, ventilation variable, vêtements, posture et pas de 60 s.

## Critères

Les métriques sont calculées sur les séries temporelles de température moyenne cutanée, tête, thorax, pied gauche et température centrale du thorax. L'écart est évalué sur les mêmes instants : maximum absolu, moyenne absolue, RMSE et écart final.

| Sortie | max absolu [°C] | moyenne absolue [°C] | RMSE [°C] | final [°C] |
|---|---:|---:|---:|---:|
| TskMean | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| TskHead | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| TskChest | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| TskLFoot | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| TcrChest | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| coupled_zero_flux_TskMean | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## Validation du chemin couplé

Pour le cas à 17 points, avec `h=10 W/m²/K` et `T_surface=T_air` dans chaque phase, le flux CFD est nul. L'écart RMSE entre la copie embarquée et le chemin couplé est de 0.000e+00 °C sur les 121 instants comparables.

Les données complètes sont dans `official_example_comparison.csv`, les métriques dans `comparison_metrics.csv` et la figure dans `official_example_comparison.png`.
