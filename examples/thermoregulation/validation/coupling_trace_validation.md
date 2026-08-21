# Traçabilité temporelle du couplage JOS-3/OpenFOAM

## Objectif

Le protocole natif `externalCoupledTemperature` conserve son format strict : `data.out` contient `area`, `T`, `qDot` et `htc`, tandis que `data.in` contient `T_surface`, `snGrad` et `valueFraction`. Les métadonnées temporelles et les valeurs agrégées sont donc écrites dans deux fichiers latéraux : `coupling_trace.csv` et `coupling_zone_trace.csv`.

## Colonnes globales

`coupling_trace.csv` contient une ligne par échange. Les temps sont en secondes physiques ; les températures sont en °C dans le journal Python ; les températures du protocole OpenFOAM restent en K dans `data.in/data.out` ; les HTC sont en W/m²/K ; les flux surfaciques sont en W/m² ; les puissances intégrées sont en W ; les aires sont en m².

| Colonne | Signification |
|---|---|
| `exchange_id` | Identifiant monotone de l’échange |
| `time_cfd_s` | Temps CFD après le pas terminé |
| `time_jos3_s` | Temps physiologique après l’appel JOS-3 |
| `deltaT_cfd_s` | Pas lu dans `system/controlDict` |
| `dtime_jos3_s` | Pas transmis à JOS-3 |
| `n_faces` | Nombre de faces du patch humain |
| `area_total_m2` | Aire totale du patch humain |
| `h_area_mean_W_m2_K` | HTC moyen pondéré par l’aire |
| `Ta_area_mean_C` | Température d’air moyenne pondérée par l’aire |
| `qDot_integral_W` | Intégrale de `qDot × area` sur les faces |
| `Ttarget_*` | Température JOS-3 avant relaxation |
| `Treturn_*` | Température effectivement retournée à OpenFOAM |
| `environment_power_W` | Puissance environnementale calculée par le réseau distribué |
| `body_power_W` | Puissance d’ancrage échangée avec les nœuds physiologiques |
| `time_error_s` | `time_cfd_s - time_jos3_s` |

## Colonnes par zone

`coupling_zone_trace.csv` contient 17 lignes par échange, une par zone JOS-3. Il donne l’aire de zone, la température d’air pondérée, le HTC pondéré, la puissance d’ancrage et la température de surface retournée pondérée par l’aire.

## Test exécuté

Le test a été réalisé sur le cas body-only avec plafond réellement ouvert, Boussinesq et `deltaT = 0,05 s`. Il comporte 4 échanges jusqu’à `0,20 s`, 9 418 faces et une aire totale de `1,548226251089 m²`.

| Échange | Temps CFD (s) | Temps JOS-3 (s) | Erreur temporelle (s) | HTC moyen (W/m²/K) | Température retournée moyenne (°C) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,05 | 0,05 | 0 | 3,8131 | 33,9999 |
| 2 | 0,10 | 0,10 | 0 | 3,8131 | 33,9998 |
| 3 | 0,15 | 0,15 | 0 | 3,8131 | 33,9998 |
| 4 | 0,20 | 0,20 | 0 | 3,8131 | 33,9997 |

Le contrôle automatique vérifie donc à chaque ligne :

\[
\Delta t_{JOS3}=\Delta t_{CFD}=0,05\;\mathrm{s},
\qquad
 t_{CFD}=t_{JOS3},
\qquad
 N_f=9418.
\]

Les fichiers de communication natifs ne sont pas enrichis de colonnes supplémentaires, ce qui évite de modifier le parseur OpenFOAM. La traçabilité est reproductible en relançant le pilote ; les deux CSV sont générés dans le répertoire du cas et sont ignorés par Git comme sorties de calcul.
