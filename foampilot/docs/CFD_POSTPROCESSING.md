# Post-traitement CFD orienté ingénieur

Le module `foampilot.postprocess.monitoring` fournit des sondes et des suivis temporels indépendants du solveur. Il s’appuie sur `FoamPostProcessing` et les fichiers VTK déjà produits par OpenFOAM.

```python
from foampilot.postprocess import CFDMonitor, FoamPostProcessing

post = FoamPostProcessing("cases/pitzDaily")
monitor = CFDMonitor(post)

# Statistiques par temps pour la pression dans le domaine.
pressure = monitor.track_region("p", region="cell")
monitor.export_csv(pressure, "postProcessing/pressure_statistics.csv")

# Suivi de la norme de vitesse au point le plus proche.
velocity_probe = monitor.track_point(
    point=(0.10, 0.02, 0.0),
    field="U",
    magnitude=True,
)
monitor.export_csv(velocity_probe, "postProcessing/velocity_probe.csv")

# Résumé JSON du dernier temps disponible.
summary = monitor.summary(
    fields=["p", "U"],
    region="cell",
    magnitudes=["U"],
)
monitor.export_json(summary, "postProcessing/case_summary.json")
```

`track_region` retourne, pour chaque temps, la moyenne, l’écart-type, les valeurs minimale et maximale, les percentiles 5/50/95 et la RMS. Pour un champ vectoriel comme `U`, le paramètre `magnitude=True` calcule la norme avant les statistiques.

`track_point` utilise la sonde de maillage la plus proche et conserve les coordonnées demandées dans la table de sortie. Cette méthode est adaptée au suivi de `p`, de `T`, de `U` ou de toute grandeur disponible en `point_data`.

Les sorties restent volontairement descriptives : elles ne remplacent pas un bilan de masse ou d’énergie intégré sur une frontière. Pour une validation physique, il est recommandé de compléter les séries par les débits massiques, les flux thermiques et les résidus issus de `functionObjects` OpenFOAM, puis de comparer les bilans avec des tolérances explicites.

## Recommandations d’intégration

Les suivis devraient être exécutés après chaque campagne de calcul et archivés avec le nom du cas, le solveur, la version OpenFOAM, le pas de temps et les paramètres physiques. Les CSV sont adaptés à l’analyse exploratoire et les JSON aux rapports automatisés. En production, il est préférable de figer la liste des champs, l’association point/cellule et les unités attendues dans la configuration du cas.
