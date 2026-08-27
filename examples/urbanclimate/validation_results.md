# Validation d’exécution des six cas urbanclimate

## Environnement et méthode

Les six cas ont été régénérés par `examples/urbanclimate/run.py` puis exécutés sous **OpenFOAM Foundation 13** avec leur script `./Allrun`. Les exécutions ont été lancées indépendamment et leurs journaux complets sont conservés dans `/tmp/urbanclimate_runs/`.

## Résultats

| Cas | Code retour | Fin normale `End` | Temps final observé | Erreur fatale |
|---|---:|---:|---:|---|
| `streetCanyon_CFD` | 0 | Oui | `0.02` | Aucune |
| `streetCanyon_CFDHAM` | 0 | Oui | `0.0199299` | Aucune |
| `streetCanyon_CFDHAM_grass` | 0 | Oui | `0.0199299` | Aucune |
| `streetCanyon_CFDHAM_veg` | 0 | Oui | `0.0199299` | Aucune |
| `windAroundBuildings_CFDHAM` | 0 | Oui | `0.0199299` | Aucune |
| `windAroundBuildings_CFDHAM_veg` | 0 | Oui | `0.0199299` | Aucune |

Les profils végétalisés ont exécuté les étapes supplémentaires `faceAgglomerate`, `calcLAI`, `viewFactorsGen` et `solarRayTracingGen` avant le solveur. Les journaux ne contiennent ni `FOAM FATAL`, ni segmentation fault, ni exception flottante, ni erreur de commande.

## Interprétation

Le résultat confirme que les six workflows sont **fonctionnels au niveau logiciel et exécutable** : génération native, maillage, prétraitement physique, initialisation multi-région et progression du solveur jusqu’à la fin du temps de calcul configuré.

Cette campagne utilise l’horizon temporel court défini par les exemples. Elle valide donc la compatibilité et le chemin d’exécution, mais ne constitue pas encore une validation scientifique complète de la convergence ou de la pertinence des résultats physiques. Les résidus des équations solides peuvent atteindre la limite d’itérations configurée ; une campagne scientifique devra prolonger `endTime`, contrôler `checkMesh`, comparer les bilans et confronter les températures, vitesses et flux aux références originales.
