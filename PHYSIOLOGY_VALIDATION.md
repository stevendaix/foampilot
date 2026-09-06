# Validation de la branche physiology

## Résultats reproductibles

| Vérification | Commande | Résultat |
|---|---|---|
| Tests physiology et couplage | `PYTHONPATH=foampilot/src pytest -q foampilot/test/test_physiology.py foampilot/test/test_external_coupled.py` | **11 passed** |
| Compilation Python | `python3 -m compileall -q foampilot/src/foampilot/physiology foampilot/src/foampilot/postprocess` | **OK** |
| Contrôle whitespace Git | `git diff --check` | **OK** |
| Comparaison JOS3 officielle | `PYTHONPATH=foampilot/src python3 examples/thermoregulation/openfoam_jos3_coupling/compare_official_example.py` | **OK**, RMSE de `5.8e-14` à `9.7e-14 °C` sur les indicateurs suivis |
| Références OpenFOAM 13 | `source /opt/openfoam13/etc/bashrc && python3 examples/thermoregulation/validation/run_openfoam13_references.py` | **OK** pour `buoyantCavity` et `coolingSphere` |
| Installation OpenFOAM | `foamRun -help` avec `WM_PROJECT_VERSION=13` | **OK** |

## Améliorations démontrées

La suite ajoutée vérifie le rejet des mappings non physiques, la validation des dimensions et valeurs finies, la conversion Kelvin–Celsius et watt par mètre carré–kelvin via `utilities/manageunits.py`, la conservation de la puissance lors de l’agrégation des faces CFD, la stabilité de l’évaporation lorsque la capacité est nulle, ainsi que la validation des postures et des pas de temps. La comparaison contre la référence officielle montre que les corrections d’interface et de robustesse ne modifient pas le résultat nominal JOS3 dans le scénario de référence, à l’erreur d’arrondi près.

## Limite de la suite globale

La commande `pytest -q` ne constitue pas actuellement une suite globale fiable pour ce dépôt : plusieurs fichiers de test sont exécutables au moment de leur import et `test_cfd_methods.py` appelle directement `argparse.parse_args()`. Cette situation provoque une erreur de collecte indépendante de la branche physiology. La pull request conserve donc les tests ciblés comme porte de validation déterministe et signale ce problème de qualité de test séparément.

## Difficultés rencontrées et statut

| Difficulté | Impact | Statut et traitement |
|---|---|---|
| Le package `foampilot` charge de nombreuses intégrations optionnelles dès l’import, ce qui bloquait les tests ciblés sur `build123d` puis `jupyter_cadquery`. | Les tests de physiology ne pouvaient pas être collectés dans un environnement minimal. | **Contournée pour cette PR** par un chargement isolé de l’arbre physiology dans les tests et le script de comparaison. Une refonte générale des imports du package reste souhaitable mais dépasse ce changement. |
| `manageunits.py` expose surtout une API scalaire, alors que les champs CFD sont des tableaux nodaux. | Une utilisation directe risquait de perdre les conversions ou de convertir implicitement des quantités incompatibles. | **Résolue** par `physiology/units.py`, qui réutilise le registre Pint de `manageunits.py` et normalise les scalaires, tableaux, quantités Pint et `ValueWithUnit`. |
| La signature réellement utilisée par `JOS3.Wet` ne correspondait pas à la signature complète de `evaporation`. | Risque d’associer des positions aux mauvais paramètres et de produire des résultats physiologiques invalides. | **Résolue** par un appel nommé et explicite avec hauteur, masse, équation BSA et âge. |
| Le choix du coefficient convectif PMV contenait une affectation erronée de `hcf` au lieu de sélectionner le maximum entre convection forcée et naturelle. | Biais systématique possible dans le calcul PMV et les échanges secs. | **Résolue** par `hc = max(hcf, hcn)` et couverte par la comparaison JOS3. |
| Les essais OpenFOAM 13 nécessitaient un dépôt APT dédié et une initialisation de l’environnement `/opt/openfoam13/etc/bashrc`; l’installation a aussi produit des messages de timeout pendant APT. | Risque de conclure à tort à une installation incomplète. | **Résolue** : `foamRun -help`, `buoyantCavity` et `coolingSphere` terminent correctement après chargement de l’environnement. |
| La référence `jos3` n’était pas disponible dans l’environnement initial. | La comparaison scientifique ne pouvait pas être exécutée. | **Résolue** par installation de la référence officielle avant la comparaison, avec des écarts de l’ordre de `10⁻¹³ °C`. |
| La suite globale `pytest -q` collecte des scripts qui exécutent `argparse.parse_args()` au moment de l’import, notamment `test_cfd_methods.py`. | La collecte globale échoue avant l’exécution de tous les tests et produit des erreurs hors du périmètre physiology. | **Non résolue dans cette PR**, documentée comme dette de test séparée. Les tests ciblés restent verts et déterministes. |
| Le rayonnement solaire dédié décrit dans la théorie JOS-3 n’est pas assemblé explicitement dans `_run`; le code accepte une injection générique `ex_q`. | La validation solaire ne peut pas être revendiquée comme complète. | **Non résolue scientifiquement** : la documentation le signale afin d’éviter une revendication excessive. Une implémentation et une validation dédiées devront faire l’objet d’une PR séparée. |

## OpenFOAM 13

OpenFOAM 13 a été installé depuis le dépôt officiel pour Ubuntu 24.04 et chargé depuis `/opt/openfoam13/etc/bashrc`. Les deux cas de référence existants du dépôt — convection naturelle `buoyantCavity` et cas CHT transitoire `coolingSphere` — ont été exécutés avec succès. Cette validation confirme l’installation du moteur CFD, mais ne constitue pas une validation expérimentale humaine du modèle thermophysiologique.
