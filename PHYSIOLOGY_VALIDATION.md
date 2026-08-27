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

## OpenFOAM 13

OpenFOAM 13 a été installé depuis le dépôt officiel pour Ubuntu 24.04 et chargé depuis `/opt/openfoam13/etc/bashrc`. Les deux cas de référence existants du dépôt — convection naturelle `buoyantCavity` et cas CHT transitoire `coolingSphere` — ont été exécutés avec succès. Cette validation confirme l’installation du moteur CFD, mais ne constitue pas une validation expérimentale humaine du modèle thermophysiologique.
