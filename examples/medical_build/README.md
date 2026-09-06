# `medical_build`

Ce répertoire contient les exemples, contrats, fixtures et cas OpenFOAM reproductibles du module médical FoamPilot. Les rapports exhaustifs, visualisations et géométries produites par les campagnes locales sont volontairement exclus du dépôt ; les scripts qui les génèrent restent versionnés.

## Éléments conservés

Le code réutilisable se trouve dans `foampilot/src/foampilot/geometry/medical_build`. Les tests sont sous `foampilot/tests/geometry/medical_build`. Les cinq fixtures VMTK sous `foampilot/test/vmtk_test_data` constituent les données de test versionnées et sont décrites dans `docs/medical_build/VMTK_TEST_DATA_SOURCES.md`.

Les cas `case_snappy_aorta*` et `openfoam_case` conservent leurs dictionnaires, champs initiaux et surfaces d’entrée nécessaires. Le sous-répertoire `case_complex/openfoam/system` conserve les dictionnaires de génération ; les sorties d’analyse et d’export sont régénérables et ne sont pas versionnées.

## Export NPZ des sections

Les contours de stations peuvent avoir des nombres de points différents. `medical_build_end_to_end.py` écrit alors des tableaux numériques paddés avec `NaN` et le tableau `section_lengths`. Pour une station `i`, seuls les éléments `[:section_lengths[i]]` sont valides. Cette représentation ne recourt pas à des tableaux objet picklés.

## Cas OpenFOAM

Depuis `examples/medical_build/openfoam_case`, charger un environnement Foundation 13 avec `FOAM_BASHRC` ou `WM_PROJECT_DIR`, puis lancer `./Allrun`. Si la surface multirégion STL versionnée est absente, fournir une surface VTP portant les identifiants `PatchId` via `MEDICAL_SURFACE_VTP`. Le runner vérifie explicitement l’environnement et les entrées ; il ne masque pas les erreurs de chargement.

Le script end-to-end avec `--openfoam` ne lance pas le solveur OpenFOAM. Il indique seulement qu’un cas externe doit être exécuté par le runner dédié. Cette distinction est intentionnelle afin d’éviter de présenter un export déclaratif comme une simulation validée.

## Validation locale

```bash
export PYTHONPATH="$PWD/foampilot/src"
pytest -q foampilot/tests/geometry/medical_build
python3 -m compileall -q examples/medical_build
bash -n examples/medical_build/openfoam_case/Allrun \
  examples/medical_build/openfoam_case/Allrun_solver
```

La validation de la chaîne CFD complète doit être effectuée dans une installation OpenFOAM Foundation 13 disponible et enregistrée avec ses versions, ses entrées et ses métriques attendues.
