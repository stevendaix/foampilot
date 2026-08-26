# Intégrations physiques OpenFOAM 13

`foampilot.openfoam13.PhysicsConfig` fournit une frontière stable entre Foampilot et les cinq dépôts externes demandés. L’intégration reste **déclarative et non destructive** : elle écrit un manifest de provenance, des propriétés urbaines, un dictionnaire AMR et des métadonnées de couplage sans écraser les fichiers de cas existants.

```python
from foampilot.openfoam13 import PhysicsConfig, check_openfoam13_case

config = PhysicsConfig(
    boundary_conditions={"inlet": {"type": "turbulentInletTable"}},
    turbulence={"model": "gammaRST"},
    urban={"referenceHeight": 10.0, "roughnessLength": 0.5},
    adaptive_mesh={"sourceField": "curl(U)", "lowerRefinementLevel": 0.1},
    pythonfoam={"enabled": False},
)
config.write_support_files(case_directory)
errors = check_openfoam13_case(case_directory)
if errors:
    raise RuntimeError(errors)
```

## Portages Foundation 13 compilés

Les sources portées sont versionnées sous `third_party/openfoam13/ported/`. Elles sont compilables avec `wmake` après chargement de `/home/ubuntu/OpenFOAM-13/etc/bashrc` et ne contiennent pas de binaires générés.

| Composant | Dépôt d’origine | Type | Adaptations Foundation 13 | Validation |
|---|---|---|---|---|
| `ZYturbulentInlet` | `ZhangYanTJU/boundaryConditions` | bibliothèque de condition limite | `Random` remplacé par `randomGenerator`, copie interdite supprimée, `autoMap/rmap` remplacés par `map/reset` | `wmake libso` réussi |
| `turbulentInletTable` | `ZhangYanTJU/boundaryConditions` | bibliothèque de condition limite générique | mêmes adaptations, `Function1::New` avec unités, `timeOutputValue()` remplacé par `Time::value()` | `wmake libso` réussi |
| `calculateNut` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | `fvCFD.H` remplacé par les includes modulaires, `userTimeName()` et structure d’application Foundation 13 | `wmake` et `-help` réussis |
| `calculateGamma` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | includes modulaires, `dimKinematicViscosity`, `fvcFlux.H`, `userTimeName()` | `wmake` et `-help` réussis |

## Modules conservés comme workflows

`urbanMicroclimateFoam-tutorials`, `adaptive-mesh-refinement` et `PythonFOAM` sont intégrés comme provenance et workflows dans le manifest, mais leurs applications complètes ne sont pas copiées sous forme de bibliothèque C++ dans cette PR. Les tutoriels urbains et PythonFOAM ne constituent pas des bibliothèques Foundation 13 autonomes, tandis que l’AMR dépend du solveur et des conventions de maillage du cas cible. Le générateur Foampilot produit donc un `dynamicMeshDict` Foundation 13 avec le champ technique `refVal`, en conservant séparément le champ physique source (`curl(U)`, `grad(p)`, `grad(T)` ou champ utilisateur).

## Vérifications exécutées

Les tests Python ciblés sont exécutés avec `PYTHONPATH=foampilot/src python3 -m pytest -q foampilot/test/openfoam13/test_physics.py`. Ils couvrent le catalogue des cinq dépôts, l’opt-in ML, l’écriture non destructive, le contrôle explicite de `nu` et l’absence des API C++ obsolètes dans les sources portées. La dernière exécution a produit **5 tests réussis**. Les quatre composants portés ont également été recompilés depuis l’arbre Foampilot avec OpenFOAM Foundation 13 ; les binaires `calculateNut` et `calculateGamma` répondent à `-help`.

La validation numérique complète d’un cas nécessite ensuite une maille, les champs physiques requis (`U`, `R`, `nut`) et les utilitaires OpenFOAM 13 du cas cible. Elle ne doit pas être confondue avec une simple réussite de linkage : les checks Foampilot imposent notamment la présence explicite de `nu` et des fichiers standard du cas.
