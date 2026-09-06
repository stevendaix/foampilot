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

Les sources portées sont versionnées sous `third_party/openfoam13/ported/`. Le script `third_party/openfoam13/build_ports.sh` compile tous les ports avec `wmake` après chargement de l’environnement OpenFOAM Foundation 13 et s’arrête à la première erreur. Le dépôt ne contient pas de binaires générés.

| Composant | Dépôt d’origine | Type | Adaptations Foundation 13 | Validation |
|---|---|---|---|---|
| `ZYturbulentInlet` | `ZhangYanTJU/boundaryConditions` | bibliothèque de condition limite | `Random` remplacé par `randomGenerator`, copie interdite supprimée, `autoMap/rmap` remplacés par `map/reset` | `wmake libso` réussi |
| `turbulentInletTable` | `ZhangYanTJU/boundaryConditions` | bibliothèque de condition limite générique | mêmes adaptations, `Function1::New` avec unités, `timeOutputValue()` remplacé par `Time::value()` | `wmake libso` réussi |
| `calculateNut` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | `fvCFD.H` remplacé par les includes modulaires, `userTimeName()` et structure d’application Foundation 13 | `wmake` et `-help` réussis |
| `calculateGamma` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | includes modulaires, `dimKinematicViscosity`, `fvcFlux.H`, `userTimeName()` | `wmake` et `-help` réussis |
| `calculateRFV` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | includes modulaires, viscosité dimensionnée Foundation 13, `userTimeName()` | `wmake` et `-help` réussis |
| `calculateRFVperp` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | includes modulaires, `fvcFlux.H`, viscosité dimensionnée et noms temporels Foundation 13 | `wmake` et `-help` réussis |
| `calculateRperp` | `mthsmcd/MachineLearningTurbulenceModels` | utilitaire ML | includes modulaires et noms temporels Foundation 13 | `wmake` et `-help` réussis |

## Modules conservés comme workflows

`urbanMicroclimateFoam-tutorials`, `adaptive-mesh-refinement` et `PythonFOAM` sont intégrés comme provenance et workflows dans le manifest, mais leurs applications complètes ne sont pas copiées sous forme de bibliothèque C++ dans cette PR. Les tutoriels urbains et PythonFOAM ne constituent pas des bibliothèques Foundation 13 autonomes, tandis que l’AMR dépend du solveur et des conventions de maillage du cas cible. Le générateur Foampilot produit donc un `dynamicMeshDict` Foundation 13 avec le champ technique `refVal`, en conservant séparément le champ physique source (`curl(U)`, `grad(p)`, `grad(T)` ou champ utilisateur).

## Vérifications exécutées

Les tests Python ciblés sont exécutés avec `PYTHONPATH=foampilot/src python3 -m pytest -q foampilot/test/openfoam13/test_physics.py`. Ils couvrent le catalogue des cinq dépôts, l’opt-in ML, l’écriture non destructive, le contrôle explicite de `nu` et l’absence des API C++ obsolètes dans les sources portées. La dernière exécution a produit **5 tests réussis**. Les sept composants portés ont également été recompilés depuis l’arbre Foampilot avec OpenFOAM Foundation 13 ; les cinq utilitaires ML répondent à `-help`.

Un cas Foundation 13 minimal à huit cellules a été construit avec `blockMesh`, contrôlé avec `checkMesh`, puis exécuté avec `calculateNut`, `calculateRperp`, `calculateRFV`, `calculateRFVperp` et `calculateGamma`. Les cinq journaux se terminent par `End`, les champs `nut`, `Rperp`, `t` et `tStar` sont produits et aucun `nan`, `inf` ou message fatal n’est détecté. La formule de `calculateNut` protège en outre le cas physiquement valide où le taux de déformation est nul, en évitant une division par zéro.

Cette validation numérique ne doit pas être confondue avec une simple réussite de linkage : les checks Foampilot imposent notamment la présence explicite de `nu` et des fichiers standard du cas.
