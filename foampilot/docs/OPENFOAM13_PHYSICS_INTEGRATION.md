# Intégrations physiques OpenFOAM 13

`foampilot.openfoam13.PhysicsConfig` fournit une frontière stable entre Foampilot et les cinq dépôts externes demandés. L’intégration est **déclarative et non destructive** : elle écrit un manifest de provenance, des propriétés urbaines, un dictionnaire AMR et des métadonnées de couplage sans écraser les fichiers de cas existants.

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

Le modèle de turbulence d’apprentissage automatique est volontairement **opt-in** et doit être associé à un module activé compatible avec le fournisseur OpenFOAM choisi. Pour l’AMR, le champ physique source (`curl(U)`, `grad(p)` ou un champ utilisateur) est distinct du champ `refVal` consommé par `dynamicRefineFvMesh`; cette séparation évite de générer un dictionnaire OpenFOAM invalide.

Le contrôle préliminaire vérifie également la présence explicite de `nu` dans `constant/transportProperties`, les répertoires standards du cas et les entrées essentielles d’AMR. La compilation des bibliothèques C++ upstream reste une étape d’installation contrôlée, car les dépôts audités ciblent principalement OpenFOAM ESI ou des versions antérieures et ne peuvent pas être déclarés Foundation 13 ABI-compatibles sans portage compilé.
