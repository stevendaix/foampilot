# Workflows

Un workflow est une orchestration fine de capabilities. Il assemble des entrées, de la géométrie, du maillage, des dictionnaires, un backend d’exécution et du post-traitement, sans devenir une bibliothèque technique générale.

## Workflows identifiés

| Domaine | Emplacements actuels principaux | Destination conceptuelle |
|---|---|---|
| Medical | `examples/medical_build/`, `examples/coa/`, thermorégulation | `workflows/medical/` |
| Marine | `examples/marine_config/`, `openfoam13/FoamPilotCases/` et validations marines | `workflows/marine/` |
| Urban | `examples/urbanclimate/`, `examples/building_*` | `workflows/urban/` |
| Energy | turbines, CHT, Cantera et cas thermiques | `workflows/energy/` |
| Multiphysics | CHT, YADE, Cantera, FSI et couplages | `workflows/multiphysics/` |

Cette table est une cartographie initiale, pas une décision de déplacement. Chaque élément doit être classé par API publique, dépendances, version OpenFOAM, domaine, présence de C++ et réutilisabilité avant migration.

## Chaîne d’orchestration

Un workflow médical peut composer VMTK, CAD, Gmsh, Case, OpenFOAM et post-traitement. Un workflow marin peut réutiliser Geometry, Meshing, Case et le backend OpenFOAM, puis déclarer une extension FSI ou overset. Aucun workflow ne doit posséder ou dupliquer ces capabilities.

## Interdictions

Le core ne doit pas importer un workflow, un exemple, un tutoriel, une validation ou une extension spécifique. Les workflows ne doivent pas importer directement un autre domaine métier : `medical → urban` et `marine → medical` sont interdits.
