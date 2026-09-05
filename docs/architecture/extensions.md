# Extensions

Les extensions ajoutent des composants techniques au backend OpenFOAM : couplages, modèles, solvers, conditions limites, function objects et bibliothèques.

## Classification cible

```text
extensions/openfoam/
├── coupling/
│   ├── native-fsi/
│   ├── yade/
│   └── cantera/
├── models/
├── solvers/
├── boundary_conditions/
├── function_objects/
└── libraries/
```

Chaque extension doit documenter son API, son build, ses dépendances, ses tests et ses cibles de compatibilité. Les sources C++ ne doivent pas être mélangées au core Python ni à un workflow métier.

## Compatibilité par extension

Une extension peut porter des adaptations internes lorsque la différence entre distributions ou versions est technique et localisée. La structure recommandée est :

```text
native-fsi/
├── src/
├── compatibility/
│   ├── foundation/v13/
│   ├── foundation/v14/
│   └── openfoam-com/
└── tests/
```

La duplication complète d’une extension pour chaque version est interdite sauf justification exceptionnelle et documentée.

## Suppression progressive de `foundation13`

Le dossier ou namespace `foundation13` ne doit pas être supprimé dans cette PR de baseline. Il sera traité lors d’une PR de migration C++ après identification de chaque élément : capability, backend, extension, patch ou validation.
