# Architecture FoamPilot v3

## Principe directeur

FoamPilot v3 sépare explicitement le **core technique**, les **workflows métier**, le **backend d’exécution OpenFOAM** et les **extensions C++ ou bibliothèques externes**.

> Les capabilities sont réutilisables. Les workflows composent les capabilities. Les extensions étendent le backend d’exécution. Les versions OpenFOAM sont des cibles de compatibilité runtime.

Une version ou une distribution OpenFOAM n’est donc ni une capability, ni un workflow, ni une catégorie racine du dépôt. Foundation 13, Foundation 14 et OpenFOAM.com doivent apparaître dans les métadonnées du backend, les règles de compatibilité et les patches lorsqu’ils sont réellement version-spécifiques.

## Modèle cible

```text
FoamPilot
├── Technical core
│   ├── case
│   ├── geometry
│   ├── meshing
│   ├── boundaries
│   ├── dictionaries
│   ├── postprocessing
│   ├── reporting
│   └── execution
├── OpenFOAM backend
│   ├── environment
│   ├── runner
│   ├── registry
│   └── extensions interface
├── Workflows
│   ├── medical
│   ├── marine
│   ├── urban
│   ├── energy
│   └── multiphysics
└── Extensions
    └── OpenFOAM coupling, models, solvers and libraries
```

Cette structure est une **cible de migration**, pas une instruction de déplacement massif immédiat. La première PR v3 documente l’existant et ajoute les règles qui permettront des migrations petites, réversibles et testables.

## Dépôt observé

La branche de baseline contient déjà plusieurs générations d’organisation : `foampilot/src/foampilot/` pour la bibliothèque Python, `examples/` pour les démonstrations et workflows historiques, `openfoam13/` pour des cas et extensions Foundation 13, `third_party/` pour des couplages et bibliothèques vendorisés, et `validation/` pour des validations scientifiques ou numériques.

L’existence de `openfoam13/` ne signifie pas que la version doit rester une catégorie architecturale. La migration devra classer chaque élément selon sa fonction réelle : capability, backend, extension, patch, workflow, cas d’exemple ou validation.

## Règles de non-régression

La migration ne supprimera aucun ancien dossier avant que la nouvelle API, les tests, les workflows, la documentation et la CI correspondante soient disponibles. Des shims pourront maintenir les imports publics pendant la transition, avec `DeprecationWarning` et une date de suppression annoncée.
