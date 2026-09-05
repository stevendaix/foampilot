# Règles de dépendances

## Graphe autorisé

```text
workflows ──► capabilities ──► bibliothèques externes
workflows ──► openfoam ──► backend ──► installation OpenFOAM
workflows ──► openfoam ──► extensions
```

Le core technique ne dépend jamais des workflows, exemples, tutoriels, validations ou extensions métier. Les workflows peuvent composer le core et déclarer leurs extensions. Le backend peut charger une extension, mais une extension ne doit pas importer un workflow.

## Interdictions architecturales

Les règles suivantes sont bloquantes :

| Import ou dépendance | Décision |
|---|---|
| `core → workflow` | Interdit |
| `core → extension spécifique` | Interdit |
| `core → examples/tutorials/validation` | Interdit |
| `medical → urban` | Interdit |
| `marine → medical` | Interdit |
| `workflow → core` | Autorisé |
| `workflow → openfoam` | Autorisé |
| `workflow → extension déclarée` | Autorisé |
| `core → VMTK` | Autorisé si l’API reste générique |

## Extensions

Une extension doit déclarer son nom, ses dépendances, ses capacités, sa compatibilité OpenFOAM et ses étapes de build. La structure cible est :

```text
extensions/openfoam/coupling/native-fsi/
├── extension.yaml
├── src/
├── Make/
├── tests/
└── README.md
```

Le nom `foundation13` ne doit pas devenir l’identité fonctionnelle de l’extension. La compatibilité Foundation 13 est une métadonnée ou un répertoire de patch lorsque la différence technique le justifie.
