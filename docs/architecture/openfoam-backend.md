# Backend OpenFOAM

Le répertoire `openfoam/` cible l’interface FoamPilot ↔ OpenFOAM, et non une version particulière. Il doit encapsuler la détection de l’environnement et exposer des capacités d’exécution stables aux workflows.

## Métadonnées runtime

Le backend doit pouvoir détecter ou recevoir explicitement :

| Métadonnée | Exemple |
|---|---|
| Distribution | Foundation, OpenFOAM.com ou autre distribution supportée |
| Version | 13, 14 ou version distribuée |
| Installation | `FOAM_BASHRC`, `WM_PROJECT_DIR` et chemins dérivés |
| Compilateur | GCC/Clang et ABI pertinente |
| MPI | implémentation et options disponibles |
| Bibliothèques | bibliothèques chargées et symboles vérifiables |
| Solvers | applications disponibles dans l’environnement |

Conceptuellement, `OpenFOAM.detect()` renvoie des métadonnées runtime. Ces données ne doivent pas contaminer les API métier ni déterminer l’arborescence des workflows.

## Interface cible

```text
openfoam/
├── backend/       # distribution, version, installation et capacités
├── environment/   # chargement et validation de l’environnement
├── runner/        # commandes, logs, MPI et codes retour
├── registry/      # solvers, extensions et capabilities déclarés
└── extensions/    # point d’intégration technique
```

## Compatibilité

Les différences spécifiques à une version doivent être limitées aux adaptateurs, compatibilités ou patches réellement nécessaires. Une extension ne doit pas être dupliquée intégralement sous `foundation13/`, `foundation14/` et `openfoam-com/`.

Un patch attaché intrinsèquement à une version appartient à `patches/openfoam/<distribution>/<version>/`. Un adaptateur de détection appartient au backend. Un composant C++ appartient aux extensions.
