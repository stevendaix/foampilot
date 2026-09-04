# Compatibilité de version

Les versions OpenFOAM sont des cibles de compatibilité d’exécution. Elles ne doivent pas définir les capabilities ou les workflows.

| Niveau | Contenu version-dépendant autorisé |
|---|---|
| Backend | détection de distribution, version, installation et capacités |
| Extension | adaptateurs C++ localisés et tests de compatibilité |
| Patch | modifications intrinsèquement attachées à une version |
| Workflow | aucune logique de version, sauf sélection déclarative d’une capacité disponible |
| Core | API générique, indépendante d’une installation OpenFOAM précise |

## Matrice initiale

| Distribution | Versions à documenter | Statut |
|---|---|---|
| OpenFOAM Foundation | 13, puis 14 | cible de validation |
| OpenFOAM.com | selon extension | à cartographier |
| Autres distributions | selon backend | hors baseline |

La matrice ne constitue pas une promesse de support. Chaque cellule doit être associée à un environnement reproductible, une liste de solvers/bibliothèques disponibles et une validation adaptée.

## Règle de migration

Lorsqu’un fichier sous `openfoam13/` est étudié, il doit être classé selon son rôle. Un outil générique rejoint le core, un adaptateur rejoint le backend, du C++ rejoint une extension, et un correctif version-spécifique rejoint `patches/openfoam/foundation/13/`. Les cas complets et leurs résultats restent dans `examples/` ou `validation/` selon leur usage.
