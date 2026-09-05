# Migration vers FoamPilot v3

## Phases

La migration suit l’ordre suivant : baseline, cartographie réelle, règles de dépendances, séparation capabilities/workflows, construction du core, backend OpenFOAM, registry, workflows, tests, CI, shims, documentation, puis migration progressive des anciens dossiers.

Cette première PR couvre uniquement la **baseline**, la cartographie et les règles d’architecture. Elle ne déplace pas massivement les modules existants.

## Shims

Pendant la transition, une ancienne API peut déléguer vers la nouvelle :

```text
ancienne API → shim → nouvelle API
```

Chaque shim doit émettre un `DeprecationWarning`, documenter la date prévue de suppression et disposer d’un test. La suppression ne peut intervenir qu’après migration des workflows et validation de la nouvelle API.

## Découpage proposé

| PR | Périmètre | Suppression fonctionnelle ? |
|---:|---|---:|
| 1 | Audit architecture et baseline | Non |
| 2 | Règles de dépendances et checks | Non |
| 3 | Core case/dictionaries | Non, shims si nécessaire |
| 4 | Core geometry | Non, shims si nécessaire |
| 5 | Core meshing | Non, shims si nécessaire |
| 6 | Postprocessing/reporting | Non |
| 7 | Backend OpenFOAM | Non |
| 8 | Registry et système d’extensions | Non |
| 9 | Migration C++ | Non sans validation équivalente |
| 10 | Migration des workflows | Non sans tests |
| 11 | Tests et CI | Non |
| 12 | Documentation finale | Non |
| 13 | Suppression legacy | Oui, uniquement après critères de sortie |

## Critères de sortie

Avant de supprimer une ancienne API ou un ancien dossier, il faut disposer de la nouvelle API, de tests unitaires et d’intégration, de workflows migrés, de documentation et d’un contrôle CI architectural. Toute validation scientifique doit rester relançable depuis un checkout propre.
