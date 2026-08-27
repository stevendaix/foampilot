# Incrément overset matriciel — Foundation 13

## Implémentation

La classe `MarineOversetMatrix` applique des contraintes fortes aux cellules `hole` et `interpolated` via l’API publique `fvMatrix::setValues` de Foundation 13. L’appel protégé `setValue` a été écarté. Les champs scalaires et vectoriels sont pris en charge.

Une classe runtime `Foam::fv::marineOversetConstraint` a ensuite été ajoutée. Elle est sélectionnable depuis `system/fvConstraints`, lit la liste `fields`, charge `zoneID`, `oversetCellStatus` et `constant/marineOversetStencils`, puis délègue à `MarineOversetMatrix`. Cette voie est compatible avec le cycle standard des solveurs modulaires Foundation 13, qui appellent `fvConstraints.constrain(eqn)` après assemblage des équations.

Le mover `marineOversetProbe` conserve également une instance persistante de l’interpolation et de l’opérateur, et expose `applyScalar` et `applyVector` pour les solveurs spécialisés qui souhaitent appeler directement l’opérateur.

## Validation

| Validation | Résultat |
|---|---:|
| Compilation `libmarineOversetProbe.so` avec OpenFOAM Foundation 13 | Réussie |
| Compilation du test `marineOversetMatrixTest` | Réussie |
| Sélection dynamique `marineOversetConstraint` via `system/fvConstraints` | Réussie |
| Test analytique : accepteur cellule 2, donneurs cellules 0/1, poids 0,5/0,5 | Valeur 1,5 |
| Test analytique : trou cellule 3 | Valeur 0 |
| Tests Python `test_marine_overset.py` | 10 réussis |

## Limite actuelle

La contrainte est maintenant branchée au mécanisme standard `fvConstraints`, mais la représentation des stencils reste une interpolation inverse-distance locale mono-maillage. L’étape suivante est l’assemblage de la casse DTC à deux maillages, la mise à jour des stencils après mouvement, puis la validation de conservation et de stabilité sur le solveur `incompressibleVoF`.
