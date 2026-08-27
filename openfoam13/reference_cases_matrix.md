# Matrice de reproduction des cas marins de référence

| Cas | Dépôt et révision locale | Cas principal | Solveur de référence | Maillage de référence | Sorties à reproduire |
|---|---|---|---|---|---|
| Maneuvering | `balabibo/maneuveringLib`, branche `main`, commit `0a5f75f083100b867971669bdf1496bbe660ce2a` | `tutorial/Turning35` | À inspecter dans le script et les dictionnaires du cas | `blockMesh` puis `snappyHexMesh`, sous-cas background/hull/rudder | Mouvement 6-DoF, trajectoire de giration, forces et moments du navire et du gouvernail |
| Propeller | `skfelix/propeller-OpenFOAM`, branche `main`, commit `eaa8e70e8ecc2c48ba9927cac5feb83c8e39e416` | `mesh` + `simulationTemplate` | Référence basée sur `rhoSimpleFoam` | cfMesh, avec sous-domaines rotor/stator et interfaces AMI | Résidus, poussée, couple, vitesse de rotation et comportement AMI/MRF |
| DTC moving overset | `myozinaung/DTCMoving_Overset`, branche `master`, commit `e8ef2d2f0f5e23645e0d81c3fa4a747fbb8f7520` | Racine du dépôt, sous-cas `background` et `hull` | `overInterDyMFoam` | `blockMesh` puis `snappyHexMesh`, maillage overset natif OpenCFD | Évolution 6-DoF, forces, interface overset et conservation de masse |

## État Foundation 13

Le cas DTC a déjà un calcul court Foundation 13 terminé jusqu’à `t = 0.01 s` avec le runtime overset custom et le mouvement rigide. Un smoke test MRF/actuationDisk Foundation 13 est également terminé sur une zone rotor synthétique, mais il ne constitue pas encore la reproduction du propeller réel.

Le cas propeller doit être porté avec un solveur Foundation 13 réellement disponible. Le dépôt de référence emploie `rhoSimpleFoam`, alors que le squelette actuel utilise `compressibleVoF`, qui n’est pas exposé par `marineFoam`. Le choix devra être arrêté avant le calcul hydrodynamique final.

Le cas Turning35 doit être traité comme un exemple indépendant, avec ses trois sous-domaines et ses modèles de mouvement, et non comme une simple variante du cas DTC.
