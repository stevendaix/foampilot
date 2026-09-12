# Overset et Foundation OpenFOAM 13

Le dépôt `DTCMoving_Overset` fourni comme référence utilise `overInterDyMFoam`, `dynamicOversetFvMesh`, `zoneID`, deux maillages fusionnés et les bibliothèques OpenCFD `overset`/`rigidBodyDynamics`. Ces symboles ne sont pas des entrées Foundation 13 natives équivalentes dans l’arbre audité.

Le solver `marineFoam` ne transforme donc pas silencieusement un dictionnaire overset OpenCFD en cas Foundation 13. Le pré-validateur le rejette avec un diagnostic explicite.

Deux chemins sont possibles :

| Chemin | Description | Statut |
|---|---|---|
| Mouvement Foundation 13 sans overset | Utiliser `incompressibleVoF`, `librigidBodyMeshMotion.so`, `mover`, un domaine unique et un maillage suffisamment large | Supporté par le driver et les helpers actuels |
| Portage overset fidèle | Porter/adapter un runtime overset aux API Foundation 13, puis ajouter les champs `zoneID`, les cell sets, l’assemblage des maillages et les interpolations | À réaliser dans un module C++ séparé |

Le second chemin ne doit commencer qu’après compilation d’un premier plugin minimal dans l’installation OpenFOAM 13 cible. La validation d’acceptation devra au minimum vérifier la création des zones, les cellules hole/acceptor/donor, la conservation du flux à travers l’interface, la stabilité du mouvement de coque et la comparaison de trajectoire avec le cas de référence.
