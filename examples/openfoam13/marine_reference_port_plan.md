# Plan de portage des cas manœuvre et propeller vers Foundation 13

## Cas Turning35 de maneuveringLib

Le cas de référence comporte trois maillages indépendants `background`, `hull` et `rudder`, des dictionnaires de mouvement `dynamicMeshDict.turning` et `dynamicMeshDict.propulsion`, ainsi que des contrôles de manœuvre. Le portage Foundation 13 doit conserver le mover six-DoF natif pour le background et représenter le gouvernail par un mouvement de maillage compatible, sans réintroduire `dynamicOversetFvMesh` ou les types de patch OpenCFD.

## Cas propeller-OpenFOAM

Le cas de référence est stationnaire et utilise `rhoSimpleFoam`, un rotor `cellZone rotor`, `MRFProperties` avec `omega=314.16 rad/s`, axe `(0 1 0)` et patches AMI non rotatifs `AMI1 AMI2`. Foundation 13 devra utiliser le module compressible approprié ou une équation marine dédiée, avec un dictionnaire `momentumTransport` et une zone rotor effectivement présente dans le maillage. Le modèle `limitTemperature` du fichier `fvOptions` OpenCFD ne doit pas être copié tel quel ; il doit être traduit en `fvModels` Foundation 13 ou désactivé dans le baseline.

## Validation prévue

| Cas | Précondition de maillage | Validation minimale |
|---|---|---|
| Turning35 | background/hull/rudder et zones de mouvement disponibles | démarrage `marineFoam`, mouvement six-DoF, rudder/propulsion, forces |
| Propeller MRF | zone cellulaire `rotor` et patches AMI réellement créés | lecture MRF, rotation non nulle, résidus et poussée |
| Propeller AMI | deux interfaces conformes ou non conformes | conservation et absence de faces AMI invalides |

Les tests Python valident déjà les paramètres et les générateurs. Les validations OpenFOAM physiques restent dépendantes de la construction des maillages correspondants.

## Choix du solver Foundation 13

Le cas propeller de référence utilise `rhoSimpleFoam`, tandis que Foundation 13 moderne expose notamment des modules `compressibleVoF` et conserve certains solvers compressibles dans les applications legacy. Le portage doit sélectionner et compiler explicitement le chemin adapté aux champs du cas avant toute validation MRF/AMI ; la présence de `MRFProperties` seule ne suffit pas à démontrer la reproduction du cas.
