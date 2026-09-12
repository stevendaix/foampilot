# Propeller Foundation 13

Ce squelette adapte la configuration du cas `propeller-OpenFOAM` à OpenFOAM Foundation 13. Le runner est `marineFoam` et le module sélectionné est `compressibleVoF`. La configuration MRF conserve une zone `rotor`, l’axe `(0 1 0)` et `omega=314.16 rad/s`, avec `AMI1` et `AMI2` comme patches non rotatifs. Un modèle Foundation 13 `actuationDisk` est également généré dans `constant/fvModels` pour les variantes qui ne disposent pas encore d’une zone rotor maillée.

Les dictionnaires `controlDict`, `MRFProperties` et `fvModels` ont été validés avec `foamDictionary`. Le cas n’est pas encore exécutable physiquement, car le dépôt de référence construit la géométrie avec cfMesh et fournit deux sous-domaines rotor/stator ; la zone `rotor` et les interfaces AMI doivent être recréées dans un maillage Foundation 13 avant toute comparaison de poussée, couple ou rendement.

Le fichier OpenCFD `fvOptions` contenant `limitTemperature` n’est pas activé dans ce squelette. Il est remplacé par le chemin `fvModels` Foundation 13 lorsque l’actionnement disk est utilisé.
