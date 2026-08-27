# Audit de référence OF14 — XiFluid/engine2Valve2D

Source normative : https://github.com/OpenFOAM/OpenFOAM-14/tree/master/tutorials/XiFluid/engine2Valve2D

Le cas utilise `XiFluid` avec une géométrie moteur 2D et trois variantes de maillage (`blockMeshDict.cylinder.orig`, `blockMeshDict.valveClosed.orig`, `blockMeshDict.valveOpen.orig`). `Allmesh` calcule dynamiquement la levée de soupape et la position du piston, modifie les dictionnaires avec `foamDictionary`, exécute plusieurs `blockMesh` dans des meshes nommés, applique `mirrorMesh`, `transformPoints`, `createBaffles`, renomme les patches, crée les couples non conformes, puis fusionne les meshes. `Allrun` initialise par `potentialFoam` et lance `foamRun` avec le solver `XiFluid`.

Les champs de référence comprennent notamment `T`, `Tu`, `U`, `Xi`, `alphat`, `b`, `egr`, `ft`, `fu`, `k`, `nut`, `omega` et `p`. Les conditions limites utilisent des entrées tabulées en angle vilebrequin, des fonctions de paroi `omegaWallFunction`, des patches `frontAndBack` de type `empty` et des familles `nonCouple.*`. Les constantes incluent `combustionProperties`, `dynamicMeshDict`, trois fichiers thermochimiques `.foam`, `fvModels`, `momentumTransport` avec `realizableKE`, `physicalProperties` et `zonesGenerator`.

Le runtime OpenFOAM 14 a été installé et vérifié (`WM_PROJECT_VERSION=14`). L’adaptation ne doit pas commencer par une approximation du maillage : il faut d’abord ajouter au cœur FoamPilot des opérations génériques de maillage et de dictionnaire capables de préserver les meshes nommés, substitutions de paramètres, transformations, baffles, renommages de patches, fusion et couples non conformes. Le cas reste en audit jusqu’à ce que ces primitives soient disponibles.
