# Audit OF13 — multiRegion/CHT/multiphaseCoolingCylinder2D

La référence OpenFOAM 13 est un cas sériel à deux régions, `fluid` et `solid`, avec un écoulement multiphasique eau/huile dans le fluide et un cylindre solide initialement à `350 K`. L’Allrun exécute `blockMesh`, `splitMeshRegions -cellZonesOnly`, supprime les champs auxiliaires `cellToRegion`, prépare le cas ParaView puis lance `foamMultiRun`. Il n’existe ni `decomposeParDict`, ni Allrun parallèle, ni `createNonConformalCouplesDict` dans la référence.

Le runner `177_multiRegion_CHT_multiphaseCoolingCylinder2D/run.py` importe intégralement les champs `alpha.water`, `alpha.oil`, `T.water`, `T.oil`, `U` et `solid/T`, ainsi que les dictionnaires de propriétés, schémas, solutions et fonctions OF13. Il reproduit la chaîne sérielle avec `blockMesh`, `splitMeshRegions -cellZonesOnly`, nettoyage géré des trois fichiers `cellToRegion`, `foamMultiRun` et reconstruction des régions.

Une première tentative a été volontairement corrigée après audit : l’ajout d’un `createNonConformalCouples` et de `decomposePar -allRegions` ne correspondait pas à la référence, qui utilise les interfaces créées par `splitMeshRegions` et s’exécute en série. Le runner final suit donc strictement l’Allrun officielle.

La validation OF13 atteint `Time=5 s` puis `End` en environ 142 secondes. MULES résout `alpha.oil` et `alpha.water`; la somme des fractions reste `1` dans les cellules. Les températures eau/huile évoluent de manière stable autour de `297–303 K` dans l’extrait final et la région solide est résolue par `DICPCG`. Aucun `FOAM FATAL` n’est observé.

Statut : **validé OF13**.
