# Réécriture #11 — XiFluid engine2Valve2D

Le runner actuel importe tous les dictionnaires `system/`, `constant/` et `0/`, ainsi que trois maillages `blockMeshDict.*.orig`; il doit être réécrit sans `import_reference_case`, `import_reference_file`, `import_reference_field`, `iterdir` ni boucle de copie.

Le cas exige une API déclarative pour une famille de maillages temporels: cylindre, soupape fermée et soupape ouverte, avec variables `pistonPos`, `valveLift`, `nPiston` et `nValve`. Le pipeline OF13 comprend `blockMesh -mesh`, `mirrorMesh`, translations, `mergeMeshes`, `createBaffles`, `splitBaffles`, rotation/mise à l’échelle, `createPatch` et `createNonConformalCouples`. Les dictionnaires spécialisés à généraliser sont `createBafflesDict`, `createPatchDict.inletFuel`, `createNonConformalCouplesDict`, `fvConstraints`, `fvModels`, `dynamicMeshDict`, `zonesGenerator` et les champs XiFluid (`Xi`, `b`, `egr`, `ft`, `fu`, `Tu`, `alphat`, `k`, `omega`, `nut`, `T`, `U`, `p`).

La réécriture doit conserver les opérations temporelles via les commandes FoamPilot `run_utility`, mais générer le contenu des dictionnaires et des champs par des builders déclaratifs. La géométrie des trois maillages doit être retranscrite dans `BlockMesher` avec définitions, sommets, blocs, arcs et frontières paramétriques; elle ne doit pas être remplacée par un simple import de la ressource OF13.
