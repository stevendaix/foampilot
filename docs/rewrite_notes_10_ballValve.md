# Note de travail — runner #10 `compressibleVoF/ballValve`

Référence OpenFOAM 13 locale: `/opt/openfoam13/tutorials/compressibleVoF/ballValve`.

Le runner actuel importe encore `constant/momentumTransport`, `system/controlDict`, `system/createNonConformalCouplesDict`, `system/fvSchemes`, `system/fvSolution` et `constant/fvModels`, ainsi qu’un `blockMeshDict` de ressource et l’asset `ballValve-torus.obj.gz`.

Entrées structurantes relevées dans les dictionnaires OF13: `controlDict` contient `solver compressibleVoF`, `endTime 0.1`, `deltaT 1e-5`, `writeControl adjustableRunTime`, `writeInterval 1e-3`, `adjustTimeStep yes`, `maxCo 0.25`, `maxAlphaCo 0.25`; `fvSchemes` contient les sections `ddtSchemes`, `gradSchemes`, `divSchemes`, `laplacianSchemes`, `interpolationSchemes`, `snGradSchemes`; `fvSolution` contient `solvers`, `potentialFlow`, `PIMPLE`, `relaxationFactors`; `createNonConformalCouplesDict` contient `nonConformalCouples`; `momentumTransport` contient `simulationType RAS` et `RAS`; `fvModels` contient le modèle `VoFCavitation`.

L’API existante couvre déjà `ConstantDirectory.configure_vof`, `PhasePhysicalPropertiesFile`, `MomentumTransportFile`, les conditions de champs, `BlockMesher.create_non_conformal_couples` et la gestion d’asset gzip. Le travail restant est de fournir des builders déclaratifs génériques pour les dictionnaires système complexes et de retranscrire la géométrie de base au lieu d’utiliser `import_reference_dict`.
