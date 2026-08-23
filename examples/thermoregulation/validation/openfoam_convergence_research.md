# Recherche OpenFOAM 13 — convergence du cas humain

## Sources consultées

[1]: https://doc.cfd.direct/openfoam/user-guide-v13/fvsolution "OpenFOAM User Guide v13 — Solution and algorithm control"
[2]: https://doc.cfd.direct/openfoam/user-guide-v13/snappyhexmesh "OpenFOAM User Guide v13 — snappyHexMesh"
[3]: https://doc.cfd.direct/openfoam/user-guide-v13/dambreak "OpenFOAM User Guide v13 — transient time-step control"
[4]: https://doc.openfoam.com/2312/examples/verification-validation/heat-transfer/buoyant-cavity/ "OpenFOAM buoyant cavity verification case"

## Points issus de la documentation

La documentation OpenFOAM indique que les solveurs de pression distinguent les matrices symétriques et asymétriques et que GAMG est utilisé pour accélérer la résolution par agglomération multigrille [1]. Pour les calculs transitoires, `relTol 0` est généralement utilisé afin de forcer la résolution jusqu’à la tolérance absolue à chaque pas [1]. Les facteurs de relaxation servent principalement à stabiliser les calculs stationnaires ou pseudo-transitoires; un facteur faible ralentit mais amortit la mise à jour [1].

Pour snappyHexMesh, la documentation recommande typiquement `nCellsBetweenLevels 3`, une maille de fond d’aspect proche de 1 près des surfaces et un contrôle strict de la non-orthogonalité et de la skewness [2]. La valeur `maxNonOrtho 65` est une limite habituelle, pas une garantie de bonne qualité. Le cas humain contient environ 4 330 cellules concaves et une non-orthogonalité maximale de 64,26°, ce qui reste proche de la limite autorisée.

Le tutoriel `buoyantCavity` officiel utilise une cavité structurée `blockMesh`, une différence de température de 19,6 K et des parois adiabatiques sur les autres faces [4]. Sa géométrie et sa topologie sont beaucoup plus régulières que le maillage MakeHuman.

## Différences constatées

Le cas humain utilise `solver fluid`, le même module que les tutoriels OpenFOAM 13. Ses `fvSchemes` sont identiques à ceux de `externalCoupledCavity`. La différence majeure est dans `fvSolution` : le tutoriel utilise `momentumPredictor yes`, `p_rgh 0,7`, `U 0,3`, `h 0,3`, tandis que le cas humain a été abaissé à `momentumPredictor no`, `p_rgh 0,3`, `U 0,1`, `h 0,1`. Ces réglages très amortis ne corrigent pas une condition limite incohérente et peuvent ralentir fortement la correction de pression.

La condition `fixedFluxPressure` est nécessaire sur toutes les parois soumises à la gravité. Le cas humain avait initialement `human { type zeroGradient; }` pour `p_rgh`, alors que les autres parois étaient en `fixedFluxPressure`. La correction du générateur est donc indispensable.

Le tutoriel `externalCoupledTemperature` OpenFOAM 13 écrit exactement : `area [m²]`, `T [K]`, `qDot [W/m²]` et `htc [W/m²/K]`. Le retour est `value [K]`, `gradient [K/m]` et `valueFraction [-]`. Le protocole n’est donc pas un échange de puissance totale en W.

## Hypothèses prioritaires

1. Condition de pression humaine incorrecte avant correction.
2. Couplage thermique externe plus raide que `fixedValue`, avec retour face-par-face sans sous-relaxation de température.
3. Surfaces CFD et JOS-3 incohérentes, avec mapping par zone très déséquilibré.
4. Maillage proche de la limite de qualité snappyHexMesh, notamment cellules concaves et petites faces.
5. Cas humain configuré en pseudo-stationnaire `steadyState` avec un échange thermique externe présenté comme transitoire; il faut distinguer le pas CFD de la relaxation couplée.
6. Absence de `relTol 0` et de `pFinal` explicite dans le cas humain, alors que la documentation recommande une résolution plus complète en transitoire.
