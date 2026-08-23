# Diagnostic de convergence OpenFOAM 13 du cas humain

## Conclusion principale

La divergence ne provient pas d’un seul facteur. Les essais séparent trois niveaux de problème : le protocole `externalCoupledTemperature` fonctionne sur une cavité stable; le cas humain converge avec une température fixe après correction de `p_rgh`; le cas humain diverge lorsque la condition mixte externe est réintroduite. Le problème restant est donc l’interaction entre maillage complexe, pression hydrostatique et condition thermique mixte, renforcée par l’écart de surface entre JOS-3 et OpenFOAM.

## Ce que disent les tutoriels et la documentation

Le tutoriel officiel `buoyantCavity` est une cavité structurée générée par `blockMesh`, avec une différence de température de 19,6 K et des parois restantes adiabatiques [1]. Cette référence est beaucoup plus régulière que le maillage MakeHuman, qui contient 4 330 cellules concaves et une non-orthogonalité maximale de 64,26°.

La documentation OpenFOAM 13 recommande généralement trois couches de cellules entre niveaux de raffinement successifs dans `snappyHexMesh`; elle indique aussi que la maille de fond doit avoir un rapport d’aspect proche de 1 près de la surface [2]. Le cas humain utilise bien deux à quatre couches selon les variantes testées, mais les cellules concaves et les petites faces restent nombreuses.

Pour la résolution, OpenFOAM recommande de distinguer les solveurs adaptés aux matrices symétriques et asymétriques, et explique que `relTol 0` est généralement préférable en transitoire lorsque l’on veut résoudre chaque pas jusqu’à la tolérance absolue [3]. Le cas humain utilise `relTol 0.01` pour `p_rgh` et `relTol 0.1` pour les autres champs, ce qui peut arrêter la correction trop tôt pendant un couplage thermique raide.

Les facteurs de relaxation sont destinés à stabiliser en particulier les calculs stationnaires ou pseudo-transitoires [3]. Le tutoriel `externalCoupledCavity` utilise `momentumPredictor yes`, `p_rgh 0.7`, `U 0.3` et `h 0.3`. Le cas humain avait été abaissé à `momentumPredictor no`, `p_rgh 0.3`, `U 0.1` et `h 0.1`. Ces faibles facteurs peuvent amortir les oscillations mais ne corrigent pas une condition limite erronée et rendent le couplage beaucoup plus lent.

La documentation rappelle que `fixedFluxPressure` ajuste le gradient de pression afin que le flux de frontière soit cohérent avec la condition de vitesse lorsqu’il existe des forces volumiques comme la gravité [4]. Dans le cas humain, `p_rgh` était initialement en `zeroGradient` sur `human`; cette incohérence a provoqué des matrices de pression extrêmement mal conditionnées. Après remplacement par `fixedFluxPressure`, le cas humain à température fixe converge jusqu’à `t=10 s`, avec des erreurs globales de continuité proches de `10^-17`.

## Causes classées par priorité

| Priorité | Cause | Preuve dans le cas |
|---:|---|---|
| 1 | `p_rgh` en `zeroGradient` sur la paroi humaine | Correction nécessaire; la température fixe devient stable après `fixedFluxPressure` |
| 2 | Condition mixte `externalCoupledTemperature` plus raide que `fixedValue` | Même température externe constante, mais divergence vers `t≈0,15 s` |
| 3 | Surface et mapping incohérents | Aire CFD 3,208 m² contre BSA JOS-3 1,874 m²; bras absents du mapping |
| 4 | Qualité snappyHexMesh limite | 4 330 cellules concaves; non-orthogonalité max 64,26° |
| 5 | Résolution linéaire trop relâchée pour un couplage transitoire | `relTol=0.01` pour la pression et `0.1` pour les champs |
| 6 | Pseudo-transitoire mal séparé du couplage | `ddt steadyState` avec `deltaT=0.05`; le pas OpenFOAM et le pas JOS-3 ne représentent pas la même échelle |

## Corrections recommandées

Premièrement, régénérer systématiquement `0/p_rgh` avec `fixedFluxPressure` sur `human`. Deuxièmement, tester `externalCoupledTemperature` avec une température renvoyée constante et une sous-relaxation côté Python :

`T_new = (1-alpha) T_old + alpha T_JOS3`, avec `alpha` initialement entre 0,05 et 0,2.

Troisièmement, ajouter `pFinal`, `UFinal` et `hFinal` avec `relTol 0` pour forcer une résolution complète du dernier correcteur. Quatrièmement, réduire temporairement `deltaT` à `0.005–0.01 s` ou utiliser un contrôle adaptatif du pas, et journaliser le nombre de Courant. Cinquièmement, remplacer les schémas `orthogonal` par `corrected` ou `limited corrected` si les termes de diffusion deviennent sensibles à la géométrie; cette modification doit être testée avec `checkMesh` et non appliquée aveuglément.

Enfin, corriger le mapping géométrique avant de calibrer les flux. La surface CFD ne doit pas être confondue avec la BSA JOS-3. Une puissance de zone doit être conservée en W et un flux retourné à OpenFOAM en W/m². Il faut soit renormaliser par zone vers les aires physiologiques JOS-3, soit recalibrer explicitement les capacités du réseau distribué sur les aires CFD.

## Références

[1]: https://doc.openfoam.com/2312/examples/verification-validation/heat-transfer/buoyant-cavity/ "OpenFOAM buoyant cavity verification case"
[2]: https://doc.cfd.direct/openfoam/user-guide-v13/snappyhexmesh "OpenFOAM User Guide v13 — snappyHexMesh"
[3]: https://doc.cfd.direct/openfoam/user-guide-v13/fvsolution "OpenFOAM User Guide v13 — Solution and algorithm control"
[4]: https://doc.cfd.direct/openfoam/user-guide-v13/dambreak "OpenFOAM User Guide v13 — fixedFluxPressure and time-step control"
