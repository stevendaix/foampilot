# Statut : Tutoriel pitzDaily_step — Demande d'implémentation

## Objectif
Implémenter le tutoriel `03_pitzDaily_step/run.py` : écoulement sur marche descendante
(backward-facing step), inspiré du tutoriel de référence OF13
`tutorials/incompressibleFluid/pitzDaily`.

## État actuel

### Fait
1. ✅ Géométrie Gmsh (marche descendante 6×1.0×0.01 m)
2. ✅ Classification des faces par bounding box + centre de masse OCC
3. ✅ Export direct vers `constant/polyMesh/` via `DirectOpenFoamExporter`
4. ✅ Correction du boundary file (`frontAndBack`→`empty`, `walls`→`wall`)
5. ✅ Exutoire du bug `GmshMesher` (référence `solver._solver` non définie)
6. ✅ Découverte de l'API `occ.extrude(..., numElements=[1])` pour 1 couche Z

### Problèmes restants (causants d'échec)
1. **BUG : `relaxationFactors` vide** — en remplaçant
   `{"fields": {}, "equations": {}}`, le bloc fvSolution génère
   `relaxationFactors { }` (vide), ce qui cause des erreurs OpenFOAM.
   → Solution : ne pas toucher aux `relaxationFactors`.

2. **BUG : `div(phi,U)` sans `bounded`** — le override manuel de
   `divSchemes` supprime le préfixe `bounded` que foampilot ajoute
   par défaut. Pour les maillages grossiers, `bounded` est important
   pour la stabilité transitoire.
   → Solution : ne pas override `divSchemes`.

3. **BUG : tolérance GAMG `1e-07`** — trop stricte pour le maillage
   grossier (586 nœuds).
   → Solution : utiliser `1e-06`.

4. **BUG : nom de module import** — `direct_openflow_exporter` devrait
   être `direct_openflow_exporter` et la classe `DirectOpenFoamExporter`
   devrait être `DirectOpenFoamExporter` (uppercase FOAM).

5. **BUG : `GAMM` vs `GAMG`** — typo dans le solveur de pression.

## Tests de validation
- **kEpsilon + transient PIMPLE + 1 couche Z + defaults foampilot** ✅ :
  Simulation converge (continuity ~1e-6), atteint `endTime`.
- **kOmegaSST + transient PIMPLE** : divergence (SIGFPE) — probablement
  dû aux bugs 1-3 ci-dessus, pas au modèle de turbulence.

## Référence OF13 (pitzDaily)
- `solver : incompressibleFluid` (foamRun)
- `ddtSchemes : Euler` (transient)
- `PIMPLE` avec `nCorrectors 2`, `nNonOrthogonalCorrectors 0`
- `GAMG` pour `p` avec `DICGaussSeidel`
- `div(phi,U) : Gauss linearUpwind grad(U)` (sans `bounded` dans la ref)
- `laplacianSchemes : Gauss linear corrected`
- `snGradSchemes : corrected`
- U inlet = 10 m/s, k = 0.375, epsilon = 14.855
- `nu = 1e-05` (air)
- `endTime = 0.3`, `maxCo = 5`

## Prochaines étapes
1. Corriger le script `run.py` avec les 5 fixes ci-dessus
2. Ne PAS override `divSchemes` ni `relaxationFactors`
3. Utiliser GAMG tolérance `1e-06`
4. Relancer la simulation
5. Valider la convergence des résidus
