# Catalyst Heat-Up

## Objet

Ce tutoriel de Tobias Holzmann présente un calcul de transfert thermique conjugué dans un système catalytique SCR. Le cas utilise trois régions — `fluid`, `SCR1` et `SCR2` — et le solveur multi-régions `foamMultiRun` d’OpenFOAM Foundation. Les échanges thermiques inter-régions sont représentés par des modèles `interRegionHeatTransfer` à coefficient constant [1].

## Portage FoamPilot

`run.py` recrée le cas à partir des dictionnaires embarqués dans `templates.py`, copie les actifs officiels et exécute les étapes suivantes : conversion du maillage UNV par `ideasUnvToFoam`, génération du maillage par `snappyHexMesh`, séparation par zones de cellules avec `splitMeshRegions -cellZones`, organisation des trois maillages régionaux et création des patches régionaux avec `createPatch`.

Le runner lance ensuite `foamMultiRun` via `Solver.run_command`. Le calcul est volontairement borné à un seul pas de temps (`endTime 1e-5`, `deltaT 1e-5`) et à un correcteur PIMPLE (`nOuterCorrectors 1`) afin de fournir un smoke run reproductible. Le niveau de raffinement de `snappyHexMesh` est réduit localement pour limiter le coût de validation ; la structure multi-régions, les zones SCR et les modèles de transfert thermique sont conservés.

L’audit de FoamPilot n’a révélé aucune méthode manquante. Aucune extension du cœur de FoamPilot n’a été ajoutée pour ce cas.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion du maillage | `ideasUnvToFoam` exécuté avec succès. |
| Séparation des régions | `splitMeshRegions -cellZones` produit les régions SCR et fluide. |
| Maillage régional | `constant/SCR1/polyMesh`, `constant/SCR2/polyMesh` et `constant/fluid/polyMesh` sont créés. |
| Couplage thermique | Les modèles `interRegionHeatTransfer` `Coupling` et `Coupling2` sont sélectionnés. |
| Mapping inter-régions | 48 144 et 48 146 couplages de cellules sont calculés pour les deux régions SCR. |
| Résolution thermique | Les équations `e` de SCR1/SCR2 et `h` du fluide convergent au premier pas. |
| Résolution fluide | Les équations de vitesse, pression, turbulence et continuité sont résolues. |
| Fin du calcul | `foamMultiRun` atteint `End` à `Time = 1e-05 s` sans erreur. |

Le cas est **validé pour le smoke run OpenFOAM 13**. La validation couvre le maillage multi-régions, le mapping inter-régions, le transfert thermique conjugué et le démarrage/arrêt normal du solveur.

## Limites

Le calcul de production original est beaucoup plus long. La version FoamPilot fournie ici ne prétend pas reproduire la durée physique complète : elle vérifie la chaîne de mise en données et un premier pas thermo-fluidique. La résolution de maillage et le nombre de correcteurs sont réduits uniquement pour la validation automatisée.

## Référence

[1]: https://holzmann-cfd.com/community/training-cases/catalyst-heat-up — Tobias Holzmann, *Catalyst Heat-Up Simulation*.
