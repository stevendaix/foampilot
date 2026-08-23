# Statut de l’intégration JOS-3/OpenFOAM 13

## Résultat exécuté

| Élément | Résultat | Interprétation |
|---|---:|---|
| Référence `buoyantCavity` | OK jusqu’à `t=1000` | Convection naturelle OpenFOAM 13 et profils expérimentaux disponibles |
| Référence `coolingSphere` CHT | OK jusqu’à `t=1 s` | Chaîne `foamSetupCHT → foamMultiRun → reconstructPar` validée |
| Maillage MakeHuman | OK | 89 604 cellules et 20 223 faces humaines |
| Mapping OpenFOAM → 17 zones JOS-3 | OK | Mapping réalisé sur les identifiants de faces et leurs centroïdes |
| Échange JOS-3 → OpenFOAM | OK sur les premières itérations | 20 223 faces, `h≈2,03–51,36 W m⁻² K⁻¹`, températures de surface JOS-3 reçues |
| Convergence CFD humaine complète | NON | Divergence de la pression après le premier pas CFD |

## Corrections intégrées

Le cas humain utilise désormais un transport laminaire, `fixedFluxPressure` pour les champs de pression, un prédicteur de quantité de mouvement désactivé et des facteurs de relaxation réduits. Ces réglages sont une stratégie de stabilisation pour la convection naturelle basse vitesse; ils ne constituent pas une validation expérimentale du corps humain.

Le lanceur `run_coupled_case.sh` démarre le pilote Python et `foamRun` en parallèle. Le script de référence `run_openfoam13_references.py` est reproductible et écrit `results/openfoam13_reference_report.md`.

## Blocage restant

La matrice de pression devient non bornée après le premier échange thermique. L’échange de données est donc fonctionnel, mais le champ fluide humain n’est pas encore physiquement ou numériquement validé. La suite correcte consiste à isoler et valider le cas humain avec une température fixe effectivement imposée, puis à réintroduire progressivement la condition mixte `externalCoupledTemperature`, le flux JOS-3 et enfin la convection naturelle complète.

## Références

[1]: https://doc.cfd.direct/openfoam/user-guide-v13/case-management "OpenFOAM User Guide v13"
[2]: https://github.com/TanabeLab/JOS-3 "JOS-3 repository"
