# Optimisation du maillage CFD humain

## Variantes testées

| Variante | Cellules | Faces humaines | Aire du patch | Non-orthogonalité max | Cellules concaves | Coût test fixe T |
|---|---:|---:|---:|---:|---:|---:|
| Référence actuelle | 89 604 | 20 223 | 3,208 m² | 64,26° | 4 330 | non testé après correction |
| `coarse`, surface `(1 1)` | 43 174 | 4 403 | 2,721 m² | 57,63° | 1 099 | 10 s CFD terminés, environ 7 s CPU |
| `progressive`, surface `(2 2)`, transition 4 cellules | 146 003 | 19 954 | 3,207 m² | 63,98° | 5 388 | 10 s CFD terminés, environ 27 s CPU |

La variante `coarse` réduit fortement le nombre de cellules, mais sous-résout la géométrie cutanée : l’aire du patch chute d’environ 15 %. Elle n’est donc pas acceptable pour un échange face-par-face fidèle. La variante `progressive` conserve l’aire corporelle, mais augmente le coût d’environ 63 % sans améliorer la non-orthogonalité ni le nombre de cellules concaves. La configuration de référence reste le meilleur compromis géométrique parmi ces essais.

## Correction CFD déterminante

L’audit a révélé que `0/p_rgh` utilisait `zeroGradient` sur la face `human`, alors que les autres parois utilisaient `fixedFluxPressure`. Cette incohérence est corrigée dans `prepare_fields.py`. Avec `fixedFluxPressure` sur la paroi humaine, les variantes coarse et progressive terminent un calcul à température fixe jusqu’à `t=10 s`, avec des erreurs de continuité globales de l’ordre de `10⁻¹⁷` et sans erreur flottante.

Cette correction est plus importante que le simple raffinement du maillage. Les variantes précédentes divergeaient même avec une température fixe parce que la condition de pression humaine était incorrecte.

## Test avec échange externe

Après correction de `p_rgh`, le pilote fictif échange correctement les données sur la variante coarse : 4 403 faces sont lues et plusieurs retours sont écrits. Cependant, même avec une température renvoyée constante, le solveur diverge vers `t≈0,15 s`, alors que le même cas à température fixe converge. Le protocole de fichiers fonctionne, mais la condition mixte `externalCoupledTemperature` produit une réponse thermique numériquement plus difficile que la condition `fixedValue`.

La conclusion est donc double. Le maillage humain initial n’était pas la seule cause du problème : la condition de pression sur le patch humain était également incorrecte. Ensuite, le maillage CFD n’est pas responsable du seul problème restant, puisque la réduction ou l’augmentation du raffinement ne supprime pas la divergence de la condition mixte externe. La prochaine étape doit porter sur l’amortissement de l’échange thermique, la valeur de `valueFraction`, le flux de retour et la cohérence énergétique de `data.out/data.in`.

## Recommandation

Conserver provisoirement la configuration de référence à environ 89 604 cellules, corriger `p_rgh` avec `fixedFluxPressure`, puis stabiliser le couplage thermique externe avant toute nouvelle augmentation de raffinement. La variante coarse peut servir à des tests rapides de protocole, mais pas à la production thermophysiologique, car son aire de peau est trop sous-évaluée.

## Références

[1]: https://doc.cfd.direct/openfoam/user-guide-v13/case-management "OpenFOAM User Guide v13"
[2]: https://github.com/TanabeLab/JOS-3 "JOS-3 repository"
