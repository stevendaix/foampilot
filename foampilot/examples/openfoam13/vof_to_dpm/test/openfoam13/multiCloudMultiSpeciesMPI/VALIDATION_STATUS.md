# Validation status — multiCloudMultiSpeciesMPI

**Date :** 2026-08-26
**Version :** OpenFOAM 13 / C++14
**Statut :** validé pour le scénario nominal `NP=2`

## Résultat validé

Le cas compile et initialise deux instances `compressibleVoFClouds`, deux `thermoCloud` nommés (`waterCloud`, `fuelCloud`) et deux composants (`H2O`, `C2H5OH`). Les objets auxiliaires sont namespacés par couple `(cloudName, alphaFieldName)`, et les confirmations utilisent la clé composite `(cloudName, alphaFieldName, fragmentId)`.

Avec deux rangs MPI, le solveur atteint `End` et l’audit retourne `pass=true` :

```text
allExpectedCommittedExactlyOnce: true
allExpectedConfirmedExactlyOnce: true
speciesMassesConserved: true
noDefaultCloudFallback: true
noFatalOrMPI: true
solverEnd: true
pass: true
```

Les confirmations observées sont :

```text
VOF confirmation cloud=waterCloud alphaField=alpha.water fragmentId=0 success=true mass=0.646099 speciesMass=2(0.452269 0.19383)
VOF confirmation cloud=fuelCloud alphaField=alpha.air fragmentId=8 success=true mass=4.3165 speciesMass=2(0.8633 3.4532)
```

La légère différence entre les valeurs affichées et les références provient de la précision d’écriture des `scalarList` dans `Info`. L’auditeur utilise une tolérance limitée et documentée ; les valeurs internes de la réconciliation restent les valeurs OpenFOAM complètes.

## Vérifications réalisées

Les bibliothèques `libcompressibleVoFClouds.so` et `libincompressibleVoFClouds.so` compilent sans erreur. Le chemin Direct Commit insère le parcel uniquement sur le rang propriétaire, écrit une confirmation locale, puis attend la réconciliation MPI avant l’application des sources.

La diffusion des identifiants globaux utilise des listes plates et non une sérialisation imbriquée `List<labelList>`. Cette correction évite la carte de cellules vide constatée précédemment sur les rangs non maîtres.

## Limites de qualification

Le test nominal qualifie `NP=2` avec deux clouds et deux fractions massiques configurées. Il ne constitue pas encore une qualification complète de `NP=4`, d’une décomposition différente, du chemin incompressible exécuté dans ce scénario, de l’évaporation, des réactions chimiques, de la diffusion des espèces ou d’un modèle thermo réactif complet.

Les masses d’espèces sont actuellement calculées à partir de `fragment.mass * speciesFractions_`. Elles valident le transport et la comptabilité du vecteur de composition, mais ne remplacent pas une validation où les espèces sont reconstruites depuis des champs physiques indépendants ou modifiées par un modèle de changement de phase.

Avant un merge de production, il est recommandé d’exécuter le même cas avec `NP=4`, de comparer les identités et les bilans, puis d’ajouter des tests négatifs pour le double commit, le cloud inconnu et les fractions dont la somme est différente de un.
