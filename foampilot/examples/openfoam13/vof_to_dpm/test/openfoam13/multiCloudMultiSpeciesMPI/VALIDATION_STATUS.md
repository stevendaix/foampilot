# Validation status — multiCloudMultiSpeciesMPI

**Date :** 2026-08-27
**Version :** OpenFOAM 13 / C++14
**Statut :** validé pour `NP=2`, `NP=4` simple et `NP=4` hiérarchique

## Résultat validé

Le cas compile et initialise deux instances `compressibleVoFClouds`, deux `thermoCloud` nommés (`waterCloud`, `fuelCloud`) et deux composants (`H2O`, `C2H5OH`). Les objets auxiliaires sont namespacés par couple `(cloudName, alphaFieldName)`, et les confirmations utilisent la clé composite `(cloudName, alphaFieldName, fragmentId)`.

Avec deux rangs MPI, quatre rangs en décomposition simple et quatre rangs en décomposition hiérarchique `(2 2 1)`, le solveur atteint `End` et l’audit retourne `pass=true`. Les trois configurations produisent les mêmes identifiants de fragments `waterCloud:0` et `fuelCloud:8`, les mêmes masses et les mêmes masses d’espèces à la tolérance de l’auditeur.

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

## Environnement et options MPI

Le runner charge `FOAM_BASHRC` ou `WM_PROJECT_DIR/etc/bashrc` sans ignorer les erreurs. Il initialise `USER` si nécessaire pour les shells minimaux et exige un `FOAM_USER_LIBBIN` existant. Les options complémentaires MPI sont fournies par `MPI_EXTRA_ARGS`; la validation locale NP=4 utilise `MPI_EXTRA_ARGS=--oversubscribe` uniquement parce que l’environnement ne fournit pas quatre slots.

La reconstruction validée utilise `/opt/openfoam13` avec les bibliothèques applicatives compilées sous `/home/ubuntu/OpenFOAM/ubuntu-13`. Le code de retour `141 pop_var_context` observé avec `set -e` est évité en chargeant le bashrc avant `errexit`; le chargement reste contrôlé explicitement par son code retour.

## Limites de qualification

Le scénario qualifie maintenant NP=2, NP=4 et deux décompositions pour deux clouds et deux fractions massiques configurées. Le chemin incompressible n’est pas exécuté dans ce scénario, et l’évaporation, les réactions chimiques, la diffusion des espèces et un modèle thermo-réactif complet restent hors périmètre.

Les masses d’espèces sont actuellement calculées à partir de `fragment.mass * speciesFractions_`. Elles valident le transport et la comptabilité du vecteur de composition, mais ne remplacent pas une validation où les espèces sont reconstruites depuis des champs physiques indépendants ou modifiées par un modèle de changement de phase.

Les garde-fous de configuration rejettent maintenant les clouds vides, les fractions négatives et les fractions dont la somme diffère de un. La détection des confirmations en double reste rejetée par la réconciliation MPI. Un test d’intégration incompressible et une validation explicite du comportement d’un nom de cloud inconnu restent recommandés avant le merge de production.
