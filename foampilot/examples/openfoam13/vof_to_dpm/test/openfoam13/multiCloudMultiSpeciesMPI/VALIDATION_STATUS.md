# Validation status — multiCloudMultiSpeciesMPI

Date: 2026-08-26

## Résultats

Le cas compile et initialise deux instances `compressibleVoFClouds`, deux `thermoCloud` nommés (`waterCloud`, `fuelCloud`) et deux composants liquides (`H2O`, `C2H5OH`). Le patch framework multi-cloud ajoute le dispatch exact par nom et transporte un vecteur `speciesMassFractions` dans `directParcelData`.

Avec deux rangs MPI, le premier modèle produit un commit Direct Commit confirmé :

```text
massDetected=0.646099 massPrepared=0.646099 enthalpyDetected=5000.73 enthalpyPrepared=5000.73
VOF direct commit cloud=waterCloud fragmentId=0 success=true mass=0.646099
massCreated=0.646099 massConfirmed=0.646099 enthalpyCreated=5000.73 enthalpyConfirmed=5000.73
```

Le scénario n’est pas encore passant. Lors de la seconde réconciliation, le processus MPI déclenche un segfault dans `vofFragmentTransition::detect()`. L’audit doit donc retourner `pass=false`. Aucune conclusion de conservation multi-cloud ne doit être publiée à partir de cette exécution.

## Diagnostic

Le cas expose vraisemblablement un état partagé ou un nom d’objet partagé entre plusieurs instances du modèle : `vofFragmentTransitionManager` doit être rendu indépendant par cloud, et les champs temporaires d’audit (`vofFragmentMask`, `vofAlphaRhoTransferRate`, `vofConfirmedTransferRate`) doivent être namespacés par instance ou cloud. Les fragments doivent également porter leur `cloudName` dès la détection ; une seule liste globale de fragments ne suffit pas pour router deux champs VOF différents.

Le prochain correctif doit introduire un manager par `(cloudName, alphaField)`, des noms de champs par cloud et des confirmations indexées par `(cloudName, fragmentId)`. Le test ne pourra être déclaré vert qu’après un lancement à `NP=2` puis `NP=4`, sans segfault, avec exactement un commit et une confirmation pour chaque clé et un bilan par espèce.
