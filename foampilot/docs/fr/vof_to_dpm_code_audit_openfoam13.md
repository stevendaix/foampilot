# Audit technique du portage VOF-to-DPM sous OpenFOAM 13

**Dépôt :** [stevendaix/foampilot](https://github.com/stevendaix/foampilot)  
**Branche auditée :** `feat/vof-to-dpm-conservative-transition`  
**Dernier commit audité :** `88ab9d1`  
**Environnement :** OpenFOAM Foundation 13, C++14, Ubuntu 24.04

## Conclusion exécutive

Le portage est **compilable et fonctionnel sur les scénarios élémentaires**, et le cas spray cross-flow montre désormais la création de parcels successifs ainsi qu’un bilan local volume–masse exact pour le premier fragment. Toutefois, l’audit révèle deux défauts de niveau élevé dans le code livré : la consommation VOF et l’injecteur sont verrouillés comme des opérations uniques dans la version committée, et le transfert compressible utilise la densité du champ de phase cible au lieu de la densité de la phase liquide pour la source ajoutée. Ces deux points empêchent de considérer le portage comme prêt pour un spray continu ou pour un transfert compressible eau–air général.

Le cas spray est donc une bonne démonstration de la chaîne logicielle, mais le nombre de parcels observé dans le run de référence doit être interprété avec prudence : la détection de fragments continue, alors que l’état `emitted_` de l’injecteur et l’état `transitionApplied_` du fvModel limitent la conversion et la consommation à une seule transition dans le code livré.

## Matrice de sévérité

| ID | Domaine | Constat | Gravité | État |
|---|---|---|---|---|
| C++-01 | Spray continu | `vofFragmentInjection` conserve `emitted_` après la première injection et ne réarme pas l’état par pas ou par identifiant de fragment | **Critique** | À corriger |
| C++-02 | Conservation compressible | `compressibleVoFClouds` utilise `rho[cell]` du champ cible dans la source de phase ; avec `rho_water != rho_air`, les kg retirés et ajoutés ne sont pas égaux | **Critique** | À corriger |
| C++-03 | Consommation VOF | `transitionApplied_` est définitif dans `incompressibleVoFClouds` et `compressibleVoFClouds` | **Élevée** | À corriger |
| C++-04 | Cohérence énergie | `energyTransferPending_` est armé par la détection, avant confirmation de création effective d’un parcel | **Élevée** | À sécuriser |
| C++-05 | Robustesse géométrique | Un échec éventuel de `findCellAtPosition` peut laisser un index de cellule invalide avant lecture de `rho_` ou de `T` | **Moyenne/élevée** | À durcir |
| TEST-01 | Couverture | Les tests Python couvrent l’algorithme hors ligne, mais pas le cycle C++ fvModel–InjectionModel ni plusieurs conversions successives | **Élevée** | À compléter |
| TEST-02 | Packaging | `pytest -q test/test_vof_to_dpm.py` échoue à la collecte car le test importe `vof_to_dpm` comme module top-level non exposé par le packaging courant | **Moyenne** | À corriger |
| CFG-01 | Exemple | Le cas spray doit imposer une cohérence entre `rhoLiquid` et `constantProperties.rho0` ; sinon la masse du parcel est déterminée par une densité de parcel différente de celle du liquide | **Élevée** | Corrigé dans l’exemple |

## Analyse détaillée

### C++-01 et C++-03 : conversion unique au lieu d’une conversion de spray

Dans `vofFragmentInjection.C`, `nParcelsToInject()` retourne zéro dès que `emitted_` est vrai. `emitted_` est positionné lorsque le dernier parcel de la liste courante reçoit sa position. Aucun mécanisme livré ne réinitialise cet état au pas suivant et aucun ensemble d’identifiants déjà injectés n’est maintenu.

Les deux fvModels contiennent une limitation analogue. Dans `correct()`, les fragments sont détectés à chaque `timeIndex`, mais la consommation n’est armée que sous la condition `!transitionApplied_`, puis `transitionApplied_` devient vrai définitivement. Le modèle peut donc détecter les fragments suivants sans retirer leur volume du champ VOF.

Ce comportement est incompatible avec un spray continu : un fragment détaché doit pouvoir être transféré lorsque son état satisfait les filtres, sans réutiliser le même fragment ni bloquer les fragments apparus plus tard. La correction robuste doit être **par pas de temps et par identifiant de fragment**, avec une politique claire pour les fragments qui restent présents plusieurs pas. Réarmer simplement un booléen par pas peut réinjecter le même fragment si la consommation échoue ou si `consumeAlpha` est désactivé ; un ensemble d’identifiants ou une transition confirmée est donc préférable.

### C++-02 : source compressible non conservative entre les deux phases

La surcharge `addSup(alpha,rho,eqn)` calcule la source avec `rho[cell]`, c’est-à-dire la densité associée à l’équation cible. Pour une conversion de liquide vers l’autre phase, le terme retiré et le terme ajouté doivent représenter exactement la même masse :

> `S_m = (alpha_liquid rho_liquid) / deltaT`

La phase liquide doit recevoir `-S_m` et l’autre phase `+S_m`. Utiliser `rho` du champ cible produit `rho_water` d’un côté et `rho_air` de l’autre dans un cas eau–air ; la conservation de masse discrète est alors fausse, même si les dimensions sont correctes. Le test compressible actuel ne déclenche pas suffisamment cette situation pour la détecter automatiquement.

### C++-04 : énergie armée avant confirmation de création

Le terme d’enthalpie est armé lorsque la détection trouve des fragments, avant que le cloud ait confirmé l’ajout des parcels. Si la position est invalide, si un filtre de masse intervient ou si le modèle d’injection échoue, le carrier peut perdre `rho h` sans parcel correspondant. Le contrat recommandé est de faire retourner à l’injecteur une information de transfert confirmé, ou de calculer la source énergétique à partir de la masse réellement créée par le cloud.

### C++-05 : index de cellule et coordonnées

Le centroid d’un fragment peut se trouver hors d’une cellule valide dans des géométries fortement non orthogonales, près d’une frontière ou après un changement topologique. Le code doit vérifier le résultat de `findCellAtPosition` avant d’utiliser `cells_[fragmentI]`. Un fragment non localisable doit être rejeté explicitement et ne doit pas armer une consommation correspondante.

### TEST-01 et TEST-02 : couverture et exécution des tests

Les tests Python vérifient correctement plusieurs propriétés de l’extraction hors ligne : volume pondéré par `alpha`, centroïde, vitesse moyenne, filtres, doublons et lecture ASCII. Ils ne vérifient pas le couplage temporel C++ ni la répétition des conversions. Il manque au minimum un test OpenFOAM avec deux fragments séparés apparaissant à deux pas distincts, un test avec `rho1 != rho2`, un test d’échec de localisation et un test thermoCloud avec source d’enthalpie non nulle.

La commande `pytest -q test/test_vof_to_dpm.py` échoue actuellement avant exécution des tests avec `ModuleNotFoundError: No module named 'vof_to_dpm'`. Le problème est reproductible dans l’environnement audité. Il faut soit importer `foampilot.utilities.vof_to_dpm`, soit exposer explicitement un module compatibilité, soit corriger le packaging et la configuration pytest.

## Validations exécutées

| Validation | Résultat |
|---|---:|
| Compilation `libincompressibleVoFClouds.so` OpenFOAM 13 | **PASS** |
| Compilation `libcompressibleVoFClouds.so` OpenFOAM 13 | **PASS** |
| `incompressibleVoFCloudsDamBreak` | **PASS** |
| `compressibleVoFCloudsDamBreak` sur l’état commité | **PASS** |
| `vofToDpmParcelInBox` | **PASS** |
| `vofToDpmSingleCell` | **PASS** |
| `sprayCrossFlow` avec post-traitement | **PASS** |
| Détection répétée de fragments dans le spray | **PASS** |
| Parcels successifs après correction expérimentale non committée | `819` en fin de run |
| Bilan local masse–volume du premier fragment | erreur relative `0.0` |
| Tests Python sans configuration de chemin | **FAIL à la collecte** |
| `git diff --check` et arbre après nettoyage | **PASS** |

Le run spray avec la correction expérimentale de rafraîchissement par pas a produit 819 parcels et a conservé un bilan local exact pour le premier fragment. Cette correction a été retirée de l’arbre avant la fin de l’audit car le run compressible associé a déclenché une instabilité sévère : `alpha.water` est devenu fortement négatif, le nombre de Courant a dépassé `1112`, puis le calcul a terminé par une exception flottante dans la thermo H2O. Cela ne prouve pas que le réarmement par pas est la seule cause, mais cela prouve que cette modification ne doit pas être intégrée sans une refonte du séquencement des sources et des tests dédiés.

## Plan de correction recommandé

La première étape doit être de refondre la transaction fragment→parcel : détecter un lot, créer les parcels, confirmer la masse et seulement ensuite consommer exactement le même volume ou la même masse. La transaction doit posséder un identifiant de lot et être réinitialisée après application complète des sources aux équations concernées.

La deuxième étape doit corriger la source compressible en utilisant systématiquement la densité de la phase liquide pour les signes positif et négatif. Cette correction doit être testée avec deux densités volontairement distinctes et un calcul de volume intégré des deux phases.

La troisième étape doit déplacer la vérification de l’index de cellule dans l’injecteur et ajouter des tests C++ ou des cas OpenFOAM qui couvrent deux lots successifs, un fragment filtré, une frontière et un maillage parallèle. Enfin, les tests Python doivent être rendus exécutables depuis une installation propre du paquet.

## Décision d’audit

Le portage peut être classé **prototype validé sur cas simples**, mais pas encore **production-ready pour spray continu compressible/thermique**. Les validations actuelles démontrent la compilation, la détection et une conversion locale correcte dans le cas nominal ; elles ne démontrent pas encore une conservation globale répétée sur une séquence de fragments ni une conservation compressible robuste lorsque les densités de phases sont différentes.

## Références

[1]: https://github.com/stevendaix/foampilot/tree/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm "Portage VOF-to-DPM OpenFOAM 13 dans foampilot"
[2]: https://github.com/stevendaix/foampilot/blob/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/applications/common/vofFragmentInjection.C "Injecteur vofFragmentInjection"
[3]: https://github.com/stevendaix/foampilot/blob/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds/compressibleVoFClouds.C "Modèle compressibleVoFClouds"
[4]: https://github.com/stevendaix/foampilot/blob/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds/incompressibleVoFClouds.C "Modèle incompressibleVoFClouds"
[5]: https://github.com/stevendaix/foampilot/tree/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/test/openfoam13 "Cas de validation OpenFOAM 13"
