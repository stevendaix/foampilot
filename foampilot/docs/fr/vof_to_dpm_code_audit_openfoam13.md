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
| C++-01 | Spray continu | L’injecteur réarme son cache à chaque `timeIndex` et conserve les identifiants encore actifs pour éviter les doublons | **Moyenne** | Corrigé, confirmation de création encore à renforcer |
| C++-02 | Conversion alpha-rho | Après vérification de `alphaSuSp.C` OpenFOAM 13, l’utilisation de `rho[cell]` du champ cible est cohérente avec la transformation source alpha-rho vers source de volume | **Aucune** | Faux positif retiré |
| C++-03 | Consommation VOF | `transitionApplied_` est réarmé par pas et les lots vides ne sont plus armés | **Moyenne** | Corrigé, confirmation de lot encore à renforcer |
| C++-04 | Cohérence énergie | `energyTransferPending_` est armé par la détection, avant confirmation de création effective d’un parcel | **Élevée** | À sécuriser |
| C++-05 | Robustesse géométrique | Un échec éventuel de `findCellAtPosition` peut laisser un index de cellule invalide avant lecture de `rho_` ou de `T` | **Moyenne/élevée** | À durcir |
| TEST-01 | Couverture | Les tests Python couvrent l’algorithme hors ligne, mais pas le cycle C++ fvModel–InjectionModel ni plusieurs conversions successives | **Élevée** | À compléter |
| TEST-02 | Packaging | L’import global de foampilot requiert `pyfluids`; le test ciblé charge désormais directement le convertisseur pour rester isolé | **Moyenne** | Partiellement corrigé |
| CFG-01 | Exemple | Le cas spray doit imposer une cohérence entre `rhoLiquid` et `constantProperties.rho0` ; sinon la masse du parcel est déterminée par une densité de parcel différente de celle du liquide | **Élevée** | Corrigé dans l’exemple |

## Analyse détaillée

### C++-01 et C++-03 : conversion unique au lieu d’une conversion de spray

Dans l’état initial audité, `vofFragmentInjection.C` conservait `emitted_` après la première liste et les deux fvModels conservaient `transitionApplied_` pour toute la durée du cas. Cette limitation a été corrigée : l’injecteur invalide maintenant son cache et réarme son état à chaque `timeIndex`, tandis que les fvModels réarment la consommation par pas et ignorent les détections vides.

Le test spray OpenFOAM 13 produit désormais des lots successifs, avec environ 810 parcels en fin de calcul, sans exception flottante dans le cas incompressible. Le registre conserve les identifiants toujours présents et oublie ceux qui ont disparu. La solution reste à renforcer par une confirmation explicite de création du lot si la consommation est désactivée ou échoue.

Ce comportement est incompatible avec un spray continu : un fragment détaché doit pouvoir être transféré lorsque son état satisfait les filtres, sans réutiliser le même fragment ni bloquer les fragments apparus plus tard. La correction robuste doit être **par pas de temps et par identifiant de fragment**, avec une politique claire pour les fragments qui restent présents plusieurs pas. Réarmer simplement un booléen par pas peut réinjecter le même fragment si la consommation échoue ou si `consumeAlpha` est désactivé ; un ensemble d’identifiants ou une transition confirmée est donc préférable.

### C++-02 : vérification de la source alpha-rho

Ce point a été réexaminé contre `alphaSuSp.C` d’OpenFOAM 13. Les fvModels fournissent une source aux équations `alpha1*rho1` et `alpha2*rho2`, puis le solveur la transforme en source de fraction volumique en divisant par la densité de la phase cible. Dans ce contrat, l’utilisation de `rho[cell]` du champ cible est cohérente pour préserver le volume de phase. Le soupçon initial de non-conservation lorsque `rho1 != rho2` est donc un faux positif.

La validation pertinente doit contrôler séparément les intégrales de volume des deux phases et la masse du parcel avec `rhoLiquid`, sans remplacer la densité cible utilisée par l’API OpenFOAM.

### C++-04 : énergie armée avant confirmation de création

Le terme d’enthalpie est armé lorsque la détection trouve des fragments, avant que le cloud ait confirmé l’ajout des parcels. Si la position est invalide, si un filtre de masse intervient ou si le modèle d’injection échoue, le carrier peut perdre `rho h` sans parcel correspondant. Le contrat recommandé est de faire retourner à l’injecteur une information de transfert confirmé, ou de calculer la source énergétique à partir de la masse réellement créée par le cloud.

### C++-05 : index de cellule et coordonnées

Le centroid d’un fragment peut se trouver hors d’une cellule valide dans des géométries fortement non orthogonales, près d’une frontière ou après un changement topologique. Le code doit vérifier le résultat de `findCellAtPosition` avant d’utiliser `cells_[fragmentI]`. Un fragment non localisable doit être rejeté explicitement et ne doit pas armer une consommation correspondante.

### TEST-01 et TEST-02 : couverture et exécution des tests

Les tests Python vérifient correctement plusieurs propriétés de l’extraction hors ligne : volume pondéré par `alpha`, centroïde, vitesse moyenne, filtres, doublons et lecture ASCII. Ils ne vérifient pas le couplage temporel C++ ni la répétition des conversions. Il manque au minimum un test OpenFOAM avec deux fragments séparés apparaissant à deux pas distincts, un test avec `rho1 != rho2`, un test d’échec de localisation et un test thermoCloud avec source d’enthalpie non nulle.

La commande `pytest -q test/test_vof_to_dpm.py` passe désormais avec **8 tests**. Le test charge directement le module de conversion afin de ne pas exiger la dépendance optionnelle `pyfluids` lors de la collecte. L’import global de tout le paquet foampilot reste plus exigeant qu’un import ciblé du convertisseur.

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
| `thermoCloud` après correction multi-pas | **PASS** |
| Détection répétée de fragments dans le spray | **PASS** |
| Parcels successifs avec registre d’identifiants | `810` en fin de run |
| Bilan local masse–volume du premier fragment | erreur relative `0.0` |
| Tests Python ciblés du convertisseur | **8 passed** |
| `git diff --check` et arbre après nettoyage | **PASS** |

Le premier essai de réarmement par pas, combiné à une modification incorrecte de la densité de source compressible, a produit une instabilité sévère : `alpha.water` fortement négatif, nombre de Courant supérieur à `1112`, puis exception flottante dans la thermo H2O. La formulation de densité a été rétablie après vérification de `alphaSuSp.C`. La correction multi-pas actuelle, avec registre d’identifiants et rejet des lots vides, compile et conserve la stabilité du cas compressible nominal, du cas thermoCloud et du spray incompressible. Le spray produit plusieurs lots et le cas thermoCloud termine avec l’appel de la source d’enthalpie.

## Plan de correction recommandé

La première étape doit être de refondre la transaction fragment→parcel : détecter un lot, créer les parcels, confirmer la masse et seulement ensuite consommer exactement le même volume ou la même masse. La transaction doit posséder un identifiant de lot et être réinitialisée après application complète des sources aux équations concernées.

La deuxième étape doit ajouter un cas avec `rho1 != rho2` pour confirmer par intégrales que la transformation alpha-rho OpenFOAM 13 conserve le volume de phase, sans modifier la densité cible utilisée par l’API. La masse du parcel doit être vérifiée séparément avec `rhoLiquid`.

La troisième étape doit déplacer la vérification de l’index de cellule dans l’injecteur et ajouter des tests C++ ou des cas OpenFOAM qui couvrent deux lots successifs, un fragment filtré, une frontière et un maillage parallèle. Enfin, l’import global du paquet devrait être rendu tolérant aux dépendances optionnelles, et la couverture C++ doit être ajoutée aux tests Python ciblés.

## Décision d’audit

Le portage peut être classé **prototype avancé validé sur cas nominaux**, mais pas encore **production-ready pour spray continu compressible/thermique**. Les validations démontrent maintenant la compilation, la détection, plusieurs lots de parcels et une conversion locale correcte ; la confirmation transactionnelle de l’énergie après création effective du parcel et la gestion par identifiant de fragment restent à finaliser.

## Références

[1]: https://github.com/stevendaix/foampilot/tree/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm "Portage VOF-to-DPM OpenFOAM 13 dans foampilot"
[2]: https://github.com/stevendaix/foampilot/blob/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/applications/common/vofFragmentInjection.C "Injecteur vofFragmentInjection"
[3]: https://github.com/stevendaix/foampilot/blob/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds/compressibleVoFClouds.C "Modèle compressibleVoFClouds"
[4]: https://github.com/stevendaix/foampilot/blob/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds/incompressibleVoFClouds.C "Modèle incompressibleVoFClouds"
[5]: https://github.com/stevendaix/foampilot/tree/feat/vof-to-dpm-conservative-transition/examples/openfoam13/vof_to_dpm/test/openfoam13 "Cas de validation OpenFOAM 13"
