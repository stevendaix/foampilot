# Rapport de consolidation VOF-to-DPM — PR #24 et PR #32

**Date de l’audit :** 27 août 2026
**Dépôt :** `stevendaix/foampilot`
**Branche de consolidation locale :** à créer après revue de ce rapport
**Décision :** ne pas merger les deux PR séparément sans décision explicite sur la stratégie finale

## 1. Conclusion exécutive

Les PR #24 et #32 constituent bien deux étapes d’un même chantier VOF-to-DPM. La PR #24 porte le socle de transition conservatrice : détection des fragments, conversion volume-masse, création initiale de parcels, confirmation atomique, tests de régression et documentation OpenFOAM 13. La PR #32 est une extension de ce socle : Direct Commit, réconciliation MPI distribuée, identité indépendante de la décomposition, namespacing multi-cloud, confirmations par espèces et cas de validation multi-cloud.

L’analyse Git montre que la tête de la PR #24 est un ancêtre direct de la tête de la PR #32. La PR #32 contient donc l’historique de #24 et ajoute cinq commits ultérieurs. Il ne faut pas merger #24 puis #32 comme deux branches indépendantes : cela créerait une séquence redondante et rendrait l’intention de consolidation moins claire. La stratégie recommandée est de traiter la branche de #32 comme la branche d’intégration du chantier, puis de fermer ou de remplacer #24 après validation des invariants.

Cette conclusion est une conclusion de structure Git ; elle ne signifie pas encore que toutes les fonctionnalités sont qualifiées en production. Le scénario multi-cloud/multi-species actuellement validé est `NP=2`. La qualification `NP=4`, le chemin incompressible exécuté de bout en bout, l’évaporation et les réactions restent à compléter.

## 2. Références Git figées

| Référence | Valeur |
|---|---|
| `origin/main` au moment de l’audit | `20bdeb9b9fdb9c93b41d7fe37630732217a36daf` |
| Tête PR #24 | `dbf8dbc7acb7bdd316128ebe60207625b38de5cb` |
| Tête PR #32 | `50d15d38f91e30fd9bcbfa6d711f259e01c50352` |
| Branche PR #24 | `feat/vof-to-dpm-conservative-transition` |
| Branche PR #32 | `feat/vof-to-dpm-direct-commit` |
| Relation | La tête de #24 est ancêtre de la tête de #32 |
| Écart #24...#32 | `0` commits uniquement dans #24, `5` commits supplémentaires dans #32 |
| État PR #24 | ouverte, mergeable, état GitHub `CLEAN` |
| État PR #32 | ouverte, mergeable, état GitHub `CLEAN` |

## 3. Historique fonctionnel

Les commits propres à la PR #24 établissent le socle suivant :

| Fonctionnalité | Commit représentatif | Rôle |
|---|---|---|
| Plan de transition conservatrice | `357fcd4` | Définit le contrat de conversion |
| Tests OpenFOAM 13 | `bb02cd3` | Établit le premier cas de validation |
| Identifiants de fragments | `ca8687e5` | Stabilise l’identité locale/globale |
| Détection incompressible | `48b98d2` | Ajoute le chemin incompressible |
| Conservation du volume | `aeab7ebc`, `f32c6c4` | Évite la perte de phase VOF |
| Initialisation thermo | `7eb8a1d2`, `3e8d604a` | Ajoute température et enthalpie |
| Injection de parcels | `bbdef648` | Crée un parcel par fragment |
| Atomicité et répétition | `3fde8ec`, `a1635b8` | Évite les doubles conversions |
| Régression énergie thermo | `588d4e5` | Ajoute le contrôle énergétique |
| État d’implémentation | `dbf8dbc` | Documente le socle |

Les cinq commits supplémentaires de #32 sont :

| Commit | Extension |
|---|---|
| `7e2ef9a` | Direct Commit MPI-safe et patch API OpenFOAM 13 |
| `4ba5348` | Validation namespacée multi-cloud/multi-species |
| `417c06b` | Documentation théorique complète |
| `0d0aed6` | Réconciliation du statut de validation et des limites |
| `50d15d3` | Diagramme de séquence MPI Mermaid et rendu PNG |

## 4. Fichiers communs et extensions de #32

Les deux PR partagent le cœur des fichiers suivants :

| Zone | Fichiers communs ou fonctionnellement correspondants |
|---|---|
| Injection | `applications/common/vofFragmentInjection.C`, `vofFragmentInjection.H` |
| Détection | `applications/common/vofFragmentTransition.H` |
| Modèle compressible | `compressibleVoFClouds.C`, `.H`, `Make/files`, `Make/options` |
| Modèle incompressible | `incompressibleVoFClouds.C`, `.H`, `Make/files`, `Make/options` |
| Enregistrement des modèles | `vofFragmentInjectionModels.C` dans les deux applications |
| Exemple spray | `example/sprayCrossFlow/**` |
| Tests de base | `compressibleVoFCloudsDamBreak`, `compressibleVoFCloudsThermoDamBreak`, `incompressibleVoFCloudsDamBreak` |
| Python | `src/foampilot/utilities/vof_to_dpm.py`, `test/test_vof_to_dpm.py` |
| Documentation | fichiers d’état, audit, validation et configuration OpenFOAM 13 |

Les fichiers introduits ou substantiellement étendus par #32 sont :

| Zone | Fichiers |
|---|---|
| Réconciliation | `vofFragmentTransitionManager.C`, `.H` |
| Confirmations | `vofLocalConfirmationStore.H` |
| Patch framework | `patches/openfoam13/commitDirect.patch`, `README.md`, `extend_multicloud_api.py` |
| Cas multi-cloud | `test/openfoam13/multiCloudMultiSpeciesMPI/**` |
| Audits | `analyze_mpi_transition_logs.py`, `analyze_thermo_conservation.py`, `analyze_multi_cloud_species.py` |
| Scripts MPI | `run_thermoDamBreak_parallel.sh`, `run_thermoDamBreak_sequential.sh` |
| Tests manager | `tests/manager/vofFragmentTransitionManagerTest.C` |
| Correctif MPI | `tools/fix_global_cell_ids_broadcast.py` |
| Documentation nouvelle | `vof_to_dpm_theory_and_method.md`, `vof_to_dpm_mpi_sequence.mmd`, `.png` |

Le diff direct entre les têtes de #24 et #32 contient 31 chemins modifiés, avec environ 4 581 lignes ajoutées et 141 supprimées. La majorité correspond à des extensions de #32, pas à une seconde implémentation indépendante du socle de #24.

## 5. API retenue

Le contrat de base retenu est celui de #24 : un fragment VOF doit être détecté, quantifié, converti au plus une fois et consommé uniquement après confirmation de la création du parcel.

#32 complète ce contrat avec une API Direct Commit framework :

```cpp
virtual bool commitDirect
(
    const directParcelData& data,
    const label ownerProc
) = 0;
```

La donnée de commit contient l’identité du cloud, la cellule locale, la position, la vitesse, le diamètre, la densité, la température, le facteur `nParticle` et les fractions massiques. Le dispatch de `parcelCloudList` sélectionne le cloud par nom et ne doit pas retomber silencieusement sur `clouds_[0]` en configuration multi-cloud.

L’identité de transaction retenue est :

```text
(cloudName, alphaFieldName, fragmentId)
```

Le namespacing des champs et des stores suit :

```text
namespaceKey = cloudName + "." + alphaFieldName
```

Le Direct Commit est local au rang propriétaire. Les opérations de détection, de fusion des fragments et de réconciliation des confirmations utilisent des collectives MPI dans un ordre strictement identique sur tous les rangs.

## 6. Invariants conservés entre #24 et #32

Les invariants du socle de #24 doivent rester non négociables dans la branche consolidée :

| Invariant | Contrat |
|---|---|
| Volume VOF | Le volume consommé est égal au volume transféré |
| Masse | `M_prepared = M_created = M_confirmed` |
| Atomicité | Une conversion rejetée ne consomme pas le VOF |
| Idempotence | Une même transaction ne crée pas deux parcels |
| Énergie | L’enthalpie créée et confirmée suit le même fragment |
| MPI | Un seul rang propriétaire effectue l’insertion |
| Espèces | Chaque composante `m_i = m Y_i` est conservée |
| Identité | L’identifiant reste stable avec une autre décomposition |
| Multi-cloud | Une transaction cible un seul cloud nommé |

#32 ajoute les contrôles de clé composite, les masses d’espèces dans les confirmations et l’audit multi-cloud. Ces extensions doivent être vues comme un renforcement du contrat #24, et non comme un contrat différent.

## 7. Validation réellement exécutée

Le cas `multiCloudMultiSpeciesMPI` a été exécuté avec `NP=2`. Les résultats observés sont :

| Contrôle | Résultat |
|---|---:|
| Deux clouds initialisés | Oui |
| Deux composants `H2O`, `C2H5OH` | Oui |
| `waterCloud` commit/confirmation | Une fois |
| `fuelCloud` commit/confirmation | Une fois |
| Masse par espèce | Audit réussi |
| Fallback vers cloud par défaut | Non détecté |
| Erreur MPI/FPE | Aucune |
| Fin du solveur | `End` |
| Audit nominal | `pass=true` |

Les valeurs sont calculées avec les fractions configurées dans le fvModel. Cette validation couvre le transport transactionnel et la comptabilité des espèces ; elle ne constitue pas encore une validation complète de l’évaporation, de la réaction ou d’un transport species thermo-réactif.

Les validations suivantes restent obligatoires avant une déclaration de couverture complète :

| Validation | État |
|---|---|
| Socle conservatif #24 | Présent dans l’ascendance de #32 ; tests historiques disponibles |
| Direct Commit thermo MPI | Validé sur le cas nominal |
| Deux clouds, deux espèces, `NP=2` | Validé |
| Deux clouds, deux espèces, `NP=4` | À exécuter |
| Décomposition différente | À exécuter et comparer bit-à-bit ou avec tolérance définie |
| Chemin incompressible multi-cloud | Compilation effectuée ; exécution complète à ajouter |
| Double commit négatif | À ajouter dans la matrice automatisée |
| Cloud inconnu | À ajouter dans la matrice automatisée |
| Fractions dont la somme est différente de un | À ajouter dans la matrice automatisée |
| Évaporation/réaction | Hors périmètre actuel |

## 8. Reproductibilité et nettoyage

Les cas consolidés doivent rester indépendants de l’environnement de développement. Aucun code ou script ne doit exiger un chemin fixe `/home/...` ou `/opt/...`. Le chemin OpenFOAM doit être fourni par l’environnement standard, par exemple `WM_PROJECT_DIR`, ou détecté avec une erreur explicite si absent.

Chaque cas final doit contenir :

| Fichier | Responsabilité |
|---|---|
| `README.md` | Prérequis, versions, commandes et limites |
| `Allrun` ou `Allrun.parallel` | Lancement reproductible |
| `Allclean` | Suppression contrôlée des résultats générés |
| `manifest.json` | Paramètres, nombre de rangs et tolérances |
| `expected/` | Références analytiques indépendantes |
| `analyze_*.py` | Audit avec code retour non nul en cas d’échec |

Les artefacts suivants ne doivent pas être versionnés : `.pyc`, `.dep`, `lnInclude` générés, objets, bibliothèques compilées, logs bruts et résultats temporaires. Les seuls rendus conservés sont ceux explicitement utiles à la documentation, comme le PNG du diagramme de séquence.

## 9. Stratégie de consolidation recommandée

La branche de #32 doit servir de base technique de consolidation, car elle contient déjà #24 dans son historique. Il faut éviter un cherry-pick mécanique de #24 sur #32. La consolidation doit plutôt effectuer une revue fonctionnelle des conflits conceptuels, puis réorganiser si nécessaire les modules en six responsabilités : extraction VOF, conservation, transition, Direct Commit, réconciliation MPI et validation.

Aucun merge ne doit être proposé avant que les contrôles suivants soient documentés : comparaison des fichiers communs, absence de fonctionnalité #24 supprimée sans remplacement, compilation des applications, test `NP=2`, test `NP=4`, comparaison de décomposition et validation des cas négatifs.

La décision de fermeture de #24 peut être prise après que #32 contient le rapport de consolidation et que son historique est compris par les mainteneurs. Cette fermeture ne doit pas être faite automatiquement dans le cadre de cet audit.

## 10. Travaux restants

Le travail prioritaire est d’exécuter `NP=4` et une décomposition différente, puis de comparer les clés, les propriétaires, les masses totales, les masses par espèce et l’enthalpie. Ensuite, il faut rendre les cas entièrement autonomes en ajoutant `Allclean` et un manifeste de paramètres, puis renforcer les tests négatifs.

Enfin, le patch framework doit être vérifié contre une installation propre d’OpenFOAM 13. Le patch ne doit pas dépendre d’une bibliothèque déjà reconstruite dans `/opt/openfoam13`, et la documentation doit indiquer précisément quels fichiers framework sont modifiés et quelle commande de compilation est nécessaire.

## 11. Décision proposée

La recommandation est de conserver **une seule ligne d’intégration**, basée sur la branche de la PR #32, tout en considérant #24 comme le socle historique et fonctionnel. La PR #32 doit être revue comme la PR de consolidation finale, avec le rapport présent dans ce fichier, sans merger #24 séparément.

Cette décision limite la duplication Git et permet de vérifier les invariants conservatifs de #24 directement sur les extensions Direct Commit et MPI de #32.

## Revue du runner et de l’environnement

Le runner `Allrun.parallel` charge maintenant l’environnement de manière stricte. La priorité est `FOAM_BASHRC`, puis `WM_PROJECT_DIR/etc/bashrc`. Si aucun des deux n’est défini vers un fichier existant, le runner s’arrête avec le code `2` et un message explicite. Aucune erreur de chargement n’est ignorée avec `|| true`.

Le contrôle sans environnement confirme l’échec immédiat attendu. Dans l’environnement de test courant, l’appel direct à `/opt/openfoam13/etc/bashrc` retourne `141` avec le message `pop_var_context`; ce comportement provient du bashrc local et doit être distingué d’un échec du cas ou du solveur. Le runner strict le signale désormais au lieu de poursuivre avec un environnement partiellement initialisé.
