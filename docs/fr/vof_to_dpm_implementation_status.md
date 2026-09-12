# État de l’implémentation VOF-to-DPM — OpenFOAM 13

**Dépôt :** foampilot  
**Branche de référence :** `feat/vof-to-dpm-conservative-transition`  
**Pull Request :** [#24](https://github.com/stevendaix/foampilot/pull/24)  
**Environnement validé :** OpenFOAM Foundation 13, C++14, Ubuntu 24.04  
**Dernière mise à jour :** 26 août 2026

## Synthèse

Le portage VOF-to-DPM est maintenant **fonctionnel et validé en séquentiel et en MPI sur OpenFOAM 13** pour le périmètre nominal documenté. Il couvre l’extraction offline de fragments, la détection native de composantes VOF, la création dynamique de parcels, le couplage mécanique incompressible et compressible, ainsi que le chemin thermodynamique `thermoCloud` avec transfert d’enthalpie. Le chemin MPI de création utilise désormais un **Direct Commit** local qui contourne les callbacks collectifs de `InjectionModel`.

La conversion est organisée comme une transaction conservatrice. Le fragment est réconcilié globalement, son rang propriétaire construit directement le parcel via l’API patchée `parcelCloudList::commitDirect()`, puis la création effective est confirmée par le cycle MPI du `fvModel`. Le volume, la masse VOF et les sources eulériennes ne sont armés qu’après cette confirmation. Aucun callback d’injection contenant une collective MPI n’est utilisé dans ce chemin.

> **Statut global :** prêt pour revue et merge pour le périmètre OpenFOAM 13 nominal, y compris le cas thermoCloud MPI à deux rangs ; les extensions multi-composants, maillages pathologiques et configurations multi-cloud restent hors qualification.

## Matrice de fonctionnalités

| Fonctionnalité | État | Validation |
|---|---:|---|
| Lecture de champs OpenFOAM ASCII | **Terminée** | Tests Python ciblés |
| Extraction de composantes connectées | **Terminée** | Tests Python et cas unitaires |
| Volume pondéré `sum(alpha × V)` | **Terminée** | Bilan spray, erreur relative `0.0` |
| Centroïde, vitesse moyenne, diamètre équivalent | **Terminée** | Tests Python |
| Détection native runtime | **Terminée** | `vofFragmentInjection` |
| Identifiants déterministes par numérotation globale | **Terminée** | Tri global des centres de cellules, indépendant de la décomposition |
| Prévention des doublons par identifiant et cellules | **Terminée** | Spray multi-pas |
| Confirmation effective de création | **Terminée** | `postInject()`, mode `nParticle` |
| Consommation VOF après confirmation | **Terminée** | Cas incompressible/compressible |
| Couplage mécanique incompressible | **Terminée** | `incompressibleVoFCloudsDamBreak` |
| Couplage alpha-rho compressible | **Terminée** | `compressibleVoFCloudsDamBreak` |
| Parcels thermodynamiques `thermoCloud` | **Terminée** | Cas thermoCloud dédié séquentiel et MPI |
| Transfert d’enthalpie confirmé | **Terminée** | Une application vers `e.water` par batch |
| Numérotation globale séquentielle/MPI | **Terminée** | Rassemblement global, diffusion et cas thermoCloud MPI à deux rangs |
| Maillages fortement non orthogonaux / topologie variable | **À faire** | Durcissement de `findCellAtPosition` |
| Mélanges multi-composants thermodynamiques | **À faire** | Cas H2O monophasé uniquement |

## Architecture actuelle

Le portage est situé dans `examples/openfoam13/vof_to_dpm/`. Le convertisseur Python et l’extracteur C++ offline servent à inspecter ou préparer les fragments. Les deux `fvModel` runtime connectent le champ VOF au `parcelCloudList` OpenFOAM 13.

Le modèle `incompressibleVoFClouds` traite le transfert de volume et le retour mécanique. Le modèle `compressibleVoFClouds` traite les équations `alpha*rho`, sélectionne la densité de la phase porteuse conformément au contrat OpenFOAM 13 et active le sink d’enthalpie lorsque `thermoCloud true` est présent dans `constant/fvModels`.

L’injecteur `vofFragmentInjection` conserve un cache par pas de temps, un état de préparation et un registre des fragments confirmés. Les méthodes `prepare()`, `postInject()` et les champs de transfert séparent respectivement la préparation, la confirmation par le cloud et l’application des sources.

## Validation reproductible

Depuis la racine `foampilot` :

```bash
PYTHONPATH=src/foampilot/utilities python -m pytest -q test/test_vof_to_dpm.py
. /opt/openfoam13/etc/bashrc
cd examples/openfoam13/vof_to_dpm
wmake applications/vofToDpm
wmake applications/incompressibleVoFClouds
wmake applications/compressibleVoFClouds
```

Les cas OpenFOAM sont lancés avec leurs scripts `Allrun` :

```bash
cd test/openfoam13/vofToDpmSingleCell && ./Allrun
cd ../vofToDpmParcelInBox && ./Allrun
cd ../incompressibleVoFCloudsDamBreak && ./Allrun
cd ../compressibleVoFCloudsDamBreak && ./Allrun
cd ../compressibleVoFCloudsThermoDamBreak && ./Allrun
cd ../../../../example/sprayCrossFlow && ./Allrun
```

Le cas thermoCloud dédié vérifie l’initialisation de `thermoCloud`, la déclaration des composants liquides H2O, la création effective du parcel, l’application de la source alpha-rho, l’application unique de la source d’enthalpie à `e.water`, la fin normale du solveur et l’absence de `FOAM FATAL` ou d’exception flottante.

## Résultats de référence

| Test | Résultat |
|---|---:|
| Tests Python | `8 passed` |
| Bibliothèques C++ OpenFOAM 13 | Compilation réussie |
| Cas incompressibles et compressibles nominaux | Fin normale |
| Exemple spray | `5` parcels finaux, erreur masse-volume `0.0` |
| Cas thermoCloud MPI | `1` commit direct confirmé de `0.646099 kg`, fin `End` |
| Source d’enthalpie thermoCloud | `1` application confirmée |
| Audit thermoCloud MPI | Tous les checks conservation, fin solveur et absence d’erreur : `pass=true` |

## Limites et recommandations

La détection n’utilise plus le hash FNV-1a ni les labels locaux pour l’identité persistante. Le manager rassemble les centres des cellules sur le maître, les trie lexicographiquement selon `(x, y, z)`, attribue les indices globaux `[0, N-1]`, puis redistribue la table locale à chaque rang. Un maillage contenant deux centres exactement coïncidents est rejeté, car une numérotation géométrique serait alors ambiguë. La réconciliation des composantes traversant les frontières processor utilise ensuite ces indices globaux pour agréger les cellules et choisir un propriétaire déterministe.

La compilation OpenFOAM 13 du manager, de la détection, du chemin Direct Commit et des deux `fvModel` est validée. L’exécution MPI complète de `run_thermoDamBreak_parallel.sh` avec deux rangs atteint `End`, produit `VOF direct commit ... success=true`, confirme `0.646099 kg` et passe l’audit automatisé. Le patch framework doit être appliqué avant la compilation de l’exemple.

La localisation du centroïde par `findCellAtPosition` doit encore être durcie pour les fragments proches d’une frontière, les maillages fortement non orthogonaux et les changements topologiques. Un fragment non localisable devra être rejeté sans armer de source ni de consommation.

La validation thermoCloud actuelle utilise une composition liquide H2O monophasée. Un cas multi-composants avec bilan d’enthalpie indépendant serait nécessaire avant d’étendre la qualification aux sprays réactifs ou aux mélanges liquides complexes.

## Patch framework requis

Le Direct Commit nécessite le patch versionné `examples/openfoam13/vof_to_dpm/patches/openfoam13/commitDirect.patch`. Les instructions d’application et les limites de l’API sont décrites dans `patches/openfoam13/README.md`. Sans ce patch, le lien dynamique échoue avec un symbole `parcelCloudList::commitDirect` absent.

## Références

- [Documentation OpenFOAM Foundation 13](https://openfoam.org/version/13/)
- [Documentation Lagrangian OpenFOAM](https://doc.cfd.direct/openfoam/lagrangian/)
- [Guide VOF-to-DPM OpenFOAM 13](vof_to_dpm_openfoam13.md)
- [Audit technique OpenFOAM 13](vof_to_dpm_code_audit_openfoam13.md)
- [Pull Request #24](https://github.com/stevendaix/foampilot/pull/24)
