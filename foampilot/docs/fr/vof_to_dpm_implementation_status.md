# État de l’implémentation VOF-to-DPM — OpenFOAM 13

**Dépôt :** foampilot  
**Branche de référence :** `feat/vof-to-dpm-conservative-transition`  
**Pull Request :** [#24](https://github.com/stevendaix/foampilot/pull/24)  
**Environnement validé :** OpenFOAM Foundation 13, C++14, Ubuntu 24.04  
**Dernière mise à jour :** 25 août 2026

## Synthèse

Le portage VOF-to-DPM est maintenant **fonctionnel et validé sur les cas nominaux séquentiels OpenFOAM 13**. Il couvre l’extraction offline de fragments, la détection native de composantes VOF, la création dynamique de parcels, le couplage mécanique incompressible et compressible, ainsi que le chemin thermodynamique `thermoCloud` avec transfert d’enthalpie.

La conversion est organisée comme une transaction conservatrice. Le fragment est d’abord préparé, puis la création effective du parcel est confirmée par le hook `postInject()`. Le volume ou la masse VOF n’est consommé et les sources Euleriennes ne sont armées qu’après cette confirmation.

> **Statut global :** prêt pour revue et merge pour le périmètre OpenFOAM 13 nominal documenté ; extension parallèle, multi-composants et géométriquement pathologique encore hors couverture de régression.

## Matrice de fonctionnalités

| Fonctionnalité | État | Validation |
|---|---:|---|
| Lecture de champs OpenFOAM ASCII | **Terminée** | Tests Python ciblés |
| Extraction de composantes connectées | **Terminée** | Tests Python et cas unitaires |
| Volume pondéré `sum(alpha × V)` | **Terminée** | Bilan spray, erreur relative `0.0` |
| Centroïde, vitesse moyenne, diamètre équivalent | **Terminée** | Tests Python |
| Détection native runtime | **Terminée** | `vofFragmentInjection` |
| Identifiants déterministes FNV-1a | **Terminée** | Cas multi-pas |
| Prévention des doublons par identifiant et cellules | **Terminée** | Spray multi-pas |
| Confirmation effective de création | **Terminée** | `postInject()`, mode `nParticle` |
| Consommation VOF après confirmation | **Terminée** | Cas incompressible/compressible |
| Couplage mécanique incompressible | **Terminée** | `incompressibleVoFCloudsDamBreak` |
| Couplage alpha-rho compressible | **Terminée** | `compressibleVoFCloudsDamBreak` |
| Parcels thermodynamiques `thermoCloud` | **Terminée** | Cas thermoCloud dédié |
| Transfert d’enthalpie confirmé | **Terminée** | Une application vers `e.water` par batch |
| Validation MPI et réconciliation inter-rangs | **À faire** | Pas encore de régression dédiée |
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
| Cas thermoCloud | `1` batch confirmé de `0.646099 kg` |
| Source d’enthalpie thermoCloud | `1` application confirmée |
| Arbre Git après nettoyage | Propre |

## Limites et recommandations

La confirmation transactionnelle sécurise le chemin nominal, mais elle ne remplace pas une stratégie de redistribution MPI. Pour une utilisation parallèle, il faudra réconcilier les composantes traversant les frontières de décomposition et garantir l’unicité du commit global.

La localisation du centroïde par `findCellAtPosition` doit encore être durcie pour les fragments proches d’une frontière, les maillages fortement non orthogonaux et les changements topologiques. Un fragment non localisable devra être rejeté sans armer de source ni de consommation.

La validation thermoCloud actuelle utilise une composition liquide H2O monophasée. Un cas multi-composants avec bilan d’enthalpie indépendant serait nécessaire avant d’étendre la qualification aux sprays réactifs ou aux mélanges liquides complexes.

## Références

- [Documentation OpenFOAM Foundation 13](https://openfoam.org/version/13/)
- [Documentation Lagrangian OpenFOAM](https://doc.cfd.direct/openfoam/lagrangian/)
- [Guide VOF-to-DPM OpenFOAM 13](vof_to_dpm_openfoam13.md)
- [Audit technique OpenFOAM 13](vof_to_dpm_code_audit_openfoam13.md)
- [Pull Request #24](https://github.com/stevendaix/foampilot/pull/24)
