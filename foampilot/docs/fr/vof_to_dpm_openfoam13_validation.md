# Validation OpenFOAM 13 du portage VOF→DPM

## Environnement

Les essais ont été réalisés sur Ubuntu 24.04.4 LTS avec le paquet officiel `openfoam13`, installé dans `/opt/openfoam13`. La version chargée par `source /opt/openfoam13/etc/bashrc` est OpenFOAM 13. La procédure d’installation suit la documentation officielle de la Fondation OpenFOAM [1].

## Comparaison avec la référence

La référence utilisée est la branche distante `origin/feat/vof-to-dpm-converter` du dépôt `stevendaix/foampilot`, actuellement au commit `51b8fea`. La comparaison des sources sous `examples/openfoam13/vof_to_dpm/applications` ne révèle aucune différence entre cette branche de référence et la branche testée pour les composants C++. Les changements de la branche testée portent sur le plan de transition conservatif Python, ses tests et sa documentation.

Les trois composants C++ ont été compilés avec `wmake` contre les bibliothèques OpenFOAM 13 installées : `vofToDpm`, `libincompressibleVoFClouds.so` et `libcompressibleVoFClouds.so`. Aucun message `error:`, `undefined reference` ou `No such file` n’a été détecté dans les journaux de compilation.

| Cas | Branche corrigée | Branche de référence | Résultat numérique |
|---|---:|---:|---|
| `vofToDpmSingleCell` | PASS | PASS | Sortie `vofToDpmFragments` identique |
| `vofToDpmParcelInBox` | PASS | PASS | Sortie `vofToDpmFragments` identique |
| `incompressibleVoFCloudsDamBreak` | PASS | PASS | Solver, modèle et cloud sélectionnés ; évolution terminée |
| `compressibleVoFCloudsDamBreak` | PASS | PASS | Solver, modèle et cloud sélectionnés ; évolution terminée |

Dans le cas cellule unique, le fragment produit a un volume de `1`, une masse de `1000`, un diamètre équivalent de `1.2407009818`, un centroïde `(0.5 0.5 0.5)` et une vitesse `(2 0 0)`. Dans le cas `parcelInBox`, le fragment produit a un volume de `0.0001`, une masse de `0.1`, un diamètre équivalent de `0.05758823823`, un centroïde `(0.05 0.05 0.005)` et une vitesse nulle. Les fichiers de sortie sont identiques entre la branche corrigée et la référence.

## Tests Python et transition conservative

La suite ciblée a été exécutée avec :

```sh
PYTHONPATH=src:src/foampilot/utilities pytest -q test/test_vof_to_dpm.py
```

Résultat : `7 passed`. Ces tests couvrent désormais le retrait du liquide converti dans le champ résiduel, la conservation du volume total, l’absence de mutation du champ d’entrée, les doublons de cellules et les volumes de fragments incohérents.

## Interprétation

Cette validation confirme que le portage est compatible avec OpenFOAM 13 au niveau compilation et exécution, et que les sorties de référence restent inchangées lorsque le nouveau chemin n’est pas activé. Le cas incompressible dispose désormais d’un chemin runtime natif optionnel : `vofFragmentInjection` crée un parcel équivalent par fragment détecté et `consumeAlpha` transfère le volume de `alpha1` vers `alpha2`. La conservation masse-énergie dans le chemin compressible, la fusion des fragments aux frontières MPI et la gestion de plusieurs parcels par fragment restent des développements distincts à valider.

## Références

[1]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM 13 — installation Ubuntu officielle"
[2]: https://doc.cfd.direct/openfoam/lagrangian/ "CFD Direct — documentation Lagrangienne OpenFOAM"

## Incréments runtime vérifiés

Les incréments suivants ont ensuite été ajoutés et vérifiés contre l’API OpenFOAM 13 :

| Incrément | Vérification | Résultat |
|---|---|---|
| `vofFragmentInjection` | Enregistrement runtime pour `collidingCloud` et compilation de `libincompressibleVoFClouds.so` | PASS |
| Injection d’un fragment | Un fragment VOF donne exactement un parcel, avec centroïde, diamètre équivalent et vitesse du fragment | PASS |
| Bilan de masse | `mass introduced = 1.63205`, cohérent avec `rho × volume = 2526 × 0.000646099` | PASS |
| Consommation alpha activée | `foamPostProcess -func 'volIntegrate(alpha.water)' -time 0.01` retourne `0` | PASS |
| Consommation alpha désactivée | Le même volume reste `0.000646099` et le parcel est toujours injecté | PASS |

La consommation native est implémentée comme un transfert borné entre `alpha1` et `alpha2` : la perte de liquide est appliquée implicitement à `alpha1` et le gain est appliqué explicitement à `alpha2`, conformément au chemin `alphaSuSp.C` du solver `incompressibleVoF` d’OpenFOAM 13. Le test vérifie également l’absence de `FOAM FATAL ERROR`, d’erreur d’édition de liens et d’artefact de compilation suivi par Git.

Cette étape couvre le cas incompressible mono-fragment. Le chemin compressible doit encore recevoir un transfert thermodynamique cohérent : la masse convertie est déjà identifiée par la phase liquide, mais l’énergie/enthalpie et la sélection du champ alpha doivent être intégrées dans une étape dédiée avant de généraliser le cas.
