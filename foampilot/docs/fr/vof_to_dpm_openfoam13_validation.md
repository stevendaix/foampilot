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

Cette étape couvre le cas incompressible mono-fragment. Le chemin compressible est maintenant étendu au transfert thermoCloud : la masse convertie est identifiée par la phase liquide et le puits d’enthalpie correspondant est ajouté à l’équation `he` au même pas temporel.

## Incrément compressible alphaRho

Le fvModel compressible déclare désormais les paires de champs attendues par OpenFOAM 13 (`alpha1/rho1` et `alpha2/rho2`) et applique un transfert de masse volumique de phase dans les cellules des fragments. Le cas de validation active `vofFragmentInjection` avec `rhoLiquid 2526` et `consumeAlpha true`.

| Vérification compressible | Résultat |
|---|---|
| Compilation de `libcompressibleVoFClouds.so` avec l’injecteur enregistré | PASS |
| Création d’un parcel depuis le fragment VOF | PASS |
| Masse introduite | `1.63205`, cohérente avec `2526 × 0.000646099` |
| Application des sources aux deux phases | PASS, `alpha.water` et `alpha.air` |
| Intégrale finale de `alpha.water` | `0` |
| Détection après conversion | `1` fragment au premier pas, puis `0` |

Le transfert d’énergie/enthalpie compressible est désormais raccordé à l’API OpenFOAM 13. Lorsque `thermoCloud true` est sélectionné, le modèle construit le cloud thermo avec la phase liquide configurée, déclare le champ `he` dans `addSupFields()` et surcharge `addSup(alpha,rho,he,eqn)`. Le terme `clouds.Sh()` couvre les échanges thermoCloud, tandis qu’un puits explicite `-d(alpha rho)/dt × he_liquide` retire l’enthalpie de la masse VOF convertie exactement une fois.

## Initialisation thermique optionnelle du parcel

L’injecteur VOF initialise désormais `ThermoParcel::T()` lorsqu’un champ `T` est présent dans l’objectRegistry et que le type de parcel fournit cet accesseur. L’appel est protégé par une résolution SFINAE : les clouds momentum-only qui ne disposent pas de `T()` continuent de compiler et de s’exécuter sans modification de leur comportement.

La compilation et les cas incompressible et compressible actifs ont été rejoués sous OpenFOAM 13 après cette modification. Les quatre validations du dépôt passent : `compressibleVoFCloudsDamBreak`, `incompressibleVoFCloudsDamBreak`, `vofToDpmParcelInBox` et `vofToDpmSingleCell`. Un cas thermoCloud multicomposant court a également été exécuté jusqu’à `0.01 s` : un fragment produit un parcel de masse `1.63205`, sa température est initialisée à `300 K`, le solveur résout `T`, et le journal contient `Applied compressible enthalpy transfer to h.water`.

## Exemple spray cross-flow

Pour rapprocher la validation du cas d’usage spray, un exemple `examples/openfoam13/vof_to_dpm/example/sprayCrossFlow` a été ajouté. Il reprend la géométrie STL et la configuration cross-flow du dépôt [atomizationFoam](https://github.com/imfd-stroemungsmechanik/atomizationFoam), référence explicitement orientée vers l’atomisation VOF–Lagrangienne [3], mais remplace le solveur monolithique ancien par `foamRun -solver incompressibleVoF` et le modèle `incompressibleVoFClouds` d’OpenFOAM 13.

Le script `Allrun` construit une copie temporaire, exécute `blockMesh`, `snappyHexMesh` et le solveur jusqu’à `0.01 s`. Le calcul cross-flow passe sous OpenFOAM 13, détecte un fragment liquide avec un volume convertible atteignant environ `2.42e-05 m3` et crée un parcel de masse `0.00679141 kg`. Le journal se termine par `End` sans erreur fatale.

Cette validation a révélé et corrigé un point important pour les sprays : `vofFragmentInjection` ne doit pas mémoriser définitivement une détection vide au premier pas. Le modèle réarme désormais son analyse lorsqu’aucun fragment n’est encore présent, ce qui permet d’injecter les gouttelettes qui apparaissent plus tard après l’entrée et la fragmentation du jet.

L’exemple est exécutable avec :

```sh
cd examples/openfoam13/vof_to_dpm/example/sprayCrossFlow
./Allrun
```

[3]: https://github.com/imfd-stroemungsmechanik/atomizationFoam "atomizationFoam — 3D-coupling VOF/Lagrangian pour l’atomisation de sprays"

## Preuve par post-traitement du cas spray

Le cas `sprayCrossFlow` inclut désormais une vérification post-traitée indépendante dans `postprocess.py`. OpenFOAM écrit l’intégrale `volIntegrate(alpha.water)` dans `postProcessing/liquidVolume/0/volFieldValue.dat`. Le script lit cette série, extrait du journal le premier volume de fragment et la masse du parcel, puis vérifie localement la relation conservatrice `m_parcel = rho_liquid V_fragment`.

Le dernier run OpenFOAM 13 produit `87` points temporels jusqu’à `0.01 s`, détecte un premier fragment de volume `2.74787479346e-06 m3`, et crée un parcel de masse `0.00274787479346 kg` avec `rho_liquid = 1000 kg/m3`. L’erreur relative du transfert est `0.0` dans le rapport JSON. Le solveur termine par `End` et la figure `spray_liquid_volume.png` montre l’entrée progressive de liquide dans le domaine.

Ce contrôle est volontairement séparé du bilan global : le domaine est ouvert et reçoit du liquide par la buse, de sorte que le volume VOF total n’est pas une constante. Il vérifie donc la conversion locale fragment→parcel, tandis que la série volumique et les statistiques du cloud permettent de contrôler l’évolution globale du cas.
