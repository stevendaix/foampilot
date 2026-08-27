# Revue finale du plan de modification — FoamPilot PR #23

**Statut : validé sous réserves, avec corrections intégrées dans ce document.**

## 1. Conclusion générale

Le plan proposé est bien orienté sur le fond : il traite la PR comme une contribution complète pour **OpenFOAM Foundation 13**, et non comme une simple collection de scripts expérimentaux. Il identifie correctement les composants indispensables : couche Python FoamPilot, applications C++, bibliothèque overset/inter-mailles, cas marins, scripts de build et validations.

Cependant, le plan doit être corrigé sur quatre points avant exécution : l’état de référence annoncé n’est plus à jour, le périmètre mélange des livrables fonctionnels et des artefacts de validation, la validation hydrodynamique est insuffisamment séparée de la validation logicielle, et la stratégie de relance depuis un checkout propre doit être explicitement testée.

> La PR ne doit être considérée comme complète que si un checkout propre peut compiler les composants Foundation 13, reconstruire les cas sans sorties locales préexistantes et exécuter au minimum un calcul court documenté pour chaque famille de cas.

## 2. Corrections de l’état de référence

Le numéro de PR est **#23**, et non `#231`. La branche et la base indiquées sont correctes, mais le commit annoncé doit être actualisé.

| Élément | Valeur correcte à figer |
|---|---|
| Dépôt | `stevendaix/foampilot` |
| PR | [#23 — CLEAN marine cases](https://github.com/stevendaix/foampilot/pull/23) |
| Branche | `feature/marine-pr` |
| Base | `main` / `origin/main` |
| Commit initial fonctionnel | `2c022bd` — `complete Foundation 13 marine runtime and reproducible cases` |
| Tête actuelle | `c4b5f33` — `docs: document marine Foundation 13 PR layout` |
| Compatibilité visée | OpenFOAM Foundation 13 uniquement |
| Références | `maneuveringLib`, `propeller-OpenFOAM`, `DTCMoving_Overset` |
| Validation actuelle | Tests Python et smoke tests Foundation 13 réussis ; convergence hydrodynamique encore à établir |

Le plan doit mentionner explicitement que `c4b5f33` est la tête actuelle de la branche, car le README de la PR a été ajouté après `2c022bd`.

## 3. Périmètre fonctionnel à conserver

Le périmètre fonctionnel défini dans le plan est pertinent et doit être conservé :

| Domaine | Livrables attendus |
|---|---|
| API Python | Génération de cas marins, mouvement 6-DoF, MRF, actuation disk, forces, contrôles et overset |
| `marineFoam` | Runner Foundation 13, sélection du solver modulaire et gestion du donor |
| `marineOversetProbe` | Classification, stencils, interpolation, matrices et contrainte `fvConstraints` |
| Tests C++ | `marineInterMesh*`, `marineOverset*` et cas matriciels minimaux |
| DTC | Maillage hull/background, surface libre, donor/receveur, mouvement 6-DoF et forces |
| Propeller | MRF/AMI ou couples non conformes Foundation 13, actuation disk et forces/couple |
| Turning35 | Hull, rudder, background, manœuvre et sorties hydrodynamiques |
| Reproductibilité | Runners, générateurs et scripts de nettoyage/reconstruction |

## 4. Point critique : distinguer sources et artefacts

La PR doit versionner les fichiers nécessaires à la reconstruction, mais exclure les sorties générées. Les fichiers suivants sont des sources ou des entrées légitimes :

- fichiers `.C`, `.H`, `Make/files` et `Make/options` ;
- scripts `Allrun`, `Allmesh`, `Allclean`, `build_*.py`, `prepare_*.py` et `export_*.py` ;
- dictionnaires `system/*` et `constant/*` nécessaires au cas ;
- champs initiaux `0/*` lorsqu’ils sont réellement des entrées du cas ;
- géométries STL nécessaires et documentées ;
- cas minimaux de tests C++ ;
- documentation et rapports synthétiques.

Les fichiers suivants ne doivent pas être inclus dans une PR reproductible :

- `constant/polyMesh/*` produit par le maillage ;
- temps calculés comme `1e-05`, `0.001`, `0.002`, etc. ;
- `postProcessing/*` ;
- `log.*`, `OpenFOAM.out`, fichiers `Make/linux*`, `lnInclude`, `.o` et `.dep` ;
- stencils massifs générés automatiquement lorsqu’un script permet de les recalculer ;
- champs dérivés comme `C`, `Ccx`, `Ccy`, `Ccz`, `phi`, `rAU` et `alphaPhi.water` lorsqu’ils ne sont pas des conditions initiales ;
- maillages ou fichiers `.eMesh` produits par `surfaceFeatures`.

Cette séparation doit être vérifiée avec `git ls-tree` et non uniquement avec le répertoire de travail local.

## 5. Architecture technique à figer

### 5.1 Couche Python

La couche Python doit rester responsable de la description et de l’écriture des cas. Elle ne doit pas supposer que Foundation 13 dispose d’un solver ou d’un dictionnaire OpenCFD. Les noms de champs, les propriétés physiques, les conditions limites et les commandes doivent être validés contre les fichiers effectivement disponibles dans l’environnement Foundation 13.

La méthode `OpenFOAMDictAddFile.write_raw` doit rester couverte par un test de régression, car elle permet de préserver les blocs Foundation 13 sans les réinterpréter de manière destructive.

### 5.2 Couche C++

Les sources C++ doivent être compilables depuis un checkout propre avec un environnement Foundation 13 chargé. Chaque bibliothèque ou application doit posséder ses propres `Make/files` et `Make/options`, et le README doit indiquer l’ordre de compilation.

L’architecture donor doit être documentée sans ambiguïté : dans le cas DTC actuel, `hull` est la région receveuse et `background` la région donor. Si `marineFoam` accepte seulement une région interne du même cas, le runner doit construire cette représentation avant le calcul ; un chemin frère ne doit pas être présenté comme supporté tant qu’il ne l’est pas réellement.

### 5.3 Maillage

Le maillage doit être reconstruit par les outils Foundation 13 disponibles : `blockMesh`, `surfaceFeatures`, `refineMesh` et `snappyHexMesh`. cfMesh v2406 peut rester documenté comme option non retenue, mais ne doit pas être une dépendance obligatoire de la PR.

Le patch `hull` doit être créé par `snappyHexMesh` à partir de la géométrie STL et vérifié dans `constant/polyMesh/boundary`. La présence du nom dans un dictionnaire ne suffit pas.

## 6. Séquencement recommandé par commits

Le plan original doit être découpé en commits indépendants et vérifiables :

### Commit A — API FoamPilot marine

Inclure les modules Python de cas, mouvement, contrôles, forces, MRF, actuation disk et overset, ainsi que les tests unitaires associés. Critère : suite Python marine réussie sans dépendre d’un maillage local.

### Commit B — Runtime C++ Foundation 13

Inclure `marineFoam`, `marineOversetProbe`, les fichiers `Make` et les tests C++ minimaux. Critère : compilation propre avec Foundation 13 et exécution des harnesses matriciels/inter-mailles.

### Commit C — Cas DTC reproductible

Inclure le générateur FoamPilot, les géométries nécessaires, les dictionnaires sources et les runners. Critère : reconstruction depuis un dossier vide, création du patch `hull`, `checkMesh` valide, `setFields` réussi et calcul court sans erreur fatale.

### Commit D — Cas propeller reproductible

Inclure les dictionnaires MRF/AMI ou couples non conformes, la géométrie source, le runner de maillage, le solver Foundation 13 effectivement disponible et le post-traitement des forces/couples. Critère : au moins un calcul court et une sortie `forces` lisible.

### Commit E — Turning35

Le cas Turning35 n’est pas encore présent sous un chemin Foundation 13 dédié dans la branche actuelle. Il doit constituer un commit séparé, avec les géométries hull/rudder/background, le mouvement de manœuvre, les contrôles et le runner. Critère : calcul court du cas assemblé et sorties de forces/moments/trajectoire. Tant que ce commit n’est pas réalisé, la PR ne peut pas être annoncée comme reproduisant les trois cas de référence.

### Commit F — Documentation et validation

Inclure le README, la matrice des références, les rapports et les commandes de reproduction. Critère : les commandes documentées correspondent aux chemins présents dans la PR.

## 7. Critères de validation obligatoires

La validation doit être présentée en trois niveaux, qui ne doivent pas être confondus.

| Niveau | Question | Critères |
|---|---|---|
| Structurel | Le cas est-il complet ? | Fichiers présents, dictionnaires lisibles, champs cohérents, patches attendus |
| Numérique | Le calcul est-il stable ? | Pas de `FOAM FATAL ERROR`, résidus, continuité, bornes `alpha.water`, pas de NaN |
| Physique | Le résultat est-il crédible ? | Forces/moments non nuls, mouvement 6-DoF cohérent, convergence temporelle/maillage, comparaison à la référence |

Pour le DTC, il faut vérifier `alpha.water` sous et au-dessus de la surface libre, la conservation de masse, `yPlus` près du hull, l’évolution de la position et de l’assiette, puis les contributions de pression et de cisaillement dans `rigidBodyForces`.

Pour le propeller, il faut vérifier la rotation effective, la zone rotor, les interfaces AMI ou couples non conformes, la poussée, le couple et leur stabilisation sur plusieurs tours. Un unique pas donnant une force non nulle constitue seulement un smoke test.

Pour Turning35, il faut vérifier la trajectoire, l’angle de lacet, les forces latérales, les moments et la cohérence de la manœuvre avec les degrés de liberté attendus.

## 8. Ordre d’exécution corrigé

L’ordre recommandé est le suivant :

```bash
# 1. Environnement
source /opt/openfoam13/etc/bashrc

# 2. Compilation
cd openfoam13/marineOversetProbe
wmake
cd ../marineFoam
wmake

# 3. Tests Python
cd ../../foampilot
PYTHONPATH=src pytest -q

# 4. Cas DTC
cd ../openfoam13
python3 build_realistic_dtc_foampilot.py
cd FoamPilotCases/DTCRealisticFoundation13
./Allmesh.FoamPilot
setFields
marineFoam -solver incompressibleVoF
postProcess -func rigidBodyForces

# 5. Propeller
cd ../propellerFoundation13
./Allmesh.FoamPilot
marineFoam -solver incompressibleVoF
postProcess -func forces

# 6. Turning35 — à activer après livraison du cas Foundation 13
# Le chemin `openfoam13/Turning35Foundation13` n’est pas encore présent dans la branche actuelle.
# Une fois le cas ajouté :
# cd ../../Turning35Foundation13
# ./Allmesh.FoamPilot
# marineFoam -solver incompressibleVoF
```

Chaque runner doit commencer par vérifier que Foundation 13 est chargé, nettoyer les sorties précédentes et arrêter le pipeline dès qu’une commande échoue. Les commandes exactes doivent être ajustées si le cas Turning35 n’est pas encore présent sous ce chemin.

## 9. Risques et décisions à documenter

Le portage cfMesh direct contre Foundation 13 ne doit pas être présenté comme terminé. La stratégie retenue est de privilégier les outils natifs Foundation 13 et de conserver Gmsh ou cfMesh uniquement comme outils externes de préparation si cela est nécessaire.

Le solver à documenter pour la chaîne VOF est `incompressibleVoF`. Toute référence à `compressibleVoF` doit être supprimée ou explicitement marquée comme non disponible tant qu’aucun module Foundation 13 correspondant n’est compilé.

L’overset custom doit être décrit comme une implémentation expérimentale FoamPilot. Il ne faut pas laisser entendre qu’il fournit déjà toutes les propriétés d’un overset natif : conservation de flux, traitement complet des cellules hole/fringe, interpolation de pression et agrégation hydrodynamique doivent être testés séparément.

## 10. Verdict sur le plan proposé

Le plan est **validable après révision**, mais il ne doit pas être exécuté tel quel. La présente version constitue la référence de travail pour la suite de la PR. Les corrections indispensables sont :

1. remplacer `PR #231` par `PR #23` ;
2. remplacer le commit de tête annoncé `2c022bd` par `c4b5f33` ;
3. séparer les sources versionnées des artefacts locaux ;
4. ajouter un vrai test de checkout propre et de reconstruction ;
5. séparer validation structurelle, numérique et physique ;
6. faire de Turning35 un livrable explicite, au même niveau que DTC et propeller ;
7. documenter l’orientation donor/receveur et la représentation multi-région réellement supportée ;
8. conditionner la validation physique à des calculs plus longs et à des études de convergence.

Avec ces corrections, le plan devient une feuille de route réaliste pour transformer la PR #23 en contribution FoamPilot livrable, plutôt qu’en simple dépôt de résultats expérimentaux.

## Références

[1]: https://github.com/stevendaix/foampilot/pull/23 "FoamPilot PR #23"
[2]: https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials/incompressibleVoF/DTCHullWave "OpenFOAM Foundation 13 — DTCHullWave"
[3]: https://github.com/OpenFOAM/OpenFOAM-13/tree/master/tutorials/incompressibleVoF/propeller "OpenFOAM Foundation 13 — propeller"
[4]: https://github.com/balabibo/maneuveringLib "maneuveringLib"
[5]: https://github.com/skfelix/propeller-OpenFOAM "propeller-OpenFOAM"
[6]: https://github.com/myozinaung/DTCMoving_Overset "DTCMoving_Overset"
