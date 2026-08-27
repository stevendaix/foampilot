# Plan d’intégration — FoamPilot PR #23

**Statut : prêt à intégrer sous réserves documentées**
**PR :** [#23](https://github.com/stevendaix/foampilot/pull/23)
**Branche :** `feature/marine-pr`
**Base :** `main`
**Cible :** OpenFOAM Foundation 13 exclusivement

## 1. Objectif d’intégration

Cette intégration apporte à FoamPilot une chaîne marine reproductible sous OpenFOAM Foundation 13. Elle regroupe le runner modulaire `marineFoam`, la bibliothèque expérimentale `marineOversetProbe`, les stencils et tests inter-mailles, ainsi que trois familles de cas : DTC Hull overset, propeller MRF/AMI et Turning35 de manœuvre avec mouvement rigide 6-DoF.

Le plan distingue explicitement la **préparation de cas**, la **validation d’exécution** et la **validation physique**. Le merge valide l’architecture, la compilation et la reproductibilité des smoke tests ; il ne prétend pas avoir démontré la convergence hydrodynamique finale des trois références.

## 2. Préconditions d’intégration

L’environnement de validation doit charger OpenFOAM Foundation 13 avant toute compilation ou exécution :

```sh
source /opt/openfoam13/etc/bashrc
foamVersion
```

La sortie attendue doit identifier OpenFOAM 13. Les clones locaux des dépôts de référence, les maillages calculés, les répertoires `processor*`, les temps numériques et les caches Python ne font pas partie de la PR.

## 3. Séquencement recommandé

| Étape | Action | Résultat attendu | Bloquant pour le merge |
|---|---|---|---|
| A | Checkout propre de `feature/marine-pr` | Aucun artefact local ni dépendance vers un clone externe | Oui |
| B | Compilation `openfoam13/Allwmake` | Bibliothèque, `marineFoam` et harnesses compilés | Oui |
| C | Tests unitaires Python marins | Suite marine verte, sans erreur de collection | Oui pour le périmètre marine |
| D | Harnesses inter-mailles | Matrice, lecture stencil et couplage donor/receveur validés | Oui |
| E | DTC Hull | Maillage, champs, overset et smoke test exécutables | Oui |
| F | Propeller | Maillage MRF/AMI et smoke test exécutables | Oui |
| G | Turning35 | Génération FoamPilot, maillage et smoke test 6-DoF exécutables | Oui |
| H | Revue documentaire | Limites physiques et statut des cas explicitement indiqués | Oui |
| I | Merge GitHub | PR `MERGEABLE` et `CLEAN`, puis fusion manuelle ou autorisée | Après A–H |

## 4. Commandes de validation

Depuis un checkout propre :

```sh
source /opt/openfoam13/etc/bashrc
cd openfoam13
./Allwmake

cd ../foampilot
PYTHONPATH=src:.. pytest -q test/test_marine_*.py
```

Pour les tests inter-mailles :

```sh
cd ../openfoam13/marineOversetMatrixTest/case
./Allrun
```

Pour Turning35 :

```sh
cd ../../openfoam13
python3 build_turning35_foampilot.py
cd FoamPilotCases/Turning35Foundation13
./Allclean
./Allmesh.FoamPilot
setFields
marineFoam -solver incompressibleVoF
```

Le cas doit terminer par `End`, conserver `0 <= alpha.water <= 1` à la tolérance numérique près et produire des forces et moments non nuls. `Allclean` doit préserver le dossier initial `0` et supprimer uniquement les sorties calculées.

Les cas DTC et propeller suivent leurs runners respectifs ; ils doivent être exécutés séparément, car leur maillage et leur topologie inter-mailles ne sont pas interchangeables.

## 5. Critères d’acceptation technique

Le code est accepté si `Allwmake` compile toutes les cibles sans erreur sous Foundation 13, si les harnesses C++ passent, si les tests Python marins passent, et si un checkout propre régénère Turning35 sans géométrie ou clone externe. Les dictionnaires doivent employer la nomenclature Foundation 13, notamment `momentumTransport`, `physicalProperties.water`, `physicalProperties.air`, `interfaceCompression` et les signatures de `snappyHexMesh` Foundation 13.

Le runner `marineFoam` doit sélectionner le module `incompressibleVoF`, charger le mouvement de maillage, appliquer les `fvModels` et permettre le chargement optionnel d’une région donor. L’overset custom doit rester présenté comme expérimental : les tests actuels valident le contrat matriciel et l’interpolation ciblée, pas toutes les garanties d’un overset natif industriel.

## 6. Critères d’acceptation numérique

Chaque smoke test doit atteindre la fin demandée sans `FOAM FATAL ERROR`, NaN ou divergence immédiate. Les logs doivent montrer les étapes de maillage et du solver, les résidus doivent être finis, la continuité doit rester contrôlée et les champs VOF doivent rester bornés. Les forces et moments servent de preuve d’exécution, non de preuve de convergence.

Pour Turning35, le mouvement rigide Newmark et la mise à jour de la maille doivent être effectivement sélectionnés. La configuration actuelle expose le patch `hull`; le patch `rudder` STL est supprimé comme patch de taille nulle par `snappyHexMesh` au niveau de raffinement fourni. Cette limite est intentionnelle et doit rester documentée jusqu’à l’exposition séparée du gouvernail.

## 7. Validation physique post-merge

Après intégration, une campagne séparée doit comparer DTC, propeller et Turning35 aux références. Elle doit comporter au minimum une étude de pas de temps, une étude de raffinement de maillage, des historiques de forces et moments, la conservation de masse et, pour le propeller, la poussée, le couple et la stabilisation sur plusieurs tours. Pour Turning35, il faut comparer trajectoire, angle de lacet, force latérale et moment de lacet à la référence `maneuveringLib`.

Cette campagne ne doit pas bloquer le merge de l’architecture si les smoke tests et la reproductibilité sont verts, mais elle doit être suivie dans une issue ou une PR dédiée avant toute annonce de validation hydrodynamique définitive.

## 8. Risques et mesures de maîtrise

| Risque | Impact | Mesure |
|---|---|---|
| Confusion OpenFOAM.com/Foundation | Échec de lecture des dictionnaires | Vérifier `foamVersion` et maintenir Foundation 13 dans les runners |
| Reproduction depuis un workspace pollué | Faux succès ou artefacts dans la PR | Checkout propre et nettoyage systématique |
| Overset présenté comme complet trop tôt | Conclusion scientifique incorrecte | Employer le statut expérimental et conserver les harnesses ciblés |
| STL hull/rudder non-manifold ou patch nul | Forces partielles ou maillage fragile | `surfaceCheck`, `checkMesh`, documenter le patch réellement présent |
| Convergence physique non démontrée | Comparaison de référence invalide | PR/issue post-merge dédiée aux études de convergence |

## 9. Décision de merge

La PR peut être fusionnée lorsque la tête distante est la version testée, que GitHub indique `MERGEABLE` et `CLEAN`, que le checkout propre et `Allwmake` passent, et que les réserves physiques figurent dans la description. La fusion elle-même reste une action distincte de la validation technique et doit être effectuée avec l’autorisation explicite du mainteneur.

## 10. Après le merge

Le commit de merge doit être suivi d’une vérification rapide de `main`, puis d’une issue consacrée à la convergence physique et à l’exposition du patch rudder. Aucun résultat de smoke test ne doit être publié comme valeur hydrodynamique de référence sans comparaison temporelle et de maillage.

**Référence principale :** [FoamPilot PR #23](https://github.com/stevendaix/foampilot/pull/23)
