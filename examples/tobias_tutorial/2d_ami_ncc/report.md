# 2D Arbitrary Mesh Interface / Non-Conformal Coupling

## Objet du tutoriel

Ce cas illustre le raccordement de deux surfaces d’interface qui ne sont pas traitées comme une paire de faces conformes classique. La géométrie comporte une zone tournante et une zone fixe séparées par deux patches d’interface, `AMI1` et `AMI2`. Le workflow combine une génération de maillage par `snappyHexMesh`, la création de baffles, leur séparation, puis la création de couples non conformes avec `createNonConformalCouples`. La rotation est pilotée par `rotatingMotion` dans une zone de cellules dédiée.

Le cas source provient de la collection des tutoriels OpenFOAM de Tobias Holzmann [1] [2]. Le runner local est volontairement déclaratif : les dictionnaires et champs sont écrits par FoamPilot, les actifs CAD/STL sont copiés dans le répertoire de cas, et toutes les applications OpenFOAM sont lancées via l’interface `Solver.run_command`.

## Structure du portage

| Élément | Rôle |
| --- | --- |
| `run.py` | Génère le cas par FoamPilot et exécute toutes les étapes du workflow. |
| `templates.py` | Contient les dictionnaires et champs textuels issus du cas source. |
| `cad/backgroundMesh.unv` | Maillage de fond utilisé par `ideasUnvToFoam`. |
| `triSurface/*.stl` | Géométries de la région, de la zone de raffinement et de l’interface. |
| `.dynamicMeshDict` | Paramétrage de la zone tournante et de `rotatingMotion`. |
| `report.md` | Présent rapport et critères de validation. |

## Workflow exécuté

Le runner supprime d’abord l’ancien répertoire de calcul, puis écrit les fichiers OpenFOAM avec `OpenFOAMDictAddFile.write_raw`. Cette méthode est nécessaire pour reproduire fidèlement les dictionnaires complexes du cas historique, notamment leurs directives et leur structure de dictionnaire, sans générer un second en-tête `FoamFile`. Les surfaces STL et le maillage UNV sont ensuite placés dans l’arborescence attendue par les utilitaires.

La séquence exécutée avec OpenFOAM Foundation 13 est la suivante :

```text
ideasUnvToFoam cad/backgroundMesh.unv
snappyHexMesh -overwrite
changeDictionary
flattenMesh
extrudeMesh
topoSet
createBaffles -overwrite
splitBaffles -overwrite
createNonConformalCouples -overwrite AMI1 AMI2
foamRun
```

`foamRun` utilise le solveur `incompressibleFluid` déclaré dans `system/controlDict`. Le calcul est volontairement court (`endTime 0.001`) afin de fournir un smoke run reproductible qui valide la mise en données, la construction du maillage et l’initialisation du solveur sans prétendre reproduire une campagne de production longue.

## Adaptations nécessaires pour OpenFOAM 13

L’archive source annonce OpenFOAM 12 dans ses en-têtes. Trois adaptations locales ont été nécessaires :

| Adaptation | Justification |
| --- | --- |
| Ajout de `nLayers` et `expansionRatio` dans `linearNormalCoeffs` | OpenFOAM 13 recherche ces entrées dans le sous-dictionnaire du modèle `linearNormal`. |
| Correction du point-virgule manquant après `type zeroGradient` dans `0.orig/U` | Le parseur OpenFOAM 13 refuse la déclaration incomplète du patch `AMI1`. |
| Copie de `0.orig` après `createNonConformalCouples` | L’utilitaire crée les patches auxiliaires non conformes et écrit la structure finale de maillage ; les champs initiaux doivent alors être présents pour la lecture du solveur. |

Ces adaptations sont conservées dans `examples/tobias_tutorial/2d_ami_ncc/run.py`. Aucune nouvelle méthode générique n’a été ajoutée à FoamPilot pour ces différences de dialecte OpenFOAM, car elles sont propres à ce cas et ne justifient pas d’imposer une hypothèse NCC à l’API partagée.

## Résultats de validation

| Vérification | Résultat observé |
| --- | --- |
| Conversion du maillage de fond | `Read 23104 points`; `Read 16875 cells and 12150 boundary faces`; fin normale (`End`). |
| Maillage `snappyHexMesh` | 140321 cellules, 458995 faces et 178961 points ; contrôle des faces en erreur avec zéro défaut dans les catégories rapportées. |
| Extrusion | Maillage écrit dans `constant/region0` ; fin normale (`End`). |
| Couple NCC | `AMI1` et `AMI2` comportent chacun 704 faces ; couverture minimale, moyenne et maximale égale à `1/1/1` sur les deux côtés ; 1408 couplages calculés. |
| Initialisation du solveur | Sélection de `incompressibleFluid`, du mouvement `solidBody` et de `rotatingMotion`. |
| Calcul | `foamRun` se termine avec `End` après initialisation du couple non conforme. |

Le cas est donc **validé** selon le protocole du projet : le script FoamPilot recrée le cas, le maillage est généré, les couples non conformes sont construits et le solveur OpenFOAM 13 s’exécute jusqu’à la fin du smoke run.

## Limites et reproductibilité

La validation porte sur une exécution séquentielle avec OpenFOAM Foundation 13 sous Ubuntu. Elle ne constitue pas une comparaison quantitative avec une solution analytique ni une validation de convergence temporelle. Les journaux complets sont générés dans `case/log.*` après exécution et ne sont pas nécessaires à la génération manuelle du cas : il suffit de lancer `python3 run.py` depuis ce répertoire avec l’environnement OpenFOAM 13 chargé.

## Références

[1]: https://holzmann-cfd.com/community/training-cases — Tobias Holzmann, *OpenFOAM Training Cases*.
[2]: https://wiki.openfoam.com/Tutorials_by_Tobias_Holzmann — OpenFOAM Wiki, collection des cas de Tobias Holzmann.
[3]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation, installation d’OpenFOAM 13 sur Ubuntu.
[4]: https://github.com/stevendaix/foampilot/pull/17 — Pull Request FoamPilot du portage.
