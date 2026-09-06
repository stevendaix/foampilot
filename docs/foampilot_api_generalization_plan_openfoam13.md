# Plan de généralisation de l’API FoamPilot

## Objet

Ce document propose une refactorisation de l’API FoamPilot à partir de l’audit des runners de tutoriels OpenFOAM 13. L’objectif n’est pas de créer une fonction par tutoriel, mais de transformer les motifs récurrents d’intégration en opérations générales compréhensibles par un utilisateur qui construit son propre cas.

## Résultats de l’audit

L’audit porte sur les runners `run.py` présents dans `foampilot/tutorials`.

| Indicateur | Mesure | Interprétation |
|---|---:|---|
| Runners recensés | 253 à la date de l’audit | Couverture importante des tutoriels, mais hétérogénéité historique. |
| Appels `run_command` | 614 | L’API d’exécution est très utilisée, souvent comme primitive directe. |
| Imports d’assets | 186 | Le besoin est transversal et doit être porté par un gestionnaire de cas/références. |
| Imports de dictionnaires système | 151 | Motif récurrent à abstraire au niveau du cas OpenFOAM. |
| Définitions locales `import_reference_case` | 139 | Duplication forte de la préparation des cas. |
| Définitions locales `run_of13` | 64 | Duplication de l’environnement et de l’appel aux utilitaires. |
| Définitions locales `run_parallel` | 30 | Orchestration MPI répétée dans les runners. |
| Occurrences de `LD_LIBRARY_PATH` | 150 | L’environnement OpenFOAM est trop souvent géré localement. |
| Occurrences de `WM_PROJECT` | 83 | Configuration de version répétée dans les runners. |
| Runners avec opérations directes détectables | 3 | La règle FoamPilot-only n’est pas encore garantie automatiquement. |
| Runner le plus volumineux | 428 lignes | Les premiers runners concentrent beaucoup de logique de mise en données. |

## Diagnostic architectural

Le cœur FoamPilot contient principalement des primitives utiles et réutilisables. Les difficultés se situent dans la couche d’utilisation. Les runners connaissent encore directement les chemins OpenFOAM, les noms des binaires, les variables `WM_*`, la structure des répertoires et les détails d’import des fichiers `.orig` ou `.gz`.

Cette situation produit trois niveaux mélangés dans un même `run.py`:

1. **Intention utilisateur**: créer un cas, choisir un solveur, construire un maillage et lancer un calcul.
2. **Orchestration OpenFOAM**: exécuter `blockMesh`, `snappyHexMesh`, `decomposePar`, `foamRun`, `reconstructPar`, etc.
3. **Mécanique interne**: copier des arbres, décompresser des fichiers, définir `LD_LIBRARY_PATH`, appliquer des overlays et analyser les journaux.

La généralisation doit déplacer les niveaux 2 et 3 dans des composants FoamPilot, afin que le runner exprime surtout le niveau 1.

## Classement des motifs

| Catégorie | État | Action recommandée |
|---|---|---|
| Import de fichier, asset ou champ | Déjà relativement générique | Unifier sous `ReferenceCase` avec gestion des `.orig`, `.gz`, régions et destinations. |
| Gestion de dictionnaires | Fonctionnelle mais bas niveau | Ajouter une façade `case.system.dictionary(...).set(...)`, `merge(...)`, `remove(...)`. |
| Exécution d’utilitaires | Trop proche du binaire OpenFOAM | Ajouter `MeshWorkflow`, `RunWorkflow` et conserver `run_command` comme échappatoire avancée. |
| Environnement OpenFOAM/MPI | Dupliqué dans les runners | Centraliser dans `OpenFOAMEnvironment` et l’injecter automatiquement. |
| Maillage | Plusieurs primitives existent | Les regrouper dans une API fluide: `block_mesh()`, `surface_features()`, `snappy_hex_mesh()`, `extrude()`, `refine()`, `check()`. |
| Calcul parallèle | Fonctionne mais paramètres dispersés | Définir un objet de configuration MPI et automatiser décomposition, exécution et reconstruction. |
| Validation | Principalement manuelle par lecture de logs | Ajouter des assertions réutilisables sur `End`, cellules, patches, champs, résidus et erreurs fatales. |
| Opérations directes dans les runners | Trois cas détectables, risque plus large avec les variantes de fichiers | Ajouter un lint/CI FoamPilot-only. |
| Runners historiques volumineux | Difficultes de maintenance | Migrer progressivement les cas représentatifs, sans réécriture massive initiale. |

## Architecture cible

### `FoamCase`

`FoamCase` doit devenir la façade principale destinée à l’utilisateur. Il devrait gérer le répertoire du cas, le solveur, la version OpenFOAM, les régions et les sous-composants `reference`, `system`, `mesh`, `run` et `validate`.

Exemple cible:

```python
case = FoamCase("my_case", solver="fluid", openfoam=13)
case.reference.import_case("fluid/cavity")
case.mesh.block_mesh()
case.run.transient(end_time=1.0)
case.validate.no_fatal_errors()
```

### `OpenFOAMEnvironment`

Ce composant doit résoudre l’installation OpenFOAM, construire `PATH` et `LD_LIBRARY_PATH`, configurer MPI et exposer l’environnement aux utilitaires. Aucun runner ne devrait recopier les variables `WM_PROJECT`, `FOAM_APPBIN` ou `FOAM_LIBBIN`.

```python
env = OpenFOAMEnvironment(version=13, mpi=True)
case = FoamCase("case", solver="fluid", environment=env)
```

### `ReferenceCase`

Ce composant doit fournir les opérations suivantes:

```python
case.reference.import_case("incompressibleVoF/damBreak")
case.reference.overlay("system/fvSchemes", source="damBreak.orig")
case.reference.import_field("0.orig/alpha.water", destination="0/alpha.water")
case.reference.import_asset("constant/geometry/curve.obj.gz", decompress=True)
case.reference.merge("system/fvSolution", "system/fvSolution.orig")
```

Les fonctions actuelles `import_reference_file`, `import_reference_field`, `import_reference_asset`, `copy_case_tree` et `merge_reference_dictionary` doivent rester compatibles mais devenir des primitives internes de cette façade.

### `MeshWorkflow`

```python
case.mesh.block_mesh()
case.mesh.surface_features()
case.mesh.snappy_hex_mesh()
case.mesh.extrude(layers=500)
case.mesh.refine(dict_name="system/refineMeshDict")
case.mesh.create_baffles()
case.mesh.check()
```

Les arguments doivent représenter les choix de l’utilisateur. Le workflow doit conserver la possibilité de passer des arguments OpenFOAM avancés, mais ne doit pas imposer au runner de construire lui-même les chemins de binaires ou l’environnement.

### `RunWorkflow`

```python
case.run.serial()
case.run.parallel(processes=4, decompose=True, reconstruct=True)
case.run.foam_run(module="incompressibleVoF")
case.run.utility("setFields")
```

L’API doit distinguer les opérations habituelles (`foam_run`, `parallel`, `reconstruct`) de l’échappatoire bas niveau (`utility` ou `command`).

### `CaseValidation`

```python
case.validate.completed()
case.validate.no_fatal_errors()
case.validate.end_time(1.0)
case.validate.mesh(cells=44800, patches=["inlet", "outlet"])
case.validate.field("alpha.water")
```

La validation doit produire des erreurs explicites et des rapports structurés plutôt que laisser chaque runner analyser ses propres journaux.

## Plan de généralisation priorisé

### Priorité P0 — sécuriser la base

Créer un lint qui interdit dans les runners les appels `subprocess`, `os.system`, `shell=True`, `shutil`, les écritures directes de cas et les suppressions directes de fichiers. Ajouter des tests unitaires sur l’import gzip, les overlays `.orig`, l’environnement et les codes retour. Cette étape ne change pas encore l’interface publique.

### Priorité P1 — centraliser l’environnement et l’exécution

Créer `OpenFOAMEnvironment` et `RunWorkflow`. Migrer d’abord les runners `fluid`, VoF, parallèle et multi-région qui répètent `WM_*`, `LD_LIBRARY_PATH`, MPI et les helpers `run_of13`/`run_parallel`. Conserver `run_command` comme API avancée documentée.

### Priorité P2 — généraliser la gestion des références

Créer `ReferenceCase` autour des imports de fichiers, champs, assets, `.gz`, `.orig`, régions et fusions. Migrer les runners qui définissent localement `import_reference_case`, en commençant par les familles `damBreak`, `snappyHexMesh` et les cas multi-région.

### Priorité P3 — créer les workflows de maillage

Créer `MeshWorkflow` avec les utilitaires récurrents: `blockMesh`, `surfaceFeatures`, `snappyHexMesh`, `extrudeMesh`, `refineMesh`, `createBaffles`, `collapseEdges`, `transformPoints`, `checkMesh` et `reconstruct`. Migrer cinq cas représentatifs: `blockMesh/sphere`, `snappyHexMesh/pipe`, `spiralPipe`, `refineMesh/sector` et un cas multi-région.

### Priorité P4 — exposer la validation utilisateur

Créer `CaseValidation` avec des vérifications standardisées et un rapport lisible. Chaque runner migré devra déclarer ses critères de réussite au lieu de seulement retourner un code nul.

### Priorité P5 — migrer et simplifier les runners

Migrer progressivement les runners par familles, supprimer les helpers locaux devenus inutiles et réduire les plus gros fichiers. Chaque migration doit conserver une validation OF13 de référence et produire un commit séparé.

## Stratégie de migration sans rupture

Les primitives actuelles doivent rester disponibles. Les nouvelles façades doivent les appeler au début, sans dupliquer une seconde implémentation de l’exécution OpenFOAM. Chaque étape doit migrer au moins deux runners de familles différentes et vérifier que les journaux et les fichiers produits restent identiques.

La matrice d’intégration doit ajouter une distinction entre `API réutilisée`, `API étendue` et `nouvelle façade utilisateur`. Le registre API doit documenter les changements de comportement, mais une fonction ne doit être ajoutée que lorsqu’elle exprime une capacité réutilisable au-delà d’un tutoriel.

## Critères d’acceptation

| Critère | Cible |
|---|---|
| Runners avec environnement OF13 recopié | 0 après migration de la tranche prioritaire |
| Helpers locaux `run_of13` | 0 dans les runners migrés |
| Opérations directes interdites | 0 dans tous les runners validés |
| Cas de test de chaque façade | Au moins 3 familles de tutoriels par façade |
| Compatibilité des primitives existantes | 100 % des runners non migrés inchangés |
| Validation de non-régression | Même maillage, mêmes fichiers clés et même statut de calcul OF13 |
| Documentation utilisateur | Exemple minimal et exemple avancé pour chaque façade |

## Conclusion

L’audit ne montre pas que les fonctions ont été conçues une par une pour chaque tutoriel. Il montre plutôt que des primitives générales ont été ajoutées progressivement, puis utilisées depuis des runners qui restent trop proches des détails OpenFOAM. La bonne prochaine étape est donc une consolidation architecturale: centraliser l’environnement, le cas de référence, le maillage, l’exécution et la validation, puis migrer les runners par familles.
