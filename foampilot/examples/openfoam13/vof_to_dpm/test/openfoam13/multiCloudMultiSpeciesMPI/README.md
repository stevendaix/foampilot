# multiCloudMultiSpeciesMPI

Ce cas est un test d’intégration du chemin **Direct Commit** avec deux clouds (`waterCloud` et `fuelCloud`) et deux espèces (`H2O` et `C2H5OH`). Il est basé sur le tutoriel OpenFOAM 13 `compressibleVoF/damBreak`, puis prépare deux champs alpha et deux instances indépendantes de `compressibleVoFClouds` afin de tester le routage par nom de cloud.

La clé transactionnelle vérifiée par l’audit est `(cloudName, fragmentId)`. Le champ alpha est également journalisé dans chaque confirmation, ce qui permet de diagnostiquer les collisions entre `waterCloud.alpha.water` et `fuelCloud.alpha.air`. Le manager utilise une numérotation globale déterministe et la réconciliation MPI ne doit produire qu’un commit et une confirmation par transaction.

## Prérequis

Le cas nécessite OpenFOAM 13, la bibliothèque framework `liblagrangianParcel.so` reconstruite après application du patch Direct Commit, ainsi que `libcompressibleVoFClouds.so`. Le patch est documenté dans `../../../patches/openfoam13/README.md`. Les bibliothèques applicatives se compilent avec `wmake libso` dans leurs répertoires respectifs.

## Lancement

Depuis ce dossier :

```bash
# Validation MPI nominale à deux rangs
NP=2 KEEP_CASE=1 ./Allrun.parallel

# Vérification avec une autre décomposition
NP=4 KEEP_CASE=1 ./Allrun.parallel

Les références numériques du cas nominal sont qualifiées pour `NP=2`; une exécution `NP=4` doit d’abord être utilisée comme contrôle de robustesse MPI, puis ses références doivent être régénérées si la numérotation globale attendue change.
```

`Allrun.parallel` crée un cas temporaire sous `/tmp`, exécute `blockMesh`, `setFields`, `decomposePar`, puis lance `foamRun` avec `mpirun`. Avec `KEEP_CASE=1`, le chemin du cas temporaire est imprimé dans la sortie et le journal reste disponible pour analyse.

## Critères d’acceptation

L’auditeur `analyze_multi_cloud_species.py` doit retourner un objet JSON avec `pass=true`. Le résultat exige que le solveur atteigne `End`, qu’il n’existe aucune erreur MPI/FPE, qu’aucun fallback vers le cloud par défaut ne soit utilisé, et que chaque clé attendue possède exactement un commit et une confirmation.

Le même audit compare également les vecteurs `speciesMassAdded` avec les masses attendues pour `H2O` et `C2H5OH`. La tolérance de `1e-6` concerne uniquement la précision d’affichage des listes dans `Info`; les comparaisons internes de la réconciliation utilisent les valeurs OpenFOAM non tronquées.

## Limites du cas

Ce cas valide le routage, le namespacing, l’insertion locale et la conservation des fractions configurées. Il ne qualifie pas encore l’évaporation, les réactions chimiques, le mélange interfacial, les modèles réactifs ou une composition calculée depuis un champ VOF indépendant par espèce. Ces mécanismes doivent être testés dans des cas physiques dédiés.
