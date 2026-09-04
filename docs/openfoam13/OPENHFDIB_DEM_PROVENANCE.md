# Provenance de openHFDIB-DEM

## Dépôt amont contrôlé

Le portage HFDIB-DEM intégré dans cette PR est comparé au dépôt amont suivant :

- Dépôt : [MartinIsoz/openHFDIB-DEM](https://github.com/MartinIsoz/openHFDIB-DEM)
- Commit contrôlé : `3ade083233d56a6b62e58399fac5e8b4043d51a4`
- Date du commit contrôlé : 14 avril 2024
- Dépôt de code maintenu indiqué par le README amont : [techMathGroup/openHFDIB-DEM](https://github.com/techMathGroup/openHFDIB-DEM)

Le README amont indique que cette base est préparée pour **OpenFOAM v8**. Elle ne constitue donc pas, à elle seule, une preuve de compatibilité avec OpenFOAM Foundation 13.

## Nature de l’intégration

Le contenu de `third_party/openHFDIB-DEM/` n’est pas une copie binaire ou une simple importation sans modification. La PR ajoute une organisation vendorisée adaptée à FoamPilot, des fichiers de compatibilité Foundation 13 sous `src/HFDIBDEM/compat/openfoam13`, des modèles et interfaces supplémentaires, ainsi qu’un portage du solver `pimpleHFDIBFoam`.

La comparaison contrôlée distingue les fichiers communs modifiés des fichiers spécifiques à FoamPilot. Les bibliothèques compilées, fichiers `.o`, `.dep`, `.so`, `lnInclude/` et répertoires `Make/linux*/` ne sont pas des sources et sont exclus du versionnage par `.gitignore`.

## Cas de validation

Le dépôt amont contrôlé fournit principalement les tutoriels `fallingParticle` et `impellerAndSphere`. Il ne fournit pas de cas nommé `normalForce_OF13`. Ce dernier ne doit donc pas être référencé comme une fixture amont implicite.

Le validateur approfondi de FoamPilot accepte désormais la variable `MULTIPHYSICS_VALIDATION_CASE`. Si aucun cas DEM local n’est disponible, il échoue explicitement avec le chemin attendu et ne transforme pas l’absence du cas en validation réussie.

La couverture déclarée doit ainsi distinguer :

| Périmètre | Signification |
|---|---|
| Sources HFDIB-DEM | Base amont comparée et adaptations Foundation 13 documentées |
| Compilation OpenFOAM 13 | À vérifier avec les bibliothèques réellement produites dans l’environnement |
| Cas `fallingParticle` / `impellerAndSphere` | Cas amont disponibles à porter et qualifier |
| Cas `normalForce_OF13` | Cas FoamPilot distinct, à ajouter explicitement ou à retirer du validateur |

## Licence et redistribution

Aucun fichier `LICENSE` ou `COPYING` autonome n’a été trouvé à la racine du checkout amont contrôlé. Avant une redistribution durable du code vendorisé, la licence et les conditions de réutilisation doivent être confirmées auprès du dépôt maintenu ou des auteurs. Cette note ne remplace pas une vérification juridique.

## Références

1. [Dépôt amont MartinIsoz/openHFDIB-DEM](https://github.com/MartinIsoz/openHFDIB-DEM)
2. [Dépôt maintenu techMathGroup/openHFDIB-DEM](https://github.com/techMathGroup/openHFDIB-DEM)
3. [Documentation OpenFOAM Foundation v8 citée par le README amont](https://openfoam.org/version/8/)
