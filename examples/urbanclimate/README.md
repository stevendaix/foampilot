# Exemples de microclimat urbain OpenFOAM 13

Ce répertoire porte dans Foampilot les six tutoriels du projet [`urbanMicroclimateFoam-tutorials`](https://github.com/OpenFOAM-BuildingPhysics/urbanMicroclimateFoam-tutorials), avec le solveur [`urbanMicroclimateFoam`](https://github.com/OpenFOAM-BuildingPhysics/urbanMicroclimateFoam) adapté aux API d’OpenFOAM Foundation 13.

Le solveur est multi-région. Il couple l’écoulement turbulent de l’air, le transport de chaleur et d’humidité, les matériaux poreux des bâtiments, le rayonnement solaire et les modèles de végétation. Les cas HAM utilisent les régions solides et les cas `*_veg` ajoutent les outils de calcul LAI, facteurs de vue et ray-tracing solaire.

## Cas disponibles

| Cas | Régions / phénomènes |
|---|---|
| `streetCanyon_CFD` | Canyon urbain, CFD et thermique de l’air |
| `streetCanyon_CFDHAM` | Canyon, chaleur–humidité des bâtiments |
| `streetCanyon_CFDHAM_grass` | Canyon HAM avec herbe |
| `streetCanyon_CFDHAM_veg` | Canyon HAM avec végétation et rayonnement |
| `windAroundBuildings_CFDHAM` | Vent autour de bâtiments avec HAM |
| `windAroundBuildings_CFDHAM_veg` | Vent autour de bâtiments avec HAM, végétation et ray-tracing |

## Compilation

Charger une installation complète d’OpenFOAM Foundation 13, puis compiler le solveur et ses utilitaires :

```sh
. /path/to/OpenFOAM-13/etc/bashrc
cd examples/urbanclimate
./Allwmake
```

Le script compile les bibliothèques `buildingMaterialModel`, `grassModel`, `blendingLayer`, `vegetationModels`, `solarLoad`, le modèle de turbulence poreux, ainsi que `urbanMicroclimateFoam`, `calcLAI` et `solarRayTracingGen`. Les adaptations Foundation 13 incluent notamment `NamedEnum`, `mappedPatchBase`, `meshSearch`, `dimensionedScalar`, les API de modèle de turbulence et la surface distribuée native.

## Génération et validation avec FoamPilot

Les six répertoires sous `cases/` sont des sorties générées. La configuration des cas est construite depuis les profils Python `UrbanClimateProfile`, `RegionSpec` et `UrbanClimateNativeCaseBuilder`. Le dossier `resources/` ne contient que les géométries et données externes lourdes ; il ne contient aucun dictionnaire `0`, `constant` ou `system`. La génération passe par `run.py` :

```sh
export PYTHONPATH="$PWD/../../foampilot/src${PYTHONPATH:+:$PYTHONPATH}"
python3 run.py --list
python3 run.py --all --generate --overwrite
```

Chaque profil est reconstruit avec ses répertoires `0`, `constant` et `system`. Les fichiers sont écrits par les APIs Foampilot de maillage, de champs, de régions et de dictionnaires OpenFOAM, puis reçoivent les fichiers de provenance et de configuration propres à FoamPilot. La génération est non destructive par défaut ; `--overwrite` est requis pour remplacer un cas existant.

La validation statique de tous les six cas se fait avec :

```sh
./validate_cases.sh
```

Elle régénère et contrôle la présence de `controlDict`, `decomposeParDict`, `Allrun`, l’application `urbanMicroclimateFoam` et la viscosité cinématique `nu` dans les dictionnaires constants. Pour exécuter un cas, `run_case.sh` appelle d’abord `run.py` :

```sh
./run_case.sh streetCanyon_CFD
./run_case.sh streetCanyon_CFDHAM
./run_case.sh streetCanyon_CFDHAM_grass
./run_case.sh streetCanyon_CFDHAM_veg
./run_case.sh windAroundBuildings_CFDHAM
./run_case.sh windAroundBuildings_CFDHAM_veg
```

Les cas végétalisés requièrent une installation fonctionnelle de `blockMesh`, `faceAgglomerate`, `calcLAI`, `viewFactorsGen`, `solarRayTracingGen` et `urbanMicroclimateFoam`. `Allrun` exécute séquentiellement `blockMesh`, le script généré `make_cell_zones.py`, `faceAgglomerate`, `calcLAI`, `viewFactorsGen`, `solarRayTracingGen`, puis le solveur. Le script crée les cartes `cellZones` et `finalAgglom` et transforme les frontières de vegetation en patches `mapped`/`mappedWall` vers air ; cette approche ne dépend donc pas de `topoSet` ou de `changeDictionary`. Les résultats sont générés dans les répertoires de cas et ne sont pas versionnés.

## Portage et limites explicites

Les cas proviennent du tag Foundation v12 du projet original. Aucun fichier n’est présenté comme Foundation 13 sans compilation : le solveur et les outils sont portés puis compilés avec `wmake` sous Foundation 13. Les modèles physiques sont conservés, tandis que les changements portent sur les interfaces retirées ou modifiées dans Foundation 13.

Les résultats numériques doivent être vérifiés avec `checkMesh`, les journaux de chaque région et les champs produits. La compatibilité d’exécution Foundation 13 a été contrôlée pour les profils végétalisés avec maillage, LAI, facteurs de vue, ray-tracing solaire et solveur complets. Une réussite d’exécution ne remplace pas une validation scientifique ; il faut notamment comparer les températures de surface, les vitesses à hauteur de piéton et les flux chaleur–humidité aux cas de référence du projet d’origine.
