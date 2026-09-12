# Flux de travail avancés

Cette page rassemble des flux de travail présents dans le package mais qui n'avaient pas été décrits précédemment dans la documentation en anglais. Ils sont plus spécialisés que les exemples basiques de cavité et d'aérodynamique externe et doivent être considérés comme expérimentaux à moins que le tutoriel correspondant n'ait été validé avec la version OpenFOAM ciblée.

## Transfert de chaleur conjugué

Le package `foampilot.cht` construit des cas multi-régions pour `chtMultiRegionFoam` et les solveurs associés. Les principaux objets sont `ChtSolver`, `FluidRegion`, `SolidRegion` et `CoupledInterface`. Les générateurs de conditions aux limites couvrent la température fixée, le flux thermique, la température d'entrée/sortie, la symétrie, la température totale, les conditions couplées à la radiation et les conditions d'interface couplée.

Un cas CHT est organisé par région :

```text
case/
├── 0/
│   ├── fluid/
│   └── solid/
├── constant/
│   ├── fluid/
│   ├── solid/
│   └── regionInterfaces/
└── system/
```

Le `controlDict` généré contient la correspondance des solveurs par région. Une exécution en série lance l'exécutable CHT autonome ; une exécution parallèle décompose toutes les régions, lance MPI et reconstruit toutes les régions.

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion

fluid = FluidRegion(name="fluid", temperature=300.0)
solid = SolidRegion(name="solid", temperature=350.0)
solver = ChtSolver(
    case_path="case",
    solver_name="chtMultiRegionFoam",
    regions=[fluid, solid],
)
solver.setup_case()
solver.run_simulation(nb_proc=1)
```

Les arguments exacts du constructeur et des matériaux dépendent de la version d'OpenFOAM utilisée par le cas. Utilisez le tutoriel `09_CHT_heatedDuct` comme référence exécutable et inspectez les dictionnaires de région générés avant d'exécuter.

Les assistants de post-traitement CHT peuvent calculer le flux de chaleur, le flux de chaleur à l'interface, le nombre de Nusselt, l'épaisseur de la couche limite thermique, le coefficient d'échange thermique, le bilan thermique total, les contours de température et la résistance thermique.

## Windkessel et utilitaires physiologiques

`WindkesselModel` est disponible depuis le package de niveau supérieur pour la modélisation aux frontières cardiovasculaires d'ordre réduit. Il doit être couplé à une convention pression/débit clairement définie et validé par rapport à la condition aux limites OpenFOAM prévue avant une utilisation en production.

Le package d'utilitaires contient aussi des aides pour la géométrie vasculaire et médicale, y compris la conversion NIfTI-vers-STL, le nettoyage de la surface de l'aorte, l'optimisation de maillage et un intégrateur CSV pour les fichiers foam. Ces outils peuvent nécessiter des packages optionnels tels que NiBabel, Trimesh, VMTK, PyFQMR ou PyACVD.

## Données météorologiques et atmosphériques

`WeatherFileEPW` lit les fichiers EnergyPlus Weather (EPW). Il peut être utilisé pour extraire la température extérieure, le vent, le rayonnement et d'autres séries temporelles avant de les convertir en conditions aux limites FoamPilot ou en forçage atmosphérique. Traitez le fichier EPW comme un jeu de données d'entrée et enregistrez sa source, son emplacement et son fuseau horaire dans les métadonnées du cas.

Les modules `foampilot.utilities.wind_profile` et `foampilot.postprocess.wind_analysis` fournissent des aides pour les profils de vent et les ensembles de vents. Ceux-ci sont utiles pour comparer plusieurs directions de vent ou hypothèses de couche limite atmosphérique, mais ils ne remplacent pas une calibration physique des conditions aux limites atmosphériques.

## CFD urbain

Le package `foampilot.urban` est une chaîne expérimentale pour la CFD à l'échelle urbaine. Il expose des modèles de données pour les bâtiments, le terrain, les routes et les domaines CFD ; la simplification et le nettoyage de la géométrie ; des générateurs de domaine quart-de-surface basés sur Gmsh ou la surface ; des objets de dimensionnement de maillage et de raffinement d'éveil ; l'assignation des patches ; des profils de couche limite atmosphérique ; et la validation de géométrie/maillage.

Un flux de travail de haut niveau :

```python
from foampilot.urban import (
    UrbanModel,
    CFDSimplifier,
    MeshConfig,
    ABLProfile,
    GeometryValidator,
)

# Charger ou construire un UrbanModel à partir du lecteur supporté.
# Simplifier la géométrie pour la CFD, construire un domaine, dimensionner le maillage,
# assigner les patches, valider, puis exporter vers le workflow OpenFOAM.
```

Les lecteurs OSM et LiDAR sont optionnels car ils dépendent de bibliothèques géospatiales et de jeux de données externes. Installez l'option supplémentaire avant de les importer :

```bash
pip install -e ".[urban]"
```

Les cas urbains doivent documenter le système de référence de coordonnées, la conversion métrique, le repère du vent, la source du terrain, les hypothèses de hauteur des bâtiments, la tolérance de simplification, le budget de maillage et le profil atmosphérique. Ces détails sont essentiels pour la reproductibilité et ne sont pas déductibles en toute sécurité à partir du maillage généré seul.

## MakeHuman et thermorégulation

Le dépôt contient un flux de travail MakeHuman-vers-STL pour des expériences de thermorégulation. Le flux de travail exporte un modèle de corps, sélectionne la surface cutanée principale, crée des zones de surface JOS-3 et écrit une cartographie de zones pour un couplage ultérieur. Il s'agit d'un flux de travail externe plutôt que d'une fonctionnalité générique du solveur FoamPilot, donc la documentation en anglais doit renvoyer à son README et indiquer explicitement ses exigences externes.

Lors de l'utilisation de ce flux de travail, enregistrez la version de MakeHuman, la pose du modèle, le groupe de surface exporté, la cartographie de zones JOS-3 et le cas OpenFOAM utilisé pour le couplage. N'interprétez pas un STL généré comme un modèle physiologique validé sans vérifier la topologie de la surface et l'affectation des zones.
