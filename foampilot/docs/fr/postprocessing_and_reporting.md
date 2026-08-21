# Post-traitement et génération de rapports

FoamPilot prend en charge deux voies de post-traitement complémentaires. La voie traditionnelle convertit un cas avec `foamToVTK` puis charge la sortie VTK avec PyVista. La voie directe lit les maillages et champs natifs OpenFOAM dans PyVista sans créer d’arborescence VTK intermédiaire.

## Lecteurs OpenFOAM natifs

Utilisez le lecteur direct lorsque le cas contient déjà un `constant/polyMesh` valide et que vous souhaitez éviter une étape de conversion externe.

```python
from foampilot.postprocess import OpenFOAMDirectReader

reader = OpenFOAMDirectReader("/path/to/case")
mesh = reader.to_pyvista(fields=["U", "p"], time_step="latest")
print(mesh.n_points, mesh.n_cells)
```

Le lecteur détecte les champs aux points et aux cellules à partir des en-têtes de champs OpenFOAM, prend en charge le chargement paresseux, met en cache les champs et peut lire des fichiers de champs compressés. La fonction utilitaire est pratique pour de petits scripts :

```python
from foampilot.postprocess import read_openfoam

mesh = read_openfoam(
    "/path/to/case",
    fields=["U", "p"],
    time_step="latest",
)
```

Pour les cas de transfert thermique conjugué (CHT), utilisez `CHTDirectReader`. Il découvre les régions fluide et solide et renvoie une structure PyVista `MultiBlock`.

```python
import pyvista as pv
from foampilot.postprocess import CHTDirectReader

reader = CHTDirectReader("/path/to/cht-case")
print(reader.region_names)
blocks = reader.get_all_meshes(fields=["T"], time_step="latest")

plotter = pv.Plotter(off_screen=True)
for region_name, region_mesh in blocks.items():
    plotter.add_mesh(region_mesh, scalars="T", name=region_name)
plotter.screenshot("temperature.png")
plotter.close()
```

Les températures d’interface peuvent être inspectées directement lorsqu’une interface de région nommée est disponible :

```python
interface = reader.get_interface_temperatures(
    "fluid_to_solid", time_step="latest"
)
print(interface["fluid_T"])
print(interface["solid_T"])
print(interface["T_interface"])
```

## Post-traitement PyVista

`FoamPostProcessing` reste utile lorsque le flux de travail existant repose sur `foamToVTK`, la découverte des instants de calcul, ou des aides de tracé de plus haut niveau.

```python
from foampilot.postprocess import FoamPostProcessing

post = FoamPostProcessing(case_path="/path/to/case")
post.foamToVTK()
time = post.get_all_time_steps()[-1]
mesh = post.load_time_step(time)["cell"]
```

Les opérations typiques incluent des coupes, isovaleurs, champs vectoriels, analyse des vortex, statistiques de maillage, export d’images et export d’animations. En environnements sans affichage (headless), créez des graphiques avec `off_screen=True` ou utilisez les utilitaires de rendu de FoamPilot pour détecter un backend hors écran exploitable.

## Présentations web interactives

Le module `foampilot.postprocess.web_presentation` propose des générateurs Plotly pour les champs de vitesse, pression et température, ainsi qu’un `CFDDashboard` pour l’exploration interactive. Un schéma minimal est :

```python
from foampilot.postprocess.web_presentation import (
    plotly_velocity_magnitude,
    plotly_pressure_contour,
    CFDDashboard,
)

velocity_figure = plotly_velocity_magnitude(mesh)
pressure_figure = plotly_pressure_contour(mesh)
# Pass the figures to the dashboard or to a Plotly/Streamlit application.
```

Le tableau de bord est destiné à l’exploration et à la communication. Pour une traçabilité d’ingénierie reproductible, enregistrez ensemble le script d’entrée, les dictionnaires générés, le journal du solveur, les figures et le rapport.

## Rapports de simulation et de maillage

Le paquet `foampilot.report` inclut des rapports structurés sur la qualité du maillage, la convergence et les études de solveurs. L’API de reporting est conçue pour s’exécuter après la simulation afin que les exécutions échouées ou incomplètes puissent être consignées plutôt qu’ignorées silencieusement.

L’API LaTeX convient lorsqu’une note de calcul PDF est requise :

```python
from foampilot.report import latex_pdf

document = latex_pdf.LatexDocument(
    title="OpenFOAM simulation report",
    author="FoamPilot",
    filename="simulation_report",
    output_dir="postProcessing/report",
)
document.add_section("Purpose", "Summary of the simulated case.")
document.add_figure("postProcessing/velocity.png", caption="Velocity magnitude")
document.generate_document(output_format="pdf")
```

Pour la génération de documents sans chaîne d’outils LaTeX, le moteur de rendu Typst expose des blocs de construction structurés tels que sections, équations, figures, tableaux, blocs de code et bibliographies. Préférez Typst lorsque le projet utilise déjà des modèles `.typ` ou lorsque la mise en page déterministe est importante.

## Études parallèles

`ParallelStudy` automatise une comparaison des décompositions en processeurs. Il peut écrire `decomposeParDict`, exécuter le cas de référence et les cas parallèles, analyser les journaux, collecter les métriques de temps et de maillage, et exporter des visualisations des frontières processeur. OpenFOAM et un environnement d’exécution MPI doivent être disponibles dans le `PATH`.

Avant de lancer une étude, faites une copie du cas ou utilisez un répertoire de sortie jetable. Les exécutions parallèles modifient le cas en créant des répertoires processeur et des sorties de reconstruction.

## Organisation recommandée des résultats

Un projet reproductible peut utiliser l’organisation suivante :

```text
case_project/
├── run.py
├── case/
│   ├── 0/
│   ├── constant/
│   └── system/
├── logs/
├── postProcessing/
│   ├── figures/
│   ├── tables/
│   └── reports/
└── README.md
```

Conservez les sorties générées séparées de la géométrie source et des entrées CSV. Cela permet de supprimer un répertoire de cas et de le reconstruire à partir du script sans perdre la provenance scientifique de l’exécution.

## Types de champs et grandeurs dérivées

Le post-traitement doit distinguer les données aux points, aux cellules et de surface. Un vecteur vitesse stocké aux centres des mailles n’est pas interchangeable avec une valeur interpolée aux sommets. La pression de surface et la contrainte de cisaillement pariétale doivent être intégrées sur le véritable patch de paroi, tandis que les moyennes volumiques requièrent les volumes de mailles.

Les grandeurs dérivées courantes incluent :

| Grandeur | Définition ou usage typique |
| --- | --- |
| Magnitude de la vitesse | $|\mathbf{U}|$ pour les cartes de vitesse et les zones de seuil. |
| Vorticité | $\nabla\times\mathbf{U}$ pour les structures rotationnelles. |
| Critère Q | Identifie les régions où la rotation domine la déformation. |
| Contrainte de cisaillement pariétale | Traction tangentielle à une paroi ; sensible au maillage proche paroi. |
| Coefficient de pression | $C_p=(p-p_\infty)/(\tfrac12\rho U_\infty^2)$ pour les écoulements externes. |
| Flux thermique | Flux thermique normal conductif ou total à une surface. |
| Nombre de Nusselt | Transfert thermique adimensionnel basé sur une longueur caractéristique donnée. |
| Fraction de phase | Localisation d’interface et diagnostics de volume liquide en VOF. |
| Indice de mélange scalaire | Uniformité ou variance d’une concentration transportée. |

La définition, l’état de référence, la convention de signe et l’opération de moyennage doivent accompagner chaque grandeur exportée.

## Résidus et convergence

Le résidu est une mesure algébrique du respect d’une équation discrétisée au cours d’une itération. Ce n’est pas automatiquement une estimation de l’erreur sur la grandeur physique d’intérêt. Un cas peut présenter de petits résidus tout en donnant un coefficient de traînée, un bilan thermique ou une répartition de débit en sortie erronés.

Un rapport de post-traitement robuste doit donc contenir :

1. les historiques de résidus du solveur pour chaque région et champ ;
2. les forces, flux, températures ou moyennes scalaires suivis ;
3. les erreurs de continuité et la conservation du volume ;
4. les statistiques finales du maillage ;
5. le temps final, le pas de temps, le nombre de Courant et les nombres d’itérations ;
6. le critère de convergence utilisé pour le résultat d’ingénierie.

`ResidualsPost` peut transformer les journaux du solveur en artefacts CSV, JSON, PNG ou HTML. Conservez le fichier journal original, car les résumés parsés peuvent masquer des avertissements, des exceptions en virgule flottante ou des redémarrages du solveur.

## Analyse des frontières et des patches

L’analyse au niveau des patches est essentielle pour l’aérodynamique externe, les écoulements biomédicaux et le CHT. Un rapport de patch fiable identifie le nom du patch, le type de patch, la surface, le nombre de faces, les valeurs min/max/moyennes des champs et le flux ou la force intégrés le cas échéant.

Pour un véhicule, rapportez les forces par patch et par direction. Pour un modèle vasculaire, rapportez le débit et la pression à chaque entrée/sortie et vérifiez la conservation. Pour le CHT, rapportez le flux thermique indépendamment côté fluide et côté solide de l’interface, en explicitant la convention de normale.

## Ensembles de vent et cas multiples

Le module d’analyse du vent fournit des objets tels que `WindRose`, `WindCaseResult`, `WindEnsemble`, `LawsonProcessor` et `LawsonVisualizer`. Ils peuvent organiser plusieurs directions de vent ou cas atmosphériques et combiner leurs résultats en synthèses directionnelles. Ils ne remplacent pas la définition physique du profil d’entrée ni le choix d’un critère de confort.

Un ensemble de vent doit consigner pour chaque cas :

| Métadonnée | Exemple |
| --- | --- |
| Direction | Convention météorologique ou cartésienne, énoncée explicitement. |
| Vitesse de référence | Hauteur et période de moyennage. |
| Profil atmosphérique | Loi logarithmique, loi de puissance, profil mesuré ou champ précurseur. |
| Stabilité | Neutre, stable, instable ou inconnue. |
| Solveur/modèle | Fermeture RANS, fonctions de paroi, pas de temps et discrétisation. |
| Poids | Fréquence ou probabilité attribuée au cas. |

## Post-traitement CHT

Pour un cas CHT, chargez toutes les régions au même temps physique. Comparer un champ fluide à un instant avec un champ solide à un autre peut créer un faux désaccord d’interface. Le `CHTDirectReader` direct peut charger les champs de température sous forme d’objet `MultiBlock` ; les utilitaires CHT peuvent calculer les températures d’interface, les flux thermiques, la résistance thermique, les coefficients de transfert thermique et les nombres de Nusselt.

Un rapport CHT minimal doit inclure :

- les noms des régions fluide et solide ;
- les propriétés des matériaux et leur dépendance en température ;
- les paires de patches d’interface ;
- la continuité de température à l’interface ;
- la continuité du flux thermique à l’interface ;
- la chaleur totale entrante, sortante et stockée ;
- les nombres de Nusselt locaux et intégrés ;
- la résolution du maillage normale à l’interface ;
- l’historique de convergence.

## Export de données et provenance

Lors de l’export d’un champ en CSV, JSON, VTK ou images, écrivez un fichier de métadonnées contenant :

```text
case identifier
OpenFOAM version
FoamPilot commit
mesh cell count
physical time
field names and locations
units
coordinate system
filter/interpolation operation
reference values
```

C’est particulièrement important pour les cas biomédicaux et urbains, où une visualisation peut être détachée de la géométrie originale, du système de référence de coordonnées ou des entrées patient/environnement.

## Types de rapports

FoamPilot prend en charge plusieurs niveaux de rapport :

| Rapport | Meilleure utilisation |
| --- | --- |
| Résidus CSV/PNG/HTML | Contrôle rapide de la santé numérique pendant le développement. |
| Rapport de qualité de maillage | Revue de géométrie et de discrétisation avant la résolution. |
| Rapport de simulation | Résumé de cas reproductible avec figures et tableaux. |
| Rapport d’étude parallèle | Comparaison du nombre de processeurs et diagnostics de décomposition. |
| PDF LaTeX | Note de calcul formelle ou rapport de type publication. |
| Document Typst | Document scientifique structuré sans flux de travail LaTeX. |
| Tableau de bord Streamlit/Plotly | Exploration interactive pour ingénieurs et collaborateurs. |

N’utilisez pas un tableau de bord comme seul archivage. L’état interactif peut être perdu ; le script du cas, les dictionnaires, le journal du solveur, les données brutes et le résumé statique constituent l’enregistrement reproductible.
