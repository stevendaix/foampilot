# Exemple complet MakeHuman–OpenFOAM 13–JOS-3

Cet exemple montre une simulation de thermorégulation humaine sur une géométrie MakeHuman réaliste, avec échange bidirectionnel face par face entre OpenFOAM 13 et le modèle physiologique JOS-3 intégré à FoamPilot. OpenFOAM fournit à chaque pas la température de surface, le coefficient d’échange convectif et la surface de chaque face humaine. FoamPilot regroupe ces faces dans les 17 zones JOS-3, avance le modèle physiologique, puis renvoie une température de surface par face via le protocole natif `externalCoupledTemperature`.

> **État validé.** Le cas body-only avec plafond réellement ouvert converge sans couplage et avec couplage JOS-3 sur plusieurs dizaines de secondes physiques. Le pilote utilise le même pas temporel que le CFD (`dtime = deltaT`) et non une seconde imposée.

## Pré-requis

Le cas est prévu pour Ubuntu avec OpenFOAM 13, Python 3.12 ou compatible, MakeHuman Community et les dépendances Python `numpy`, `meshio`, `trimesh`, `pyvista` et `jos3`. Le dépôt JOS-3 doit être disponible dans `/home/ubuntu/JOS-3` ou être adapté dans le pilote.

```bash
source /opt/openfoam13/etc/bashrc
python3 -m pip install --user numpy meshio trimesh pyvista
```

L’installation de MakeHuman peut être réalisée avec le script fourni :

```bash
cd examples/thermoregulation/makehuman
bash install_makehuman_ubuntu.sh
```

MakeHuman doit ensuite être lancé avec l’exporteur socket fourni. La procédure complète d’export est décrite dans `../README.md` et dans `export_makehuman_socket.py`.

## Données MakeHuman et sélection body-only

Le fichier interne MakeHuman `base.npz` contient la géométrie et les groupes de faces. Tous les groupes ne représentent pas la peau physique : les groupes `joint-*` et `helper-*` contiennent des joints, helpers, yeux, dents, cheveux, vêtements ou autres éléments auxiliaires. Leur inclusion crée des composantes déconnectées et des bords ouverts.

Le pipeline retenu sélectionne uniquement le groupe MakeHuman `body` (`group=0`) :

```bash
python3 convert_makehuman_meshio.py \
    --input "$HOME/makehuman/v1py3/data/3dobjs/base.npz" \
    --output output/makehuman_body_meshio \
    --group body
```

Lorsque l’export socket est utilisé :

```bash
python3 export_makehuman_socket.py --out output
python3 convert_makehuman_meshio.py \
    --input output/base.npz \
    --output output/makehuman_body_meshio \
    --group body
```

Le groupe body-only doit être fermé avant son utilisation CFD. Les scripts `audit_makehuman_source.py` et `audit_makehuman_topology.py` permettent de vérifier les faces dégénérées, les composantes, les bords ouverts et les arêtes non-manifold.

## Génération du cas OpenFOAM

Le générateur applique l’échelle `0,1` au maillage MakeHuman, de sorte que la hauteur humaine soit d’environ `1,7 m`. Le domaine d’air mesure `x ∈ [-0,75, 0,75] m`, `y ∈ [-1,10, 1,10] m` et `z ∈ [-0,40, 0,60] m`.

```bash
cd examples/thermoregulation/makehuman/openfoam_cube_case
python3 ../create_openfoam_cube_case.py
```

Le domaine versionné possède une ouverture physique au plafond. Les faces `inlet` et `outlet` sont des murs historiques conservés pour la compatibilité du maillage, tandis que `ceiling` est le seul patch ouvert. `prepare_fields.py` reproduit automatiquement ces conditions à chaque reconstruction.

```bash
source /opt/openfoam13/etc/bashrc
./Allrun
```

`Allrun` exécute `prepare_fields.py`, `blockMesh`, `snappyHexMesh`, `createExternalCoupledPatchGeometry`, la génération du mapping face-à-zone et `checkMesh`. Les sorties de maillage ne sont pas versionnées ; elles sont recréées localement.

## Mapping vers les 17 zones JOS-3

Le fichier `zone_mapping_openfoam.csv` associe chaque face réelle du patch OpenFOAM `human` à une zone JOS-3 et à une aire en m². Le mapping est construit à partir des centres de faces OpenFOAM, et non à partir des identifiants de triangles STL. Il doit contenir exactement autant de lignes que le nombre de faces du patch humain et la somme des aires doit être égale à la surface CFD totale.

```bash
python3 map_openfoam_human_faces.py
python3 ../../validation/compare_openfoam_human_surfaces.py
```

Les 17 zones sont celles de JOS-3 : tête, cou, poitrine, dos, bassin, épaules, bras, mains, cuisses, jambes et pieds selon la nomenclature canonique intégrée dans `foampilot.physiology.jos3`.

## Couplage OpenFOAM–JOS-3

Après la préparation du maillage et du protocole :

```bash
mkdir -p comms
createExternalCoupledPatchGeometry T
python3 ../../openfoam_jos3_coupling/openfoam13_jos3_driver.py "$PWD" &
foamRun -solver fluid
```

Le protocole écrit dans `comms/data.out` une ligne par face :

```text
area[m2]  T[K]  qDot[W/m2]  htc[W/m2/K]
```

Le pilote lit ces données, convertit les températures en degrés Celsius pour JOS-3, effectue l’échange distribué sur les 9 418 faces, puis écrit dans `comms/data.in` :

```text
T_surface[K]  snGrad  valueFraction
```

Le pilote lit `deltaT` dans `system/controlDict` et le transmet comme `dtime` à JOS-3. Cette synchronisation est obligatoire dans un calcul transitoire.

## Référence JOS-3 seule

Pour comparer la dynamique physiologique sans CFD :

```bash
python3 ../../openfoam_jos3_coupling/run_jos3_only_comparison.py \
    "$PWD" 584 0.05
```

La référence réutilise les surfaces et le mapping CFD. Par défaut, elle utilise le dernier champ HTC disponible comme environnement imposé. Pour une validation stricte, il faut enregistrer les champs `h`, `Ta`, `Tr`, températures et puissances à chaque échange, puis les rejouer dans JOS-3 seul.

## Résultats de validation

Le cas réellement ouvert au plafond a été testé avec Boussinesq et gravité active. Une simulation OpenFOAM seule à température de peau fixe atteint `5 s` sans erreur. Le calcul couplé atteint environ `29,5 s` physiques dans le benchmark long, avec 584 échanges complets avant l’arrêt contrôlé du benchmark pour limiter le temps d’exécution.

| Indicateur | Résultat |
|---|---:|
| Faces humaines échangées | 9 418 |
| Zones physiologiques | 17 |
| Pas CFD/JOS-3 | 0,05 s |
| Échanges couplés réalisés | 584 |
| Durée couplée observée | ≈ 29,2 s |
| Température retournée finale | 33,55–34,07 °C |
| HTC observé | 1,351–13,44 W·m⁻²·K⁻¹ |
| Modèle thermophysique | Boussinesq |
| Gravité | (0 −9,81 0) m·s⁻² |

La référence JOS-3 seule sur 29,2 s donne une température cutanée moyenne de 34,382 à 34,312 °C, avec une plage finale de 33,765 à 35,412 °C. Les différences de plage sont attendues : le calcul couplé retourne une température distribuée par face sous-relaxée, tandis que JOS-3 seul expose ses 17 températures cutanées physiologiques.

Le rapport détaillé est disponible dans `../../validation/long_coupled_vs_jos3_only.md`, et le rapport de stabilité CFD dans `../../validation/openfoam_stability_tests.md`.

## Visualisation FoamPilot/PyVista

Les résultats OpenFOAM peuvent être lus directement avec le post-traitement FoamPilot/PyVista. Les sorties `0.05`, `0.10`, etc. sont créées par OpenFOAM lorsque `writeInterval` est activé. Pour une visualisation locale :

```python
from foampilot.postprocess.openfoam_direct import OpenFOAMCaseReader

reader = OpenFOAMCaseReader("examples/thermoregulation/makehuman/openfoam_cube_case")
mesh = reader.read_time("0.2")
plotter = mesh.plot(scalars="T", cmap="coolwarm")
plotter.show()
```

Les cartographies recommandées sont la température `T`, la vitesse `U`, la pression relative `p_rgh`, le HTC face par face et le flux thermique retourné par JOS-3. Les fichiers générés par la visualisation restent hors du dépôt.

## Nettoyage et reproductibilité

```bash
./Allclean
./Allrun
```

Le dépôt versionne les scripts, dictionnaires, STL body-only, mapping et rapports. Les fichiers `constant/polyMesh`, `comms`, les champs temporels, les fichiers `human_*.obj` et les journaux sont ignorés ou supprimés de la version de référence, car ils dépendent de l’exécution locale et peuvent atteindre plusieurs dizaines de mégaoctets.

## Limites scientifiques

Ce cas valide une chaîne logicielle et un couplage thermique face par face ; il ne constitue pas encore une validation expérimentale complète du confort thermique humain. Une étude définitive devra documenter la condition d’ambiance, le rayonnement, l’humidité, les propriétés dépendantes de la température, la ventilation et le bilan d’énergie par zone. Le benchmark JOS-3 seul doit également être alimenté par les mêmes séries temporelles CFD pour séparer proprement les effets physiologiques et aérodynamiques.

## Références

[1]: https://openfoam.org/version/13/ OpenFOAM 13
[2]: https://github.com/TanabeLab/JOS-3 JOS-3
[3]: https://github.com/stevendaix/foampilot FoamPilot
