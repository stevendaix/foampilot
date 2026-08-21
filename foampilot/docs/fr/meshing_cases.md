# Maillage des cas et stratégie de maillage

Le maillage n'est pas un détail de pré-traitement pouvant être séparé de la physique. Il détermine quels gradients peuvent être résolus, quel traitement des parois est valide, avec quelle précision les forces et flux thermiques sont intégrés, et quelle quantité de diffusion numérique entre dans la solution. FoamPilot orchestre plusieurs stratégies de maillage, mais l'utilisateur reste responsable du choix de la représentation géométrique, des cibles de raffinement, de la topologie des patches et des critères de qualité.

## Sélection de la stratégie de maillage

| Géométrie ou objectif | Itinéraire recommandé | Pourquoi |
| --- | --- | --- |
| Cavité rectangulaire, canal, conduit ou benchmark 2-D | `blockMesh` | Topologie explicite, cellules prévisibles, excellent pour la vérification. |
| Géométrie multi-blocs structurée | `classy_blocks` / `blockMesh` | Contrôle fort du grading, des blocs, des arcs et des patches nommés. |
| Solide CAD ou géométrie STEP | Gmsh | Maillage surface/volume non structuré flexible et import CAD. |
| Surface triangulée STL, OBJ, bâtiment, véhicule ou biologique | Maillage de fond `blockMesh` + `snappyHexMesh` | Raffinement local et snapping autour de surfaces triangulées complexes. |
| Maillage OpenFOAM existant | Lecteur/exporteur de maille direct | Évite le remeshing et permet un flux de post-traitement contrôlé par Python. |
| Grand jeu de données urbaines | Lecteurs urbains + simplification + Gmsh/générateur de surface | Contrôle la complexité géométrique, les coordonnées métriques et le budget de cellules. |
| Cas CHT fluide/solide | `blockMesh` ou Gmsh + zones cellulaires + séparation de régions | Le maillage doit représenter les deux régions et leur interface couplée. |

## 1. Cas structurés `blockMesh`

`blockMesh` est la voie de choix pour les cas de vérification parce que la topologie est explicite. L'utilisateur contrôle les sommets, les blocs, les arêtes, le grading et les patches de frontière. Cela le rend approprié pour la cavité, le canal scalaire, la pièce chauffée par flottabilité, le pas divergent, et le maillage de fond de conduits chauffés.

Un cas structuré doit définir :

1. le système de coordonnées et les dimensions ;
2. la connectivité des blocs et le nombre de cellules ;
3. le ratio de grading dans chaque direction ;
4. les noms de patches et les types de patch OpenFOAM ;
5. l'échelle dimensionnelle ;
6. la résolution de paroi prévue et les hypothèses de symétrie.

Les principaux risques sont un ordre incorrect des sommets, des normales de faces incohérentes, un grading excessif, et des patches qui ne correspondent pas au code des conditions aux limites. Exécutez `blockMesh` et `checkMesh` avant d'écrire le reste du cas.

## 2. `classy_blocks` et géométrie multi-blocs

`classy_blocks` est utile lorsqu'une géométrie est naturellement assemblée à partir de cylindres, d'extrusions, d'anneaux, de coudes ou de blocs chaînés. Le guide utilisateur de FoamPilot montre la construction de formes, le chaînage, l'expansion, le remplissage, la découpe directionnelle, et l'assignation de patches.

L'avantage est le contrôle géométrique. L'inconvénient est que l'utilisateur doit comprendre comment les blocs se rencontrent et comment le grading des cellules se propage aux interfaces de blocs. Utilisez-le pour une géométrie dont la topologie est connue ; ne l'utilisez pas pour dissimuler une surface CAD mal comprise.

## 3. Cas Gmsh

Gmsh est approprié pour la géométrie de type STEP/IGES/CAD et pour les domaines où un maillage tétraédrique non structuré ou hybride est préférable. Un cas Gmsh doit documenter :

| Entrée | Décision requise |
| --- | --- |
| Unités CAD | Confirmer si la source est en mètres, millimètres ou un autre système d'unités. |
| Groupes physiques | Définir explicitement entrée, sortie, parois, symétrie, interfaces et régions solides. |
| Ordre des éléments | Choisir des éléments linéaires ou d'ordre supérieur en cohérence avec la chaîne du solveur. |
| Qualité de surface | Retirer les faces dupliquées, auto-intersectées ou mal orientées. |
| Fermeture de volume | Confirmer que chaque volume fluide ou solide est étanche. |
| Conversion | Vérifier comment le maillage généré est converti vers OpenFOAM et comment les noms de patch survivent. |

Le raffinement Gmsh doit être dicté par la physique : interstices étroits, surfaces à forte courbure, arêtes de séparation, interfaces thermiques et couches limites nécessitent plus de cellules que les régions uniformes.

## 4. Cas `snappyHexMesh`

La séquence standard pour géométrie complexe est :

```text
background blockMesh
→ surfaceFeatureExtract
→ castellatedMesh
→ snap
→ addLayers (optional)
→ checkMesh
```

Le maillage de fond définit le domaine externe. La géométrie de surface est placée sous `constant/triSurface` ou dans le répertoire de géométrie configuré. `snappyHexMesh` supprime ou raffine les cellules en fonction des intersections géométriques, aligne les points sur la surface, et peut ajouter des couches prismatiques.

### Régions de raffinement

Utilisez un raffinement local autour de :

- arêtes d'attaque et de fuite ;
- coins de bâtiments et lignes de toit ;
- roues de véhicule, carénages et interstices sous-voiture ;
- régions d'éveil (wake) derrière des corps bluff ;
- interfaces thermiques et passages fluides étroits ;
- sténoses médicales, collets d'anévrisme, bifurcations, et extensions d'entrée/sortie.

Le niveau de raffinement doit être équilibré avec le modèle de turbulence et le traitement des parois. Une fine maille de surface avec une couche limite sous-résolue n'est pas automatiquement un bon maillage CFD.

### Vérifications de surface et de features

Avant d'exécuter un cas complexe, inspectez la surface dans un visualiseur et vérifiez :

| Vérification | Conséquence typique en cas d'échec |
| --- | --- |
| Surface fermée et orientable | Fuites, cellules manquantes, classification intérieur/extérieur incorrecte. |
| Échelle cohérente | Géométrie trop grande ou trop petite par rapport à la vitesse et à la viscosité. |
| Extraction de features | Les arêtes vives sont arrondies ou les patches sont fusionnés de façon inattendue. |
| Noms de patch | Les conditions aux limites sont appliquées à la mauvaise surface. |
| Normales de surface | Orientation des parois ou signes des flux incorrects. |
| Faisabilité des couches | Les couches prismatiques s'effondrent ou créent des cellules non orthogonales. |

## 5. Maillages urbains et atmosphériques

La CFD urbaine nécessite une étape géospatiale avant le maillage OpenFOAM. Convertissez les données dans un système de coordonnées métriques, définissez le repère du vent, retirez les objets non pertinents, simplifiez les empreintes de bâtiments, assignez des hauteurs, et établissez le terrain et les marges du domaine. Le package urbain contient des modèles pour bâtiments, routes, terrain, domaines CFD, simplification de géométrie, nettoyage, dimensionnement de maille, raffinement wake, couches limites, assignation de patches et validation.

Le domaine de maillage doit être justifié par la couche limite atmosphérique entrante et le wake en aval. Un domaine trop court recycle la pression et les perturbations de turbulence dans la région d'intérêt. Un domaine trop étroit latéralement contraint le vent et exagère le blocage.

## 6. Maillages de surface et de volume biomédicaux

Les maillages biomédicaux exigent un soin supplémentaire parce que la géométrie est spécifique au patient et que les quantités d'intérêt dépendent souvent de dérivées : WSS, perte de charge, temps de résidence, ou transfert de chaleur. Le flux de travail inclut typiquement la segmentation d'images ou l'import de surface, le nettoyage, la fermeture de trous, le lissage avec une tolérance contrôlée, l'extension des entrées/sorties, le remeshing de surface, le maillage volumique, et le raffinement de couche limite lorsqu'approprié.

Une opération de traitement de géométrie ne doit jamais être décrite seulement comme « nettoyage ». Enregistrez l'algorithme, la tolérance, la longueur d'arête cible, le nombre de triangles, les itérations de lissage, et si l'opération modifie le volume du lumen ou les diamètres de branche. Validez le maillage final par rapport à la surface d'imagerie d'origine.

Pour l'écoulement sanguin, raffinez les régions de forte courbure, sténose, bifurcation, recirculation, et celles avec des gradients attendus élevés de contrainte de cisaillement pariétal. Prolongez suffisamment les sorties pour réduire l'influence des conditions aux limites artificielles sur la région d'intérêt.

## 7. Maillages CHT et interfaces de régions

Un maillage CHT doit distinguer les cellules fluides des cellules solides et doit préserver une interface conforme ou autrement correctement couplée. Le tutoriel utilise un maillage de fond et des définitions de zones cellulaires avant de scinder le cas en régions `fluid` et `solid`.

L'interface requiert :

- faces appariées ou correctement mappées ;
- champs de température spécifiques à chaque région ;
- propriétés thermophysiques dans chaque région ;
- conditions aux limites couplées de température et de flux thermique ;
- une direction normale cohérente et une convention de nommage d'interface ;
- une résolution suffisante à travers la couche limite thermique et le chemin de conduction solide.

La plus petite taille de cellule doit être justifiée par les gradients de quantité de mouvement et thermiques. Un maillage peut résoudre la vitesse tout en sous-résolvant la température, ou l'inverse. Utilisez des estimations de la couche limite thermique et le nombre de Pr local pour guider le premier maillage, puis effectuez une étude de raffinement.

## 8. Indicateurs de qualité de maillage

`checkMesh` est nécessaire mais pas suffisant. Signalez au moins les indicateurs suivants :

| Indicateur | Interprétation |
| --- | --- |
| Non-orthogonalité | De grandes valeurs augmentent l'erreur de discrétisation et peuvent nécessiter une correction ou un maillage différent. |
| Skewness | Une forte skewness dégrade la reconstruction des gradients et des flux. |
| Rapport d'aspect | De grands rapports peuvent être valides dans les couches limites mais nuisibles dans les régions mal alignées. |
| Rapport de volume | Des changements brusques de taille de cellule peuvent produire une raideur numérique. |
| Volume négatif ou nul | Maillage invalide ; arrêter avant de résoudre. |
| Nombre de couches limites | Détermine si le modèle de paroi ou le traitement low-Re est approprié. |
| Distribution de $y^+$ | Doit être compatible avec le traitement de paroi sélectionné. |
| Nombre de cellules par région | Important pour les bilans CHT et la décomposition parallèle. |

## 9. Résolution des parois et $y^+$

Le $y^+$ cible dépend du traitement des parois. Les approches low-Re visent à résoudre la sous-couche visqueuse, typiquement avec $y^+$ proche de l'unité. Les approches par fonction de paroi positionnent la première cellule dans une région logarithmique et requièrent une plage cible compatible avec la fonction de paroi et le modèle de turbulence particulieren. La cible exacte n'est pas universelle.

Utilisez :

$$
 y^+ = \frac{u_\tau y}{\nu},
$$

où $u_\tau=\sqrt{\tau_w/\rho}$ est la vitesse de frottement et $y$ est la distance de la paroi au centre de la cellule. Comme $u_\tau$ est initialement inconnu, estimez-le à partir d'une corrélation pour plaque plane ou écoulement de conduite, créez un maillage préliminaire, exécutez le cas, puis inspectez le champ réel de $y^+$.

## 10. Protocole de convergence de maillage

Une étude de maillage défendable change un paramètre de résolution à la fois et compare les sorties d'ingénierie qui comptent : perte de charge, coefficient de traînée, longueur de réattachement, coefficient d'échange thermique, nombre de Nusselt, WSS, ou indice de mélange scalaire. Comparez à la fois les grandeurs globales et les profils locaux. Un petit résidu ne prouve pas l'indépendance du maillage.

Pour les cas transitoires, effectuez une étude du pas de temps séparément. Pour les cas multiphasiques, surveillez la résolution de l'interface et la conservation de volume. Pour CHT, incluez le bilan thermique total et la continuité de la température à l'interface dans les critères de convergence.
