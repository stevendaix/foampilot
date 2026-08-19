# Procédure consolidée : reproduction Python de VMTK dans foampilot

## 1. Objectif et périmètre

Cette procédure décrit une implémentation de centerlines inspirée de VMTK, entièrement écrite en Python dans foampilot, sans importer `vmtk`, `vtkvmtk` et sans ajouter de fichiers C/C++. Elle utilise VTK standard depuis Python pour les primitives géométriques, NumPy/SciPy pour les calculs numériques, trimesh ou Shapely pour certaines opérations de maillage, et Numba comme **accélérateur optionnel**.

L’objectif est une **équivalence algorithmique mesurable** : même type de surface cappée, Delaunay volumique, dual de Voronoi, pôles, coût favorisant les grands rayons, backtracking, géométrie de centerline et sections. L’identité numérique bit-à-bit avec VMTK natif n’est pas un objectif réaliste.

La chaîne finale est :

```text
STL/VTP/MHA
→ surface lumen validée
→ nettoyage et normales
→ boucles frontières
→ caps contraints
→ seeds inlet/outlet
→ Delaunay 3D VTK
→ candidats de tétraèdres internes
→ classification géométrique et connectivité
→ sphères circonscrites
→ Voronoi dual filtré
→ pôles par distance transformée
→ graphe pondéré
→ Dijkstra ou solveur Eikonal discret
→ backtracking
→ raffinement des endpoints
→ resampling monotone et lissage contrôlé
→ géométrie centerline
→ sections et phase-lock
→ réseau de branches
→ comparaison et rapport de confiance
```

## 2. Dépendances et modes d’exécution

Installer les dépendances de base :

```bash
sudo pip3 install numpy scipy vtk trimesh scikit-image shapely mapbox-earcut networkx pytest
```

Installer Numba en option :

```bash
sudo pip3 install numba
```

Numba reste une dépendance facultative. Les sources de foampilot restent Python et le code doit fonctionner sans compilation explicite. Numba compile à la volée certains noyaux lors de l’exécution ; il ne doit pas être considéré comme une nouvelle implémentation algorithmique.

Détection :

```python
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False
```

Les modes doivent être explicites :

```python
acceleration = "numpy"          # référence déterministe
acceleration = "numba"          # noyaux accélérés
acceleration = "auto"           # numba si disponible et problème assez grand
```

Les classes VTK standard sont autorisées : `vtkCleanPolyData`, `vtkTriangleFilter`, `vtkPolyDataNormals`, `vtkFeatureEdges`, `vtkDelaunay3D`, `vtkSelectEnclosedPoints`, `vtkCutter`, `vtkOBBTree`, `vtkWindowedSincPolyDataFilter`, `vtkXMLPolyDataWriter` et `vtkXMLUnstructuredGridWriter`. Aucune classe dont le nom commence par `vtkvmtk` ne doit être utilisée.

## 3. Organisation des fichiers

Créer ou compléter :

```text
foampilot/foampilot/src/foampilot/geometry/topology/vmtk/
├── vmtksurfacepreprocess_local.py
├── vmtksurfacecapper_local.py
├── vmtkdelaunay_local.py
├── vmtkinternaltetrahedra_local.py
├── vmtkvoronoi_local.py
├── vmtkfastmarching_local.py
├── vmtkcenterlinegeometry_local.py
├── vmtkcenterlineresampling_local.py
├── vmtkcenterlinesections_local.py
├── vmtkcenterlinesnetwork_local.py
├── vmtkcenterlines_python.py
└── vmtkcompare_local.py
```

Le module `vmtkcenterlines_python.py` orchestre la chaîne. Chaque autre module doit produire un objet explicite et sérialisable : `SurfaceModel`, `BoundaryLoop`, `Cap`, `DelaunayVolume`, `InternalTetraMesh`, `VoronoiGraph`, `Pole`, `Centerline`, `Section` et `CenterlineNetwork`.

## 4. Entrées et conversion MHA

### 4.1 Surface STL/VTP

```python
import vtk


def read_polydata(path):
    path = str(path)
    if path.lower().endswith(".stl"):
        reader = vtk.vtkSTLReader()
    elif path.lower().endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif path.lower().endswith(".ply"):
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Surface inconnue: {path}")
    reader.SetFileName(path)
    reader.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(reader.GetOutput())
    return result
```

### 4.2 Volume MHA

`Aorta_voi.mha` est un volume, pas un masque lumen et pas une centerline. Pour une donnée médicale réelle, l’entrée recommandée est un masque binaire déjà segmenté. Un simple seuil peut servir à produire une surface candidate, mais ne doit pas être présenté comme une segmentation clinique.

```python

def mha_to_surface(path, threshold):
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(str(path))
    reader.Update()

    binary = vtk.vtkImageThreshold()
    binary.SetInputConnection(reader.GetOutputPort())
    binary.ThresholdBetween(float(threshold), 1.0e12)
    binary.SetInValue(1)
    binary.SetOutValue(0)
    binary.SetOutputScalarTypeToUnsignedChar()

    cubes = vtk.vtkFlyingEdges3D()
    cubes.SetInputConnection(binary.GetOutputPort())
    cubes.SetValue(0, 0.5)
    cubes.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(cubes.GetOutputPort())
    triangles.Update()
    return triangles.GetOutput()
```

Tester plusieurs seuils, inspecter les composantes connexes et vérifier l’absence de branches ou de cavités artificielles avant de poursuivre.

## 5. Prétraitement et contrôle de qualité

```python

def preprocess_surface(surface, smooth=False, flip_normals=False):
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surface)
    clean.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()

    current = triangles.GetOutput()
    if smooth:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(current)
        smoother.SetNumberOfIterations(10)
        smoother.SetPassBand(0.08)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.Update()
        current = smoother.GetOutput()

    copy = vtk.vtkPolyData()
    copy.ShallowCopy(current)
    copy.GetPointData().SetNormals(None)
    copy.GetCellData().SetNormals(None)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(copy)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.SetFlipNormals(bool(flip_normals))
    normals.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(normals.GetOutput())
    return result
```

Le lissage doit rester faible. Il est destiné à supprimer le bruit de Marching Cubes, pas à déplacer une sténose. Comparer systématiquement le volume, l’aire et la distance à la surface non lissée.

Le rapport de surface doit contenir : nombre de points, nombre de triangles, arêtes frontières, arêtes non-manifold, composantes connexes, volume signé et diamètre de boîte englobante.

## 6. Boucles frontières et caps

Détecter les arêtes frontières avec `vtkFeatureEdges`, puis reconstruire les boucles dans un graphe d’arêtes. Avant de parcourir une boucle, fusionner les sommets coïncidents et traiter explicitement les nœuds de degré différent de deux. Une boucle avec degré supérieur à deux doit être réparée ou rejetée, jamais parcourue arbitrairement.

Pour chaque boucle :

```text
BoundaryId
points ordonnés
centre barycentrique
normale PCA
périmètre
aire projetée
planéité
```

Pour un contour convexe et quasi planaire, la triangulation en éventail est acceptable. Pour un contour concave, utiliser impérativement une triangulation contrainte avec `mapbox-earcut` ou Shapely ; `scipy.spatial.Delaunay` libre peut créer des triangles hors contour.

```python
from scipy.spatial import Delaunay
import numpy as np


def pca_frame(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    return center, vh[0], vh[1], vh[2]


def planar_cap(points):
    center, u, v, normal = pca_frame(points)
    uv = np.column_stack(((points - center) @ u,
                          (points - center) @ v))
    tri = Delaunay(uv)
    faces = tri.simplices.copy()
    for k, face in enumerate(faces):
        a, b, c = points[face]
        if np.dot(np.cross(b-a, c-a), normal) < 0:
            faces[k, 1], faces[k, 2] = faces[k, 2], faces[k, 1]
    return points, faces, center, normal
```

Chaque cap doit être validé : aire positive, triangles non dégénérés, normale cohérente, centre situé dans le polygone, absence d’intersection avec la paroi et distance acceptable avec les autres surfaces. Conserver `CapCenterId` et le lien vers `BoundaryId`.

## 7. Delaunay et classification interne

Construire la tessellation avec VTK Python :

```python

def build_delaunay(capped_surface, tolerance=1e-3):
    d = vtk.vtkDelaunay3D()
    d.CreateDefaultLocator()
    d.SetInputData(capped_surface)
    d.SetTolerance(float(tolerance))
    d.Update()
    result = vtk.vtkUnstructuredGrid()
    result.DeepCopy(d.GetOutput())
    return result
```

La classification interne est effectuée en deux niveaux.

### Niveau 1 : présélection vectorisée

Extraire les tétraèdres et leurs quatre sommets dans des tableaux NumPy. Calculer les centroïdes par moyenne vectorisée. Tester les centroïdes et les sommets avec `vtkSelectEnclosedPoints` en batch, ou avec un masque voxel si la surface est grande.

### Niveau 2 : validation géométrique et topologique

Pour chaque candidat, tester le centroïde, les quatre sommets, le centre de la sphère circonscrite, des points sur les six arêtes, la qualité volumique et l’absence d’intersection avec la surface. Rejeter les slivers :

```python
volume = abs(np.linalg.det(tetra[1:] - tetra[0])) / 6.0
edge_lengths = [np.linalg.norm(tetra[i] - tetra[j])
                for i in range(4) for j in range(i+1, 4)]
quality = volume / max(max(edge_lengths)**3, 1e-12)
```

Après le filtrage, construire la connectivité entre tétraèdres par faces partagées et conserver la composante volumique contenant le seed inlet. La connectivité est indispensable : un tétraèdre localement plausible mais isolé ou relié à l’extérieur doit être supprimé.

## 8. Sphères circonscrites et Voronoi dual

Pour un tétraèdre `p0,p1,p2,p3`, le centre `c` résout :

```text
2 (pi-p0)·c = |pi|²-|p0|², i=1,2,3
```

Calculer les sphères par blocs NumPy. Numba est autorisé pour cette étape lorsque le nombre de tétraèdres est important, mais la version NumPy reste la référence.

Le Voronoi dual est construit en groupant les faces Delaunay :

```python
faces = np.stack([
    tets[:, [0,1,2]], tets[:, [0,1,3]],
    tets[:, [0,2,3]], tets[:, [1,2,3]],
], axis=1).reshape(-1, 3)
faces = np.sort(faces, axis=1)
unique_faces, inverse, counts = np.unique(
    faces, axis=0, return_inverse=True, return_counts=True
)
```

Une face apparaissant deux fois relie les deux centres circonscrits des tétraèdres adjacents. Filtrer ensuite : centres extérieurs, rayons aberrants, tétraèdres dégénérés, arêtes duales traversant la paroi, branches mortes et zones parasites proches des caps.

Exporter deux graphes :

```text
voronoi_raw.vtp
voronoi_filtered.vtp
```

Le tableau de points doit comprendre `MaximumInscribedSphereRadius`.

## 9. Pôles et seeds

La sélection des pôles doit être guidée par la distance transformée du volume interne. Construire un masque voxel à résolution contrôlée, puis :

```python
from scipy.ndimage import distance_transform_edt, maximum_filter

clearance = distance_transform_edt(mask, sampling=spacing)
local_maxima = (clearance == maximum_filter(clearance, size=5)) & mask
```

Associer les maxima locaux aux nœuds Voronoi par proximité, direction intérieure, rayon et continuité. Pour chaque cap, limiter la recherche à un corridor orienté par la normale interne. Cette contrainte évite qu’un maximum situé dans un anévrisme ou un bulbe soit choisi à la place du pôle du col.

Effectuer un raffinement local du seed en recherchant une position qui augmente la clearance tout en restant proche de l’ouverture et du graphe Voronoi.

## 10. Coût du graphe et chemin

Ne pas confondre les fonctions suivantes :

| Nom | Formule | Signification |
|---|---|---|
| champ rayon | `R` ou `1/R` | propriété locale |
| coût discret | `L/Rmoyen` | approximation simple |
| coût intégré | `∫ ds/R(s)` | recommandation principale |
| Eikonal | `|∇T|=1/R` | approximation avancée |

Utiliser par défaut le coût intégré par quadrature de Gauss :

```python

def edge_cost(p0, p1, r0, r1, floor=1e-6):
    xi = (-0.7745966692, 0.0, 0.7745966692)
    wi = (0.5555555556, 0.8888888889, 0.5555555556)
    length = np.linalg.norm(p1-p0)
    total = 0.0
    for x, w in zip(xi, wi):
        a = 0.5*(x+1.0)
        r = (1-a)*r0 + a*r1
        total += w / max(r, floor)
    return 0.5*length*total
```

Construire une matrice creuse symétrique et exécuter Dijkstra avec prédécesseurs :

```python
from scipy.sparse.csgraph import dijkstra

distance, predecessor = dijkstra(
    graph, indices=source_id,
    directed=False, return_predecessors=True
)
```

Le backtracking doit remonter les prédécesseurs et vérifier les cycles. Pour le backend `python_eikonal`, ajouter une relaxation sur le graphe et choisir au retour le voisin qui minimise la solution locale plus le coût d’arête. Le résultat reste une approximation discrète de l’Eikonal, pas un fast marching continu exact.

## 11. Numba : accélération contrôlée

Numba ne doit accélérer que les noyaux mesurés comme goulots. Les opérations déjà optimisées par VTK, SciPy ou NumPy ne doivent pas être réécrites inutilement.

| Noyau | NumPy/SciPy de référence | Numba recommandé |
|---|---|---|
| Sphères circonscrites | calcul par blocs | Oui |
| Échantillonnage masque voxel | indexation vectorisée | Oui si massif |
| Faces Delaunay | `np.unique(axis=0)` | préparation possible |
| Coûts d’arêtes | vectorisation | Oui |
| Delaunay | `vtkDelaunay3D` | Non |
| Dijkstra sparse | SciPy | Non en première intention |
| KD-tree | `cKDTree` | Non |
| EDT | `scipy.ndimage` | Non |
| I/O et VTK | VTK | Non |

Exemple de noyau :

```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def edge_costs_numba(points, radii, edges, radius_floor):
    result = np.empty(edges.shape[0], dtype=np.float64)
    for k in prange(edges.shape[0]):
        i, j = edges[k]
        length = np.sqrt(np.sum((points[j]-points[i])**2))
        radius = 0.5*(radii[i]+radii[j])
        result[k] = length / max(radius, radius_floor)
    return result
```

Chaque noyau Numba doit avoir une version NumPy de référence, un test d’erreur relative et un benchmark après échauffement. Ne pas utiliser `parallel=True` pour le mode de comparaison déterministe.

## 12. Resampling et géométrie

Ne pas appliquer une `CubicSpline` libre en première intention. Elle peut overshooter et sortir du lumen. Utiliser :

```text
chemin brut
→ suppression des doublons
→ resampling linéaire par longueur d’arc
→ lissage Taubin ou Laplacien contraint
→ projection ou validation dans le volume
```

Les endpoints doivent rester fixes. Chaque point lissé doit conserver une clearance positive, rester proche du chemin brut et respecter une courbure maximale.

Calculer tangentes et repères par transport parallèle, non par Frenet seul. Le transport parallèle doit être testé sur ligne droite, cercle, hélice, coude et bifurcation. Les arrays de sortie sont :

```text
MaximumInscribedSphereRadius
Abscissas
Curvature
Torsion
Tortuosity
FrenetTangent
ParallelTransportNormals
ParallelTransportBinormals
```

## 13. Sections et lofts

À chaque station, créer un plan tangent avec le repère de transport parallèle et utiliser `vtkCutter` ou une intersection locale avec `vtkOBBTree`. Pour éviter 1 000 intersections du maillage complet, limiter la recherche aux cellules proches de la station lorsque le coût devient important.

`vtkCutter` peut produire plusieurs contours. Sélectionner le contour par cascade : fermeture, distance au centre, inclusion du centre, aire plausible et continuité avec la station précédente. Une coupe ambiguë doit être rejetée ou recalculée avec une station intermédiaire.

Rééchantillonner tous les contours avec le même nombre de points, puis verrouiller le décalage cyclique et le sens par rapport au contour précédent. Ne jamais recentrer les sections indépendamment dans XY ni inventer une coordonnée `z`.

## 14. Réseau de branches

Calculer Delaunay et Voronoi une seule fois pour toutes les branches. Les nœuds candidats de bifurcation doivent avoir un degré supérieur ou égal à trois, une clearance continue et des directions sortantes suffisamment séparées.

Utiliser une structure :

```python
CenterlineNetwork(
    points=..., edges=..., nodes=...,
    centerline_ids=..., group_ids=...,
    tract_ids=..., blanking=...
)
```

Utiliser `GroupId=0` pour le tronc inlet jusqu’à la première bifurcation, puis un identifiant par branche sortante. `TractId` désigne un chemin inlet/outlet.

## 15. Objet résultat et confiance

La sortie principale doit être plus riche qu’une simple polyline :

```python
@dataclass
class CenterlineResult:
    points: np.ndarray
    radii: np.ndarray
    abscissas: np.ndarray
    tangents: np.ndarray
    curvature: np.ndarray
    torsion: np.ndarray
    branches: list
    quality: dict
    diagnostics: dict
```

Calculer :

```text
surface_quality
delaunay_quality
internal_volume_quality
voronoi_quality
pole_quality
path_quality
radius_continuity
curvature_quality
section_quality
topology_quality
```

Retourner un statut `PASS`, `WARNING` ou `FAIL`. Refuser automatiquement une centerline qui sort du volume, possède une chaîne de prédécesseurs invalide, contient des rayons nuls ou traverse une zone de sections ambiguës trop importante.

## 16. Tests et comparaison

Commencer par des géométries synthétiques : tube droit, tube courbe, coude en U, hélice et bifurcation en Y. Vérifier longueur, tortuosité, rayon analytique, topologie et continuité du repère.

Ensuite récupérer les données officielles :

```bash
git clone https://github.com/vmtk/vmtk-test-data.git
export VMTK_TEST_DATA=/chemin/vers/vmtk-test-data
```

Les références utiles sont :

```text
vmtk-test-data/aorta-surface.vtp
vmtk-test-data/aorta-surface-open-ends.stl
vmtk-test-data/aorta-surface-branch-split.vtp
vmtk-test-data/aorta-centerline.vtp
vmtk-test-data/aorta-centerline-branches.vtp
vmtk-test-data/centerlinereference/
```

Comparer : distance moyenne, Hausdorff symétrique, longueur, tortuosité, rayon, nombre de branches, topologie, continuité des tangentes et arrays VTP. Comparer les modes `numpy` et `numba` avec une tolérance géométrique et non seulement une tolérance scalaire.

## 17. Ordre de développement final

| Phase | Travail | Critère de sortie |
|---:|---|---|
| A | Surface, boucles, caps et contrôles | surface fermée, caps validés |
| B | Delaunay et classification à deux niveaux | volume interne connecté |
| C | Circonsphères et Voronoi vectorisés | graphe filtré et rayons positifs |
| D | Pôles EDT et raffinement | seeds stables |
| E | Dijkstra, coût intégré et Eikonal discret | chemin valide et reproductible |
| F | Resampling, géométrie et transport parallèle | arrays continus |
| G | Sections et phase-lock | aucune torsion artificielle |
| H | Réseau de branches et confiance | topologie validée |
| I | Backend Numba et benchmarks | accélération sans divergence |
| J | Comparaison officielle et pipeline aorte | rapport reproductible |

## 18. Commande cible

```bash
python3 -m foampilot.geometry.topology.vmtk.vmtkcenterlines_python \
  --input aorta-surface-open-ends.stl \
  --backend python_eikonal \
  --acceleration auto \
  --resampling-step 1.0 \
  --output aorta-centerline-python.vtp \
  --voronoi-output aorta-voronoi-python.vtp \
  --delaunay-output aorta-delaunay-python.vtu \
  --diagnostics-output aorta-centerline-diagnostics.json
```

Le rapport doit indiquer le backend, la disponibilité de Numba, les temps de chaque phase, la mémoire maximale, les métriques de qualité et les avertissements.

## Références

[1]: https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkcenterlines.py "Wrapper officiel VMTK centerlines"

[2]: https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkdelaunayvoronoi.py "Wrapper officiel VMTK Delaunay/Voronoi"

[3]: https://vmtk.github.io/tutorials/Centerlines.html "Tutoriel officiel VMTK centerlines"

[4]: https://github.com/vmtk/vmtk-test-data "Données de comparaison VMTK"
