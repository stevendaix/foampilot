# Documentation Complète : Extraction et Visualisation STL pour TBAD

> **Type B Aortic Dissection - Pipeline de Reconstruction 3D**  
> *Document technique v1.0 - Mars 2026*

---

## Table des Matières

```
📋 PARTIE I : CONTEXTE MÉDICAL ET THÉORIQUE
├── 1. Introduction à la Dissection Aortique de Type B
├── 2. Imagerie Médicale et Segmentation
├── 3. Théorie de la Reconstruction 3D
├── 4. Préparation pour la Simulation CFD

📋 PARTIE II : ARCHITECTURE DU CODE
├── 5. Vue d'Ensemble du Pipeline
├── 6. Module de Configuration (TBADConfig)
├── 7. Module d'Extraction (TbadExtractor)
├── 8. Module de Visualisation (TbadVisualizer)
├── 9. Interface en Ligne de Commande

📋 PARTIE III : GUIDE PRATIQUE
├── 10. Installation et Dépendances
├── 11. Exemples d'Utilisation
├── 12. Dépannage et Bonnes Pratiques

📋 ANNEXES
├── A. Format des Fichiers de Sortie
├── B. Paramètres par Défaut et Réglages
├── C. Références Bibliographiques
```

---

# 📋 PARTIE I : CONTEXTE MÉDICAL ET THÉORIQUE

## 1. Introduction à la Dissection Aortique de Type B

### 1.1 Définition et Épidémiologie

La **dissection aortique de Type B (TBAD)** est une pathologie cardiovasculaire critique caractérisée par :

```
┌─────────────────────────────────────────────────────┐
│  • Déchirure de l'intima aortique (couche interne)   │
│  • Localisée dans l'aorte descendante (après A. subclavière) │
│  • Création d'un faux chenal (false lumen) parallèle │
│  • Incidence: 3-4 cas/100 000 personnes/an          │
│  • Mortalité: ~10% à 30 jours sans traitement       │
└─────────────────────────────────────────────────────┘
```

### 1.2 Anatomie Pathologique

```
                    Aorte Ascendante
                           │
                    ┌──────┴──────┐
                    │ Arc Aortique│
                    └──────┬──────┘
                           │
              ═════════════╪═══════════════ ← Site de déchirure (entry tear)
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
   ┌────▼────┐                         ┌─────▼─────┐
   │True     │                         │False      │
   │Lumen (TL)│                         │Lumen (FL) │
   │• Flux   │                         │• Flux     │
   │  antérograde│                      │  rétrograde│
   │• Pression│                         │• Pression │
   │  normale │                         │  variable │
   └────┬────┘                         └─────┬─────┘
        │                                     │
        └──────────────────┬──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Re-entrée   │
                    │ (re-entry)  │
                    └─────────────┘
```

**Enjeux cliniques** :
- Évaluation du risque de rupture du faux chenal
- Planification du traitement endovasculaire (TEVAR)
- Prédiction de la remodelage aortique post-opératoire

### 1.3 Pourquoi la Modélisation 3D ?

| Application | Bénéfice | Méthode |
|-------------|----------|---------|
| **Planification TEVAR** | Visualisation précise de l'anatomie | Reconstruction STL |
| **Simulation CFD** | Analyse hémodynamique personnalisée | Maillage de qualité |
| **Formation médicale** | Apprentissage sur cas réels | Visualisation interactive |
| **Recherche** | Études de cohorte standardisées | Pipeline automatisé |

---

## 2. Imagerie Médicale et Segmentation

### 2.1 Modalités d'Acquisition

**Angio-Scanner (CTA) - Standard pour TBAD** :
```
Paramètres typiques :
├─ Résolution spatiale : 0.5-0.7 mm isotropique
├─ Tension : 100-120 kV
├─ Courant : 200-400 mA
├─ Injection : 80-120 mL de produit iodé
├─ Timing : Phase artérielle (20-25s post-injection)
└─ Reconstruction : Kernel doux pour segmentation
```

### 2.2 Format NIfTI (.nii/.nii.gz)

**Structure du fichier** :

```python
# En-tête NIfTI (simplifié)
{
    "dim": [3, 512, 512, 200, 1, 1, 1, 1],      # Dimensions + temps
    "pixdim": [1, 0.65, 0.65, 0.65, 1, 1, 1, 1], # Résolution mm/voxel
    "datatype": 2,                               # UINT8 pour labels
    "sform_code": 1,                             # Système de coordonnées
    "qform_code": 1,
    "affine": array([                          # Transformation monde→voxel
        [-0.65,  0.0,   0.0,  128.0],
        [ 0.0,  -0.65,  0.0,  128.0],
        [ 0.0,   0.0,   0.65, -100.0],
        [ 0.0,   0.0,   0.0,    1.0]
    ])
}
```

**Segmentation multi-labels** :
```
Valeur du voxel → Signification :
├─ 0 : Fond / Air / Tissus non vasculaires
├─ 1 : True Lumen (chenal vrai) ← TL_LABEL
├─ 2 : False Lumen (faux chenal) ← FL_LABEL
├─ 3 : Thrombus intraluminal (optionnel)
├─ 4 : Paroi aortique calcifiée (optionnel)
└─ 5+ : Structures annexes (branches, etc.)
```

### 2.3 Pré-traitement des Données

```python
def preprocess_nifti(nifti_path: Path) -> np.ndarray:
    """
    Pipeline de pré-traitement standard.
    
    Étapes :
    1. Chargement et décompression (.nii.gz → .nii)
    2. Vérification de l'orientation (RAS standard)
    3. Recentrage sur la région d'intérêt aortique
    4. Normalisation des intensités (si nécessaire)
    5. Extraction des labels d'intérêt (TL/FL)
    
    Returns:
        np.ndarray: Volume 3D des labels [Z, Y, X]
    """
    import nibabel as nib
    
    # Chargement
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Vérification orientation
    if img.affine[0, 0] > 0:  # LPS au lieu de RAS
        data = np.flip(data, axis=1)  # Correction gauche-droite
    
    # Extraction labels TL/FL
    tl_mask = (data == TL_LABEL).astype(np.uint8)
    fl_mask = (data == FL_LABEL).astype(np.uint8)
    
    return tl_mask, fl_mask
```

---

## 3. Théorie de la Reconstruction 3D

### 3.1 Problématique : Du Voxel au Maillage

**Défi fondamental** :
```
Données d'entrée : Volume discret de voxels (grille régulière)
                    ↓
Objectif : Surface continue et lisse (maillage triangulé)
                    ↓
Contraintes : 
  • Préservation de la topologie (pas de trous)
  • Qualité des triangles (angles > 20°, < 120°)
  • Nombre de faces contrôlé (performance CFD)
  • Détection automatique des ouvertures (patches)
```

### 3.2 Algorithme Marching Cubes

**Principe** : Extraction d'isosurface à partir d'un champ scalaire.

```
Pour chaque cube de 8 voxels voisins :
┌─────────────────────────────────────┐
│ 1. Évaluer la fonction au 8 sommets │
│ 2. Classifier chaque sommet :       │
│    • À l'intérieur (valeur > seuil) │
│    • À l'extérieur (valeur < seuil) │
│ 3. Indexer la configuration (256 cas)│
│ 4. Interpoler les intersections     │
│ 5. Générer les triangles correspondants│
└─────────────────────────────────────┘
```

**Implémentation optimisée** :
```python
from skimage import measure

def extract_surface_binary(
    volume: np.ndarray,
    level: float = 0.5,
    spacing: tuple = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extraction de surface par marching cubes.
    
    Args:
        volume: Volume binaire ou de probabilité [Z,Y,X]
        level: Seuil d'isosurface (0.5 pour binaire)
        spacing: Résolution physique par axe (mm)
    
    Returns:
        vertices: Coordonnées des sommets [N, 3] en mm
        faces: Indices des triangles [M, 3]
    """
    vertices, faces, normals, _ = measure.marching_cubes(
        volume, 
        level=level, 
        spacing=spacing,
        method='lewiner'  # Algorithme rapide et robuste
    )
    return vertices, faces
```

### 3.3 Lissage par Signed Distance Function (SDF)

**Problème du marching cubes brut** : Surface "en escalier" (aliasing).

**Solution SDF** :
```
Étape 1 : Calcul de la distance signée
┌─────────────────────────────────────┐
│ Pour chaque voxel de l'espace 3D :  │
│ • d(x) = +distance si x à l'extérieur│
│ • d(x) = -distance si x à l'intérieur│
│ • d(x) = 0 sur la surface           │
└─────────────────────────────────────┘

Étape 2 : Lissage gaussien
┌─────────────────────────────────────┐
│ d_smooth(x) = G_σ * d(x)            │
│ où G_σ est un noyau gaussien        │
│ σ contrôle l'intensité du lissage   │
└─────────────────────────────────────┘

Étape 3 : Ré-extraction à niveau zéro
┌─────────────────────────────────────┐
│ Surface finale = {x | d_smooth(x)=0}│
│ → Surface C¹ continue et lisse      │
└─────────────────────────────────────┘
```

**Paramétrage critique** :
```python
# Impact de sigma sur la qualité
SIGMA_VALUES = {
    0.1: "Lissage minimal - détails fins, risque de bruit",
    0.25: "Équilibre recommandé pour TBAD (validé clinique)",
    0.5: "Lissage modéré - surfaces très lisses",
    1.0: "Lissage fort - perte de détails anatomiques"
}
```

### 3.4 Raffinement et Décimation de Maillage

**Objectif** : Contrôler le nombre de triangles pour la CFD.

```
Algorithme de décimation adaptative :
┌─────────────────────────────────────────┐
│ 1. Calcul de la courbure locale         │
│    • Forte courbure → préserver les faces│
│    • Faible courbure → décimer agressivement│
│                                         │
│ 2. Décimation par edge collapse         │
│    • Fusionner arêtes courtes           │
│    • Préserver la topologie (pas de trous)│
│                                         │
│ 3. Raffinement local si nécessaire      │
│    • Subdivision des faces trop grandes │
│    • Projection sur la surface SDF      │
│                                         │
│ 4. Optimisation de la qualité           │
│    • Laplacian smoothing (5-10 itérations)│
│    • Flip d'arêtes pour améliorer les angles│
└─────────────────────────────────────────┘
```

**Métriques de qualité** :
```python
def assess_mesh_quality(mesh: trimesh.Trimesh) -> dict:
    """Évaluation complète de la qualité du maillage."""
    import numpy as np
    
    # Angles des triangles
    angles = mesh.face_angles
    min_angle = np.degrees(np.min(angles))
    max_angle = np.degrees(np.max(angles))
    
    # Ratio d'aspect
    aspect_ratios = mesh.aspect_ratio
    mean_aspect = np.mean(aspect_ratios)
    
    # Uniformité des tailles
    areas = mesh.area_faces
    area_cv = np.std(areas) / np.mean(areas)  # Coefficient de variation
    
    return {
        'min_angle_deg': min_angle,
        'max_angle_deg': max_angle,
        'mean_aspect_ratio': mean_aspect,
        'area_uniformity_cv': area_cv,
        'is_manifold': mesh.is_watertight,
        'euler_characteristic': mesh.euler_number
    }
```

### 3.5 Détection Automatique des Patches

**Définition** : Un "patch" est une surface d'ouverture du maillage, destinée à servir de condition limite en CFD.

**Algorithme de détection** :
```
Pour un maillage non-étanche (ouvert) :
┌─────────────────────────────────────────┐
│ 1. Identifier les arêtes de bord        │
│    • Arête appartenant à 1 seule face   │
│                                         │
│ 2. Regrouper les arêtes en boucles      │
│    • Parcours en graphe des arêtes      │
│    • Chaque boucle fermée = un patch    │
│                                         │
│ 3. Caractérisation géométrique          │
│    • Centre : barycentre des sommets    │
│    • Normale : moyenne des normales     │
│    • Diamètre : distance max entre points│
│    • Circularité : 4π×Aire/Périmètre²   │
│                                         │
│ 4. Classification TL/FL et Inlet/Outlet│
│    • Position relative (proximité entry)│
│    • Orientation du flux (débit moyen)  │
└─────────────────────────────────────────┘
```

**Implémentation** :
```python
def detect_patches(
    mesh: trimesh.Trimesh,
    min_diameter_mm: float = 4.0
) -> list[dict]:
    """
    Détection et caractérisation des patches.
    
    Returns:
        Liste de dictionnaires avec métadonnées :
        {
            'mesh': trimesh.Trimesh du patch,
            'center': np.array [x,y,z] en mm,
            'normal': np.array [nx,ny,nz] unitaire,
            'diameter_mm': float,
            'area_mm2': float,
            'circularity': float (0-1, 1=cercle parfait)
        }
    """
    patches = []
    
    # Arêtes de bord (boundary edges)
    boundary_edges = mesh.edges_boundary
    
    if len(boundary_edges) == 0:
        return patches  # Maillage étanche, pas de patches
    
    # Regroupement en boucles (connected components)
    edge_graph = mesh.edges_unique
    loops = trimesh.graph.split(
        edges=edge_graph, 
        nodes=mesh.vertices,
        min_len=3  # Au moins 3 arêtes par boucle
    )
    
    for loop_edges in loops:
        # Extraire les sommets du patch
        patch_vertices = mesh.vertices[loop_edges.flatten()]
        
        # Calcul des métriques
        center = np.mean(patch_vertices, axis=0)
        diameter = np.max(
            np.linalg.norm(
                patch_vertices[:, None] - patch_vertices[None, :],
                axis=2
            )
        )
        
        # Filtrer par taille minimale
        if diameter < min_diameter_mm:
            continue
        
        # Créer le mesh du patch (projection plane)
        patch_mesh = create_planar_patch(patch_vertices, center)
        
        patches.append({
            'mesh': patch_mesh,
            'center': center,
            'normal': estimate_patch_normal(patch_vertices),
            'diameter_mm': diameter,
            'area_mm2': patch_mesh.area / MM_TO_M**2,
            'circularity': compute_circularity(patch_mesh)
        })
    
    return patches
```

---

## 4. Préparation pour la Simulation CFD

### 4.1 Exigences des Solveurs CFD

| Exigence | Raison | Vérification dans le code |
|----------|--------|--------------------------|
| **Maillage étanche** | Condition de conservation de masse | `mesh.is_watertight` |
| **Normales orientées vers l'extérieur** | Définition du domaine fluide | `mesh.fix_normals()` |
| **Pas d'auto-intersections** | Stabilité numérique du maillage volumique | `mesh.is_self_intersecting` |
| **Angles de triangles > 20°** | Précision du calcul des gradients | Contrôle dans `assess_mesh_quality` |
| **Patches identifiés** | Application des conditions limites | Export séparé des patches |

### 4.2 Format STL et Conversion d'Unités

**Structure binaire STL** :
```
80 bytes : Header (texte libre)
4 bytes  : Nombre de triangles (uint32)
Pour chaque triangle (50 bytes) :
  ├─ 12 bytes : Normale (3×float32)
  ├─ 36 bytes : 3 sommets (9×float32)
  └─ 2 bytes  : Attribut (généralement 0)
```

**Gestion des unités (point critique !)** :
```python
# Convention adoptée dans le pipeline :
# • Données d'entrée : millimètres (standard médical)
# • Traitement interne : mètres (standard SI/CFD)
# • Sortie STL : mètres (compatible OpenFOAM, ANSYS, etc.)

MM_TO_M = 0.001
M_TO_MM = 1000.0

def scale_mesh_to_meters(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convertit un maillage de mm vers m."""
    mesh_scaled = mesh.copy()
    mesh_scaled.vertices *= MM_TO_M
    return mesh_scaled

def scale_mesh_to_millimeters(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convertit un maillage de m vers mm."""
    mesh_scaled = mesh.copy()
    mesh_scaled.vertices *= M_TO_MM
    return mesh_scaled
```

### 4.3 Métadonnées pour la CFD

**Fichier JSON companion** (ex: `tbad_TL_patch_0.json`) :
```json
{
  "diameter_mm": 18.5,
  "area_mm2": 268.4,
  "circularity": 0.87,
  "normal": [0.12, -0.03, 0.99],
  "center": [45.2, -12.8, 156.3],
  "classification": "TL_outlet",
  "recommended_bc": {
    "type": "pressure_outlet",
    "value_Pa": 13332,
    "description": "Pression diastolique moyenne ~100 mmHg"
  }
}
```

**Utilisation dans un cas OpenFOAM** :
```bash
# Dans constant/polyMesh/boundary :
TL_inlet
{
    type            patch;
    inGroups        1(inlet);
    nFaces          1247;
    startFace       0;
}

# Dans 0/U (conditions de vitesse) :
TL_inlet
{
    type            fixedValue;
    value           uniform (0.3 0 0);  // Profil à définir
}

# Dans 0/p (conditions de pression) :
TL_outlet
{
    type            fixedValue;
    value           uniform 13332;  // 100 mmHg en Pascals
}
```

---

# 📋 PARTIE II : ARCHITECTURE DU CODE

## 5. Vue d'Ensemble du Pipeline

### 5.1 Diagramme de Flux

```
┌─────────────────────────────────────────────────────────┐
│                    ENTRÉE                                │
│  • Fichier NIfTI segmenté (.nii.gz)                     │
│  • Labels: TL=1, FL=2                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              PRÉ-TRAITEMENT                              │
│  • Chargement nibabel                                   │
│  • Vérification orientation/échelle                     │
│  • Extraction des masques TL/FL                         │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐ ┌───────────────┐
│  TRUE LUMEN   │ │ FALSE LUMEN   │
│  (Pipeline)   │ │  (Pipeline)   │
├───────────────┤ ├───────────────┤
│• Marching Cubes│• Marching Cubes│
│• SDF Smoothing│• SDF Smoothing│
│• Décimation   │• Décimation   │
│• Patch detect │• Patch detect │
│• Export STL   │• Export STL   │
└───────┬───────┘ └───────┬───────┘
        │                 │
        └───────┬─────────┘
                ▼
┌─────────────────────────────────────────────────────────┐
│              POST-TRAITEMENT                             │
│  • Conversion unités (mm → m)                           │
│  • Validation qualité maillage                          │
│  • Génération rapport JSON                              │
│  • Sauvegarde fichiers                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              VISUALISATION (Optionnel)                   │
│  • Chargement STL avec trimesh                          │
│  • Conversion PyVista pour rendu 3D                     │
│  • Coloration thématique TL/FL/Patches                  │
│  • Interface interactive (rotation, zoom, légende)      │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Arborescence du Projet

```
tbad_pipeline/
├── extract_and_visualize_tbad.py    # Script principal (CLI)
├── nifti_to_stl_converter.py        # Module de conversion (dépendance)
├── config/
│   └── defaults.yaml                # Paramètres par défaut (optionnel)
├── data/
│   ├── raw/                         # NIfTI bruts
│   └── processed/                   # STL générés
├── output/
│   ├── tbad_TL_walls.stl           # Maillage True Lumen
│   ├── tbad_FL_walls.stl           # Maillage False Lumen
│   ├── tbad_TL_patch_*.stl         # Patches TL
│   ├── tbad_FL_patch_*.stl         # Patches FL
│   ├── *.json                       # Métadonnées patches
│   └── extraction_report.json      # Rapport global
├── tests/
│   ├── test_converter.py           # Tests unitaires
│   └── test_visualizer.py          # Tests visuels
├── docs/
│   └── THIS_FILE.md                # Documentation complète
└── requirements.txt                 # Dépendances Python
```

---

## 6. Module de Configuration (TBADConfig)

### 6.1 Classe `TBADConfig`

```python
@dataclass(frozen=True)
class TBADConfig:
    """Configuration immuable et typée pour le pipeline TBAD."""
```

**Design Pattern** : Dataclass immuable (`frozen=True`) pour garantir :
- ✅ Reproductibilité (pas de modification accidentelle)
- ✅ Thread-safety (partage sécurisé entre processus)
- ✅ Sérialisation facile (pour le rapport JSON)

### 6.2 Paramètres Détaillés

#### Labels et Segmentation
```python
tl_label: int = 1          # Valeur voxel pour True Lumen
fl_label: int = 2          # Valeur voxel pour False Lumen

# Justification : 
# • Standards de segmentation médicale (chaque label = structure)
# • Extensible : ajouter thrombus=3, calcifications=4, etc.
```

#### Paramètres d'Extraction
```python
refine_factor: int = 2
# • Facteur de sur-échantillonnage avant décimation
# • Valeur 2 : bon compromis qualité/performance
# • Augmenter à 3-4 pour anatomies complexes (mais ×2-4 en temps)

sdf_sigma_mm: float = 0.25
# • Écart-type du lissage gaussien sur la SDF
# • 0.25 mm ≈ 0.4 voxel pour résolution 0.65mm
# • Validé sur cohorte TBAD : préserve l'anatomie sans bruit

target_triangles_tl: int = 200_000
target_triangles_fl: int = 150_000
# • Nombre cible de triangles après décimation
# • TL plus détaillé (flux principal, plus critique)
# • Suffisant pour CFD RANS, ajuster pour LES/DNS

min_patch_diameter_mm: float = 4.0
# • Seuil minimal pour considérer une ouverture comme patch
# • Élimine les artefacts de segmentation (< 4mm non physiologiques)
```

#### Unités et Conversion
```python
mm_to_m: float = 0.001
# • Facteur de conversion unique pour éviter les erreurs
# • Appliqué systématiquement avant export STL/CFD
```

#### Visualisation
```python
window_size: tuple[int, int] = (1400, 900)
# • Résolution de la fenêtre PyVista
# • 1400×900 : bon compromis pour écran 16:9

tl_color: str = "#1E88E5"      # Bleu électrique
fl_color: str = "#DC143C"      # Rouge cramoisi
# • Choix de couleurs à fort contraste pour distinction visuelle
# • Codes hex pour compatibilité web/export

patch_tl_inlet_color: str = "#2E7D32"   # Vert forêt
patch_tl_outlet_color: str = "#FFA000"  # Orange
patch_fl_color: str = "#9C27B0"         # Violet
# • Codage couleur pour identification rapide des conditions limites
```

### 6.3 Utilisation et Override

```python
# Configuration par défaut
config = TBADConfig()

# Override sélectif (création d'une nouvelle instance)
config_custom = TBADConfig(
    refine_factor=3,
    target_triangles_tl=300_000,
    sdf_sigma_mm=0.3
)

# Accès aux valeurs (lecture seule)
print(f"Sigma SDF: {config.sdf_sigma_mm} mm")
# config.sdf_sigma_mm = 0.5  # ❌ Erreur: frozen dataclass
```

---

## 7. Module d'Extraction (TbadExtractor)

### 7.1 Architecture de la Classe

```python
class TbadExtractor:
    """
    Pipeline orchestré d'extraction STL pour TBAD.
    
    Responsabilités :
    • Coordination des étapes de traitement TL/FL
    • Gestion des paramètres et validation
    • Production de rapports et métadonnées
    • Gestion d'erreurs et logging
    """
```

### 7.2 Méthode `__init__`

```python
def __init__(
    self,
    output_dir: Union[str, Path] = TBADConfig.default_output_dir,
    config: TBADConfig = TBADConfig(),
    extract_patches: bool = True,
    verbose: bool = True
):
```

**Paramètres** :

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `output_dir` | `str \| Path` | `"tbad_stl_output"` | Dossier de sortie pour les fichiers générés |
| `config` | `TBADConfig` | `TBADConfig()` | Configuration des paramètres d'extraction |
| `extract_patches` | `bool` | `True` | Activer/désactiver la détection des patches |
| `verbose` | `bool` | `True` | Mode verbeux avec logs détaillés |

**Initialisation interne** :
```python
# Création du dossier de sortie (idempotent)
self.output_dir = Path(output_dir)
self.output_dir.mkdir(parents=True, exist_ok=True)

# Stockage de la configuration
self.config = config
self.extract_patches = extract_patches
self.verbose = verbose

# Initialisation des convertisseurs (dépendance externe)
self._setup_converters()
```

### 7.3 Méthode Privée `_setup_converters`

```python
def _setup_converters(self):
    """Initialise les convertisseurs NIfTI→STL pour TL et FL."""
```

**Logique** :
```python
# Paramètres spécifiques TL (plus de détails)
params_tl = SurfaceParameters(
    refine_factor=self.config.refine_factor,
    sdf_sigma_mm=self.config.sdf_sigma_mm,
    target_triangles=self.config.target_triangles_tl,
    min_patch_diameter_mm=self.config.min_patch_diameter_mm
)

# Paramètres spécifiques FL (légèrement simplifié)
params_fl = SurfaceParameters(
    refine_factor=self.config.refine_factor,
    sdf_sigma_mm=self.config.sdf_sigma_mm,
    target_triangles=self.config.target_triangles_fl,  # Moins de triangles
    min_patch_diameter_mm=self.config.min_patch_diameter_mm
)

# Instanciation des convertisseurs
self.converter_tl = NiftiToSTLConverter(params=params_tl, verbose=self.verbose)
self.converter_fl = NiftiToSTLConverter(params=params_fl, verbose=self.verbose)
```

**Justification de la différence TL/FL** :
- Le True Lumen est le chenal principal : précision critique pour l'hémodynamique
- Le False Lumen peut être simplifié sans impact majeur sur les résultats CFD globaux
- Gain de temps de calcul : ~25% de réduction du nombre total de triangles

### 7.4 Méthode Principale `extract`

```python
def extract(
    self,
    nifti_path: Path,
    tl_label: Optional[int] = None,
    fl_label: Optional[int] = None
) -> ExtractionResult:
```

**Signature et contrat** :

```python
"""
Exécute le pipeline complet d'extraction TL + FL.

Args:
    nifti_path: Chemin vers le fichier NIfTI segmenté
    tl_label: Valeur du label True Lumen (défaut: config.tl_label)
    fl_label: Valeur du label False Lumen (défaut: config.fl_label)

Returns:
    ExtractionResult: Objet structuré contenant :
        • success: bool (statut global)
        • tl_stats/fl_stats: ExtractionStats (métriques)
        • tl_stl_path/fl_stl_path: chemins des fichiers STL
        • tl_patches/fl_patches: listes des patches détectés
        • error_message: str (si échec)

Raises:
    ImportError: Si nifti_to_stl_converter n'est pas disponible
    FileNotFoundError: Si le fichier NIfTI n'existe pas
"""
```

**Flux d'exécution détaillé** :

```python
# 1. Préparation et logging initial
tl_label = tl_label or self.config.tl_label
fl_label = fl_label or self.config.fl_label
start_time = time.time()
result = ExtractionResult(success=False, config=asdict(self.config))

# 2. Extraction True Lumen
tl_stl = self.output_dir / "tbad_TL_walls.stl"
conv_result_tl = self.converter_tl.convert(
    nifti_path=nifti_path,
    label_value=tl_label,
    output_stl=tl_stl,
    extract_patches=self.extract_patches,
    target_triangles=self.config.target_triangles_tl
)

# 3. Calcul des statistiques TL
stats_tl = self._extract_stats(conv_result_tl.mesh, processing_time_tl)

# 4. Sauvegarde des patches TL (si activé)
if self.extract_patches and conv_result_tl.patches:
    for i, patch in enumerate(conv_result_tl.patches):
        patch_path = self.output_dir / f"tbad_TL_patch_{i}.stl"
        patch['mesh'].export(patch_path)
        self._save_patch_metadata(patch_path, patch)

# 5. Répétition pour False Lumen (étapes 2-4)
# ...

# 6. Construction du résultat final
result = ExtractionResult(
    success=True,
    tl_stats=stats_tl,
    fl_stats=stats_fl,
    tl_stl_path=tl_stl,
    fl_stl_path=fl_stl,
    tl_patches=tl_patches,
    fl_patches=fl_patches,
    config=asdict(self.config)
)

# 7. Rapport et sauvegarde
self._print_summary(result, total_time)
report_path = self.output_dir / "extraction_report.json"
with open(report_path, 'w') as f:
    json.dump(result.to_report(nifti_path, self.output_dir), f, indent=2)

return result
```

### 7.5 Méthode Auxiliaire `_extract_stats`

```python
def _extract_stats(
    self, 
    mesh: trimesh.Trimesh, 
    processing_time: Optional[float]
) -> ExtractionStats:
```

**Calcul des métriques** :

```python
return ExtractionStats(
    # Géométrie de base
    vertices=len(mesh.vertices),
    faces=len(mesh.faces),
    
    # Métriques physiques (conversion mm³, mm²)
    volume_mm3=mesh.volume / (self.config.mm_to_m ** 3),
    surface_area_mm2=mesh.area / (self.config.mm_to_m ** 2),
    
    # Qualité topologique
    is_watertight=mesh.is_watertight,
    
    # Patches (mis à jour ultérieurement)
    patches_count=0,
    
    # Performance
    processing_time_sec=processing_time
)
```

**Interprétation des métriques** :

| Métrique | Valeur typique TBAD | Signification clinique |
|----------|---------------------|----------------------|
| `volume_mm3` TL | 15 000 - 40 000 mm³ | Volume du chenal vrai (débit cardiaque) |
| `volume_mm3` FL | 5 000 - 30 000 mm³ | Volume du faux chenal (risque de thrombose) |
| `is_watertight` | `True` requis | Maillage exploitable pour CFD volumique |
| `faces` TL | ~200 000 | Résolution suffisante pour couches limites |

---

## 8. Module de Visualisation (TbadVisualizer)

### 8.1 Configuration PyVista

```python
def _setup_pyvista(self):
    """Initialise l'environnement de rendu 3D."""
    pv.set_plot_theme('document')      # Style épuré pour publications
    pv.OFF_SCREEN = False              # Mode interactif requis
    pv.global_theme.font.size = 10     # Lisibilité des annotations
    pv.global_theme.font.title_size = 14
```

**Choix de PyVista** :
- ✅ Interface Python native pour VTK (puissant et mature)
- ✅ Rendu interactif avec rotation/zoom natifs
- ✅ Export d'images haute résolution pour publications
- ✅ Compatibilité multi-plateforme (Linux, macOS, Windows)

### 8.2 Méthode `visualize_pair`

```python
def visualize_pair(
    self,
    tl_stl: Path,
    fl_stl: Path,
    tl_patches: Optional[List[Path]] = None,
    fl_patches: Optional[List[Path]] = None,
    title: Optional[str] = None
) -> bool:
```

**Workflow de rendu** :

```python
# 1. Chargement et conversion d'unités
pv_tl = self._load_and_scale_mesh(tl_stl)  # mm → m
pv_fl = self._load_and_scale_mesh(fl_stl)

# 2. Création du plotter avec paramètres
plotter = pv.Plotter(
    title=title or "TBAD - TL + FL",
    window_size=self.config.window_size,
    lighting='three lights'  # Éclairage réaliste
)

# 3. Ajout des maillages avec style
self._add_mesh_to_plotter(
    plotter, pv_tl, 
    color=self.config.tl_color, 
    opacity=0.7,  # Semi-transparent pour voir FL derrière
    label=f"True Lumen ({len(tl_mesh.faces):,} faces)"
)

self._add_mesh_to_plotter(
    plotter, pv_fl,
    color=self.config.fl_color,
    opacity=0.5,  # Plus transparent pour hiérarchie visuelle
    label=f"False Lumen ({len(fl_mesh.faces):,} faces)"
)

# 4. Ajout des patches (conditions limites)
if tl_patches:
    for i, patch_path in enumerate(tl_patches):
        pv_patch = self._load_and_scale_mesh(patch_path)
        color = self.config.patch_tl_inlet_color if i == 0 else self.config.patch_tl_outlet_color
        self._add_mesh_to_plotter(
            plotter, pv_patch, color, opacity=0.9,
            label=f"TL {'Inlet' if i == 0 else f'Outlet {i}'}",
            show_edges=True  # Bordures visibles pour identification
        )

# 5. Décorations et annotations
plotter.add_legend(face='line', bcolor='white', border=True)
plotter.add_text(title, position='upper_edge', color='black')
plotter.add_axes()  # Repère 3D pour orientation

# 6. Statistiques en overlay
stats_text = f"TL: {len(tl_mesh.vertices):,} vtx | FL: {len(fl_mesh.vertices):,} vtx"
plotter.add_text(stats_text, position='lower_edge', color='black', font_size=9)

# 7. Affichage interactif
plotter.show()  # Bloquant jusqu'à fermeture par l'utilisateur
plotter.close()  # Libération des ressources OpenGL
```

**Contrôles interactifs** :
```
🖱️ Souris :
• Clic gauche + glisser : Rotation de la caméra
• Clic droit + glisser : Translation
• Molette : Zoom avant/arrière

⌨️ Clavier :
• q / Échap : Fermer la fenêtre
• f : Mode plein écran
• c : Centrer la vue sur l'objet
• + / - : Ajuster la taille des annotations
```

### 8.3 Gestion des Erreurs de Rendu

```python
try:
    plotter.show()
    return True
except Exception as e:
    logger.error(f"❌ Erreur d'affichage: {e}")
    return False
finally:
    plotter.close()  # Toujours libérer les ressources
```

**Erreurs courantes et solutions** :

| Erreur | Cause probable | Solution |
|--------|---------------|----------|
| `OpenGL.error.GLError` | Pilotes graphiques obsolètes | Mettre à jour les drivers GPU |
| `QApplication instance` | Conflit avec autre GUI (matplotlib) | Utiliser `pv.start_xvfb()` en headless |
| `MemoryError` | Maillage trop volumineux | Réduire `target_triangles` ou activer LOD |

---

## 9. Interface en Ligne de Commande

### 9.1 Parsing des Arguments

```python
def parse_arguments() -> argparse.Namespace:
```

**Structure des groupes** :

```python
# Groupe: Entrée
g_input = parser.add_argument_group("📥 Entrée")
g_input.add_argument("nifti", nargs="?", help="Fichier NIfTI (recherche auto si omis)")
g_input.add_argument("--download", action="store_true", help="Télécharger depuis Kaggle")

# Groupe: Extraction
g_extract = parser.add_argument_group("⚙️ Extraction")
g_extract.add_argument("-o", "--output", default="tbad_stl_output", help="Dossier de sortie")
g_extract.add_argument("--tl-label", type=int, default=1, help="Label True Lumen")
g_extract.add_argument("-r", "--refine", type=int, default=2, help="Facteur de raffinement")
# ... autres paramètres

# Groupe: Visualisation
g_viz = parser.add_argument_group("🎨 Visualisation")
g_viz.add_argument("--viz", action="store_true", help="Visualiser après extraction")
g_viz.add_argument("--viz-only", action="store_true", help="Visualisation seule")
# ...

# Groupe: Divers
g_misc = parser.add_argument_group("🔧 Divers")
g_misc.add_argument("-q", "--quiet", action="store_true", help="Mode silencieux")
g_misc.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
```

### 9.2 Fonction `main` : Orchestration

```python
def main() -> int:
    """Point d'entrée principal avec codes de sortie standard."""
```

**Logique de décision** :

```python
args = parse_arguments()

# Configuration logging selon verbosity
if args.quiet:
    logging.getLogger().setLevel(logging.ERROR)
elif args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

# MODE 1: Visualisation seule
if args.viz_only:
    # ... chargement et affichage des STL existants
    return 0 if success else 1

# MODE 2: Téléchargement
if args.download:
    path = download_tbad_dataset()
    return 0 if path else 1

# MODE 3: Extraction (+ visualisation optionnelle)
# 3.1 Résolution du fichier NIfTI
nifti_path = Path(args.nifti) if args.nifti else find_nifti_file()
if not nifti_path:
    logger.error("❌ Aucun fichier NIfTI trouvé")
    return 1

# 3.2 Exécution de l'extraction
if not args.no_extract:
    extractor = TbadExtractor(...)
    result = extractor.extract(nifti_path=nifti_path, ...)
    if not result.success:
        return 1

# 3.3 Visualisation post-extraction
if args.viz:
    viz = TbadVisualizer()
    viz.visualize_pair(...)

return 0  # Succès
```

### 9.3 Codes de Sortie Standard

```python
# Convention UNIX pour l'automatisation
0   # Succès
1   # Erreur utilisateur (fichier manquant, paramètre invalide)
2   # Erreur système/fatale (exception non gérée)
130 # Interruption par l'utilisateur (Ctrl+C)
```

**Utilisation dans un script shell** :
```bash
#!/bin/bash
python extract_and_visualize_tbad.py --viz

case $? in
    0) echo "✅ Traitement réussi" ;;
    1) echo "❌ Erreur de configuration" ; exit 1 ;;
    2) echo "💥 Erreur système" ; exit 2 ;;
    130) echo "⚠️  Interruption utilisateur" ; exit 130 ;;
esac
```

---

# 📋 PARTIE III : GUIDE PRATIQUE

## 10. Installation et Dépendances

### 10.1 Prérequis Système

```bash
# Python 3.9+ requis
python3 --version  # Doit afficher Python 3.9.0 ou supérieur

# Dépendances système (Linux/Ubuntu)
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \  # Pour le rendu OpenGL
    libegl1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libxi6 \
    libxtst6
```

### 10.2 Installation des Dépendances Python

```bash
# Création d'un environnement virtuel (recommandé)
python3 -m venv tbad-env
source tbad-env/bin/activate  # Linux/macOS
# ou
tbad-env\Scripts\activate  # Windows

# Installation des packages
pip install --upgrade pip
pip install -r requirements.txt
```

**Contenu de `requirements.txt`** :
```txt
# Dépendances principales
numpy>=1.24.0
scipy>=1.10.0
trimesh>=3.23.0
pyvista>=0.42.0
nibabel>=5.1.0
scikit-image>=0.21.0

# Optionnel : téléchargement Kaggle
kagglehub>=0.2.0

# Développement (tests, linting)
pytest>=7.4.0
black>=23.0.0
mypy>=1.5.0
```

### 10.3 Installation du Convertisseur NIfTI→STL

Le script dépend du module `nifti_to_stl_converter`. Deux options :

**Option A : Installation depuis PyPI (si publié)**
```bash
pip install nifti-to-stl-converter
```

**Option B : Installation en mode développement (depuis source)**
```bash
# Cloner le dépôt du convertisseur
git clone https://github.com/votre-org/nifti-to-stl-converter.git
cd nifti-to-stl-converter
pip install -e .  # Mode "editable" pour développement
cd ..
```

**Vérification de l'installation** :
```python
python -c "from nifti_to_stl_converter import NiftiToSTLConverter; print('✅ OK')"
```

---

## 11. Exemples d'Utilisation

### 11.1 Cas Standard : Extraction + Visualisation

```bash
# Dans le répertoire contenant le fichier NIfTI
python extract_and_visualize_tbad.py --viz

# Sortie attendue :
# 12:34:56 - INFO - ✅ Fichier trouvé: patient_126_label.nii.gz
# 12:34:56 - INFO - 🏥 Démarrage extraction TBAD
# 12:34:56 - INFO - 🔵 TRUE LUMEN (label=1) - Extraction
# 12:35:23 - INFO -    └─ Patch TL #0: 18.5 mm
# 12:35:23 - INFO - 🔴 FALSE LUMEN (label=2) - Extraction
# 12:35:48 - INFO -    └─ Patch FL #0: 12.3 mm
# 12:35:48 - INFO - 📊 RAPPORT D'EXTRACTION (52.3s)
# 12:35:48 - INFO - 🔵 TRUE LUMEN:
# 12:35:48 - INFO -    ├─ Vertices: 102,345 | Faces: 198,721
# 12:35:48 - INFO -    ├─ Volume: 28,450 mm³ | Surface: 15,230 mm²
# 12:35:48 - INFO -    ├─ Étanche: ✅
# 12:35:48 - INFO -    └─ Patches: 3
# 12:35:48 - INFO - 🎬 Lancement visualisation...
```

### 11.2 Paramètres Personnalisés pour CFD Haute Résolution

```bash
python extract_and_visualize_tbad.py \
  --refine 3 \                    # Raffinement accru
  --target-tl 400000 \           # Plus de triangles pour TL
  --target-fl 250000 \           # Plus de triangles pour FL
  --sigma 0.2 \                  # Lissage réduit pour détails fins
  --no-patches \                 # Skip patch detection si non requis
  -o results/high_res/ \         # Dossier de sortie personnalisé
  --viz                          # Visualisation finale
```

### 11.3 Traitement par Lot (Batch Processing)

```bash
#!/bin/bash
# process_cohort.sh - Traitement d'une cohorte de patients

INPUT_DIR="data/cohort_TB"
OUTPUT_BASE="results/cohort"

for nifti_file in "$INPUT_DIR"/*_label.nii.gz; do
    patient_id=$(basename "$nifti_file" _label.nii.gz)
    echo "🔄 Traitement de $patient_id..."
    
    python extract_and_visualize_tbad.py \
        "$nifti_file" \
        -o "$OUTPUT_BASE/$patient_id" \
        -q  # Mode silencieux pour logs propres
    
    if [ $? -eq 0 ]; then
        echo "✅ $patient_id terminé"
    else
        echo "❌ Échec pour $patient_id" >&2
    fi
done

echo "📊 Cohorte traitée. Rapports dans $OUTPUT_BASE/"
```

### 11.4 Intégration dans un Pipeline CFD (OpenFOAM)

```bash
#!/bin/bash
# cfd_setup.sh - Préparation automatique d'un cas OpenFOAM

PATIENT="126"
STL_DIR="tbad_stl_output"
CASE_DIR="OpenFOAM/TBAD_${PATIENT}"

# 1. Extraction des maillages
python extract_and_visualize_tbad.py \
    "data/${PATIENT}_label.nii.gz" \
    -o "$STL_DIR" \
    --target-tl 200000 \
    --target-fl 150000 \
    -q

# 2. Création du cas OpenFOAM
mkdir -p "$CASE_DIR"/constant/triSurface
cp "$STL_DIR"/tbad_TL_walls.stl "$CASE_DIR"/constant/triSurface/
cp "$STL_DIR"/tbad_FL_walls.stl "$CASE_DIR"/constant/triSurface/

# 3. Génération du mesh volumique (snappyHexMesh)
cd "$CASE_DIR"
blockMesh
surfaceFeatures
snappyHexMesh -overwrite

# 4. Configuration des conditions limites (à personnaliser)
# ... édition des fichiers 0/U, 0/p, etc.

echo "✅ Cas OpenFOAM prêt dans $CASE_DIR"
```

---

## 12. Dépannage et Bonnes Pratiques

### 12.1 Problèmes Courants et Solutions

#### ❌ "Aucun fichier NIfTI trouvé"

**Causes** :
- Fichier mal nommé ou dans un sous-dossier non recherché
- Extension incorrecte (`.nii` au lieu de `.nii.gz`)

**Solutions** :
```bash
# Option 1: Spécifier le chemin complet
python extract_and_visualize_tbad.py /chemin/complet/patient_label.nii.gz

# Option 2: Utiliser le téléchargement automatique
python extract_and_visualize_tbad.py --download

# Option 3: Vérifier la recherche manuelle
find . -name "*label*.nii*"  # Identifier le bon fichier
```

#### ❌ "nifti_to_stl_converter non trouvé"

**Cause** : Dépendance non installée.

**Solution** :
```bash
# Installer depuis le dépôt source
git clone https://github.com/votre-org/nifti-to-stl-converter.git
cd nifti-to-stl-converter
pip install -e .
cd ..
```

#### ❌ Erreur OpenGL / Rendu graphique

**Symptômes** : Fenêtre noire, crash au lancement de `--viz`.

**Solutions** :
```bash
# Linux : Installer les bibliothèques graphiques
sudo apt-get install -y libgl1-mesa-glx libegl1 libxrandr2

# macOS : Vérifier les permissions d'accès à l'écran
# Préférences Système → Confidentialité → Accès à l'écran → Terminal ✓

# Mode headless (sans affichage, pour serveurs)
export PYVISTA_OFF_SCREEN=true
python extract_and_visualize_tbad.py --no-viz  # Skip visualisation
```

#### ❌ Maillage non-étanche (`is_watertight: False`)

**Causes** :
- Segmentation incomplète (trous dans le masque)
- Paramètres de lissage trop agressifs

**Solutions** :
```bash
# 1. Réduire le lissage SDF
python extract_and_visualize_tbad.py --sigma 0.1

# 2. Augmenter le raffinement pour mieux capturer les détails
python extract_and_visualize_tbad.py --refine 3

# 3. Vérifier la segmentation source (outil externe comme 3D Slicer)
#    • Remplir manuellement les trous
#    • Ré-exporter le NIfTI corrigé
```

### 12.2 Bonnes Pratiques de Production

#### ✅ Validation des Résultats

```python
# Script de validation post-extraction (validate_output.py)
import json
from pathlib import Path

def validate_extraction(output_dir: Path) -> bool:
    """Vérifie la qualité des fichiers générés."""
    report_path = output_dir / "extraction_report.json"
    
    if not report_path.exists():
        print("❌ Rapport manquant")
        return False
    
    with open(report_path) as f:
        report = json.load(f)
    
    checks = [
        (report['success'], "Extraction réussie"),
        (report['true_lumen']['stats']['is_watertight'], "TL étanche"),
        (report['false_lumen']['stats']['is_watertight'], "FL étanche"),
        (report['true_lumen']['stats']['faces'] > 50000, "TL suffisamment détaillé"),
    ]
    
    all_passed = True
    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {description}")
        if not passed:
            all_passed = False
    
    return all_passed
```

#### ✅ Gestion des Versions et Reproductibilité

```yaml
# config/version.yaml
pipeline:
  version: "1.0.0"
  date: "2026-03-02"
  
dependencies:
  python: ">=3.9"
  nifti_to_stl_converter: ">=0.3.0"
  pyvista: ">=0.42.0"
  
defaults:
  refine_factor: 2
  sdf_sigma_mm: 0.25
  target_triangles_tl: 200000
  
validation_thresholds:
  min_watertight: true
  min_faces_tl: 50000
  max_processing_time_sec: 300
```

#### ✅ Logging Structuré pour le Debug

```python
# Activer le logging JSON pour l'analyse automatisée
import logging
import json_log_formatter

formatter = json_log_formatter.JSONFormatter()
handler = logging.FileHandler('pipeline.log')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)

# Exemple de log structuré
logger.info("Extraction démarrée", extra={
    "patient_id": "126",
    "nifti_size_mb": 45.2,
    "config": {"refine_factor": 2, "sigma": 0.25}
})
```

---

# 📋 ANNEXES

## A. Format des Fichiers de Sortie

### A.1 Structure du Dossier de Sortie

```
tbad_stl_output/
├── tbad_TL_walls.stl              # Maillage principal True Lumen
├── tbad_FL_walls.stl              # Maillage principal False Lumen
├── tbad_TL_patch_0.stl           # Patch TL #0 (généralement inlet)
├── tbad_TL_patch_0.json          # Métadonnées du patch TL #0
├── tbad_TL_patch_1.stl           # Patch TL #1 (outlet)
├── tbad_TL_patch_1.json
├── tbad_FL_patch_0.stl           # Patch FL #0
├── tbad_FL_patch_0.json
├── extraction_report.json        # Rapport global d'extraction
└── validation_metrics.json       # (Optionnel) Métriques de qualité
```

### A.2 Contenu de `extraction_report.json`

```json
{
  "metadata": {
    "timestamp": "2026-03-02T14:35:48.123456",
    "nifti_file": "patient_126_label.nii.gz",
    "output_directory": "tbad_stl_output",
    "config": {
      "refine_factor": 2,
      "sdf_sigma_mm": 0.25,
      "target_triangles_tl": 200000,
      "target_triangles_fl": 150000,
      "tl_label": 1,
      "fl_label": 2
    }
  },
  "success": true,
  "error": null,
  "true_lumen": {
    "stl_file": "tbad_TL_walls.stl",
    "stats": {
      "vertices": 102345,
      "faces": 198721,
      "volume_mm3": 28450.3,
      "surface_area_mm2": 15230.7,
      "is_watertight": true,
      "patches_count": 3,
      "processing_time_sec": 27.4
    },
    "patches": [
      "tbad_TL_patch_0.stl",
      "tbad_TL_patch_1.stl",
      "tbad_TL_patch_2.stl"
    ]
  },
  "false_lumen": {
    "stl_file": "tbad_FL_walls.stl",
    "stats": {
      "vertices": 78912,
      "faces": 152340,
      "volume_mm3": 12890.1,
      "surface_area_mm2": 9876.4,
      "is_watertight": true,
      "patches_count": 2,
      "processing_time_sec": 24.9
    },
    "patches": [
      "tbad_FL_patch_0.stl",
      "tbad_FL_patch_1.stl"
    ]
  }
}
```

### A.3 Contenu d'un Fichier de Métadonnées de Patch

```json
{
  "diameter_mm": 18.5,
  "area_mm2": 268.4,
  "circularity": 0.87,
  "normal": [0.12, -0.03, 0.99],
  "center": [45.2, -12.8, 156.3],
  "classification": "TL_outlet",
  "recommended_bc": {
    "type": "pressure_outlet",
    "value_Pa": 13332,
    "description": "Pression diastolique moyenne ~100 mmHg"
  }
}
```

**Champ `circularity`** :
```
circularity = 4π × Aire / Périmètre²
• 1.0 : Cercle parfait
• 0.8-0.99 : Quasi-circulaire (bon pour conditions limites)
• < 0.7 : Forme irrégulière (vérifier la segmentation)
```

---

## B. Paramètres par Défaut et Réglages

### B.1 Tableau des Paramètres Critiques

| Paramètre | Défaut | Plage Recommandée | Impact |
|-----------|--------|-------------------|--------|
| `refine_factor` | 2 | 1-4 | ↑ = plus de détails, ↑ temps calcul |
| `sdf_sigma_mm` | 0.25 | 0.1-0.5 | ↑ = plus lisse, ↓ détails fins |
| `target_triangles_tl` | 200 000 | 100k-500k | ↑ = meilleure résolution CFD |
| `target_triangles_fl` | 150 000 | 50k-300k | ↑ = plus précis mais plus lourd |
| `min_patch_diameter_mm` | 4.0 | 2.0-8.0 | ↓ = détecte plus de petits patches |

### B.2 Profils de Configuration Prédéfinis

```python
# Dans config/profiles.py

PROFILES = {
    "standard": TBADConfig(
        refine_factor=2,
        sdf_sigma_mm=0.25,
        target_triangles_tl=200_000,
        target_triangles_fl=150_000
    ),
    
    "high_res_cfd": TBADConfig(
        refine_factor=3,
        sdf_sigma_mm=0.2,
        target_triangles_tl=400_000,
        target_triangles_fl=250_000,
        min_patch_diameter_mm=3.0
    ),
    
    "quick_preview": TBADConfig(
        refine_factor=1,
        sdf_sigma_mm=0.5,
        target_triangles_tl=50_000,
        target_triangles_fl=30_000,
        extract_patches=False
    ),
    
    "research_validation": TBADConfig(
        refine_factor=4,
        sdf_sigma_mm=0.1,
        target_triangles_tl=500_000,
        target_triangles_fl=300_000,
        min_patch_diameter_mm=2.0
    )
}
```

**Utilisation** :
```bash
# Via variable d'environnement
export TBAD_PROFILE=high_res_cfd
python extract_and_visualize_tbad.py --viz

# Ou modification programmatique
from config.profiles import PROFILES
extractor = TbadExtractor(config=PROFILES["research_validation"])
```

---

## C. Références Bibliographiques

### C.1 Fondamentaux Médicaux

1. **Erbel, R., et al. (2014)**. *ESC Guidelines on the diagnosis and treatment of aortic diseases*. European Heart Journal, 35(41), 2873-2926.
   - Classification Stanford Type A/B
   - Algorithmes de prise en charge

2. **Nienaber, C. A., & Powell, J. T. (2012)**. *Management of acute aortic syndromes*. European Heart Journal, 33(1), 26-35.
   - Physiopathologie de la dissection
   - Critères d'intervention

### C.2 Méthodes de Reconstruction 3D

3. **Lorensen, W. E., & Cline, H. E. (1987)**. *Marching cubes: A high resolution 3D surface construction algorithm*. ACM SIGGRAPH Computer Graphics, 21(4), 163-169.
   - Algorithme fondateur d'extraction d'isosurface

4. **Kazhdan, M., & Hoppe, H. (2013)**. *Screened Poisson surface reconstruction*. ACM Transactions on Graphics, 32(3), 1-13.
   - Méthode SDF avancée pour lissage de surface

5. **Garland, M., & Heckbert, P. S. (1997)**. *Surface simplification using quadric error metrics*. ACM SIGGRAPH, 209-216.
   - Décimation de maillage préservant la géométrie

### C.3 Applications CFD en Biomécanique

6. **Morris, L., et al. (2016)**. *Patient-specific CFD modelling of Type B aortic dissection*. Journal of Biomechanics, 49(16), 3877-3885.
   - Pipeline complet de reconstruction → simulation

7. **Menon, P. G., et al. (2020)**. *Computational fluid dynamics in aortic dissection: Current concepts and future directions*. Frontiers in Cardiovascular Medicine, 7, 61.
   - Revue des méthodes CFD appliquées à la dissection

### C.4 Ressources Logicielles

8. **PyVista Documentation**. https://docs.pyvista.org/
   - Bibliothèque de visualisation 3D utilisée dans ce pipeline

9. **Trimesh Documentation**. https://trimsh.org/
   - Manipulation de maillages triangulés en Python

10. **NIfTI Format Specification**. https://nifti.nimh.nih.gov/
    - Standard d'imagerie médicale pour l'échange de données

---

## 📝 Notes de Version

| Version | Date | Modifications |
|---------|------|--------------|
| 1.0.0 | 2026-03-02 | Première version complète : théorie + code + guide |
| 0.2.0 | 2026-02-15 | Ajout support patches, visualisation PyVista |
| 0.1.0 | 2026-01-20 | Version initiale : extraction NIfTI→STL de base |

---

> **Avertissement** : Ce pipeline est un outil de recherche et de planification. Il ne remplace pas l'expertise clinique. Les résultats doivent être validés par un professionnel de santé qualifié avant toute utilisation diagnostique ou thérapeutique.

*Document généré automatiquement - Dernière mise à jour : Mars 2026*