https://github.com/prs-eth/point2cad

# Plan détaillé : Génération de géométrie CAD à partir des données imageTBAD

## 0. Contexte et cadrage

Votre projet `foampilot` est actuellement un pipeline **NIfTI → STL → CFD OpenFOAM** : `data_preproc` extrait les maillages (TL, FL, paroi) via SDF + marching cubes, et le reste du dépôt (`CoA_test_foampilot`, `blockmesh.json`, `validation_windkessel.py`) montre que ces STL alimentent un maillage `snappyHexMesh` et une simulation avec conditions de Windkessel.

Passer à une **géométrie CAD (B-rep / NURBS, export STEP/IGES)** a du sens pour :
- l'édition paramétrique (études virtuelles, stenting, sizing d'endoprothèses),
- l'export vers des solveurs FEM/FSI qui exigent du B-rep (ANSYS, SimVascular svFSI),
- l'archivage et l'échange de géométries propres et watertight.

**Spécificités TBAD à gérer** : trois entités (True Lumen, False Lumen, paroi externe) + le **flap intimal** (membrane quasi-épaisseur nulle) + les **tears** (primary entry + re-entries) qui connectent TL et FL. C'est ce qui rend le cas beaucoup plus difficile qu'une aorte saine.

---

## 1. Architecture cible du pipeline

```
imageTBAD/*.nii.gz (image + label)
        │  (1) Segmentation / contrôle qualité
        ▼
Masques TL / FL / paroi
        │  (2) Extraction maillages de référence (déjà dans foampilot)
        ▼
STL de référence (TL, FL, wall)  ← sert de "ground truth" pour validation
        │  (3) Structuration : centerlines + coupes perpendiculaires
        ▼
Contours 2D par station le long de TL et FL
        │  (4) Construction CAD : fitting B-spline + lofting
        ▼
Solides B-rep (TL, FL) → flap, tears, paroi via booléens/offset
        │  (5) Export STEP/IGES/BREP
        ▼
(6) Validation (Hausdorff, watertight, aires de sections) → réinjection foampilot/CFD
```

---

## 2. Recherche détaillée des modules existants (par étape)

### 2.1 Segmentation (optionnelle, labels déjà présents)

| Module | Type | Pertinence TBAD |
|---|---|---|
| **nnU-Net** | Open source (Python) | Validé spécifiquement sur la dissection aortique : segmentation TL / FL / flap intimal sur CT contrasté [[31]], et évalué sur Type B [[33]]. **Meilleur choix** si vous devez resegmenter. |
| **TotalSegmentator** | Open source (Python, Apache 2.0) [[30]] | Très robuste en anatomie normale, mais **insuffisant en pathologique** (aortes disséquées) sans correction manuelle [[29]], [[34]]. À utiliser seulement en pré-segmentation. |
| **3D Slicer** | Open source (GUI + Python) | Excellent pour le contrôle/correction manuelle des labels existants, et export STL. |

→ **Recommandation** : garder les labels `imageTBAD` existants ; ajouter un module de QA visuel ; nnU-Net uniquement si extension de la cohorte.

### 2.2 Extraction et nettoyage de maillages (étape déjà couverte)

- **Vos scripts existants** (`extract_tbad_full.py`, `niftitostlconverter.py`) : SDF adaptatif + marching cubes Lewiner + décimation → c'est votre *ground truth* STL.
- **SimVascular** : segmentation level-set 2D le long des vaisseaux pour obtenir des contours de paroi lisses [[46]].
- **VMTK** : segmentation level-set également disponible (`vmtklevelsetsegmentation`).
- **trimesh / PyMeshLab** : nettoyage, lissage Taubin, séparation de composantes connexes, décimation — indispensables en post-traitement.

### 2.3 Centerlines et sections perpendiculaires (clé de voûte de l'approche CAD)

| Module | Type | Notes |
|---|---|---|
| **VMTK** (`vmtkcenterlines`, `vmtkbranchsections`) | Open source (CLI/Python) | L'outil de référence pour l'extraction de centerlines vasculaires [[10]]. Attention : le site vmtk.org a expiré [[58]], mais le projet vit sur GitHub et via 3D Slicer [[11]]. |
| **SlicerExtension-VMTK** | Open source | Interface graphique idéale pour valider interactivement les centerlines sur vos cas difficiles [[11]], export des centerlines possible [[54]]. |
| **SimVascular** (path extraction + 2D segmentations) | Open source | Workflow complet "centerline → coupes 2D → contours" [[45]], utilisé par des équipes cliniques sur des aortes [[5]]. |
| **CGAL mesh skeletonization** | Open source | Alternative citée pour des maillages imparfaits, plutôt pour squelettisation [[13]]. |
| Travaux récents | Recherche | Extraction de centerlines automatisée sur maillages vasculaires (2026) [[9]] ; mesures de sections aortiques automatisées le long de la centerline [[3]]. |

→ **Recommandation** : VMTK en script (batch) + Slicer/VMTK en validation interactive. Traiter **TL et FL comme deux arbres vasculaux distincts**.

### 2.4 Conversion vers CAD — 4 familles d'approches

#### Approche A — Lofting de sections NURBS le long des centerlines (**recommandée**)

C'est la méthode standard de la communauté cardiovasculaire :

- **SimVascular** : le seul package open source couvrant segmentation → modèle solide → simulation [[66]]. Construit des solides loftés à partir de contours 2D perpendiculaires aux paths [[45]], s'appuie sur **OpenCASCADE** pour le noyau CAD [[48]], et dispose d'une **interface Python scriptable** (`sv` package) pour automatiser tout le pipeline [[61]]. Des équipes l'utilisent déjà sur des dissections aortiques avec ParaView [[84]].
- **OpenCASCADE (OCC)** directement, via **pythonOCC / pyOCCT** (bindings Python) [[47]] : loft par `BRepOffsetAPI_ThruSections` [[49]], booléens, offsets (`MakeThickSolid`/`MakeOffsetShape` pour la paroi), export STEP/IGES/BREP.
- **CadQuery / build123d / FreeCAD (Python)** : surcouches plus ergonomiques au-dessus d'OCC pour scripter lofts et booléens.
- **geomdl (NURBS-Python)** : fitting de courbes/surfaces B-spline par interpolation et moindres carrés [[69]], [[67]] — idéal pour lisser vos contours 2D en courbes B-spline avant loft, et pour interpoler les sections.
- Référence académique : framework CAD-intégré de modélisation vasculaire **solide NURBS** basé templates (ICES Report 17-24) [[12]].

**Points durs TBAD** : sections non convexes (TL collabé en croissant), alignement des seams anti-twisting, raccord des branches (subclavière, etc.), cohérence du nombre de pôles entre sections.

#### Approche B — Quad-remeshing + fitting NURBS par patches (haute fidélité, R&D)

- **Quadriflow** (intégré à Blender) [[86]], **Instant Meshes** (open source) [[87]], **AutoRemesher** [[92]] : remaillage en quads.
- Conversion quads → patches NURBS : fitting moindres carrés par patch avec **geomdl**, puis assemblage B-rep dans OCC. Chez les pros, Rhino (SubD/T-Splines) fait cela de façon quasi-manuelle [[17]], [[19]].
- **Avantage** : fidélité géométrique maximale (bosses, calcifications). **Inconvénient** : l'assemblage B-rep watertight à partir de patches est *très* délicat en open source (continuité, trimming, T-junctions).

#### Approche C — Mesh→B-rep direct / outils dédiés

- **Analysis Situs** (open source) : conversion directe mesh→B-rep (facettes → faces CAD) [[43]] — produit un B-rep lourd, peu "éditable", mais watertight.
- **Mesh-to-BRep (Nature Architects)** [[38]], [[40]] et **CadExchanger** [[44]] : solutions récentes/commerciales du problème polygonal→B-rep.
- Utile en solution de secours pour la paroi externe si l'approche A perd trop de détails.

#### Approche D — Deep learning B-rep (veille, non production)

- **Point2CAD** [[28]], **Point2Brep** [[24]], **PartCAD** [[27]] ; surveys récents [[20]], [[21]] ; liste de référence awesome-brep-reconstruction [[26]].
- **Verdict inchangé** : ces méthodes sont entraînées sur des datasets mécaniques (ABC, Fusion 360, DeepCAD…) [[26]] et décomposent les formes en primitives + arêtes vives. **Inadaptées aux formes organiques** de l'aorte ; à garder en veille (progression rapide 2024-2026 [[23]]).

#### Approche E — Commercial (référence/benchmark uniquement)

- **Materialise Mimics / 3-matic** (module centerline aorte [[7]]), **Geomagic Design X** (mesh→NURBS), **Simpleware**, **Rhino**. Robustes mais payants, peu automatisables, hors philosophie open source de foampilot. À utiliser seulement comme *benchmark* de validation si budget disponible.

### 2.5 Tableau récapitulatif des choix

| Étape | Choix n°1 | Choix n°2 (secours) |
|---|---|---|
| Segmentation | Labels existants + QA | nnU-Net [[33]] |
| Maillage de référence | `niftitostlconverter.py` (foampilot) | 3D Slicer |
| Centerlines/sections | VMTK [[10]] (+ Slicer pour QA [[11]]) | SimVascular paths [[45]] |
| Fitting courbes 2D | geomdl [[69]] | OCC `GeomAPI_Interpolate` |
| Lofting / solides | pyOCCT ou CadQuery (OCC) [[47]], [[49]] | SimVascular sv Python [[61]] |
| Paroi externe | Offset OCC du solide lumens | Loft contours externes / quad+NURBS (B) |
| Export | STEP AP242 via OCC | IGES/BREP |
| Validation | trimesh/PyMeshLab (Hausdorff) | Benchmark Mimics |

---

## 3. Plan de mise en œuvre (work packages)

### WP1 — Spécifications & données (≈ 1 sem.)
- Définir les usages du CAD (CFD externe, FSI, planning), tolérances (écart max vs STL de référence : viser **< 0,5 mm** en Hausdorff moyen), formats (STEP AP242 + BREP).
- Inventaire des cas `imageTBAD` (patient 58 etc.), vérification labels TL=1/FL=2.
- **Livrable** : doc de spécification + jeu de test (3 cas : flap complet, FL partiellement thrombosé, TL collabé).

### WP2 — Maillages de référence & QA (≈ 1 sem.)
- Réutiliser `extract_tbad_full.py` / `niftitostlconverter.py` ; ajouter nettoyage trimesh/PyMeshLab (composantes connexes, lissage Taubin léger).
- Script de QA : watertightness, volumes, aires de sections vs mesures cliniques [[3]].
- **Livrable** : STL de référence versionnés par patient.

### WP3 — Structuration vasculaire (≈ 2 sem.)
- Centerlines VMTK séparées pour TL et FL ; détection des branches ; identification des **tears** (points de connexion TL↔FL) par analyse de distance entre surfaces.
- Coupes perpendiculaires tous les ~2-5 mm ; extraction des contours (trimesh slice ou `vmtkbranchsections`) ; stockage (centerline, matrice de repère local, contours).
- **Livrable** : base de sections par patient + visualisation de contrôle.

### WP4 — Construction CAD par lofting (≈ 3-4 sem., cœur du projet)
1. Fitting B-spline de chaque contour (geomdl interpolation [[67]]), degré 3, **même nombre de pôles** par famille de sections, seam aligné (anti-twist).
2. Loft `BRepOffsetAPI_ThruSections` [[49]] → solide TL, solide FL (solid=True).
3. **Flap intimal** : surface partagée = intersection/zone de contact des lofts TL/FL ; modélisation soit en shell épaisseur nulle (CFD), soit en solide mince par épaississement (FSI).
4. **Tears** : booléens — fusion TL∪FL puis soustraction du flap tronqué aux niveaux des tears pour créer les orifices de communication.
5. **Paroi externe** : offset (`MakeOffsetShape`) du solide sanguin ou loft des contours externes issus du masque paroi.
6. Export STEP/BREP ; le tout scripté en Python (pyOCCT/CadQuery) dans un nouveau module `examples/coa/cad_reconstruction/` miroir de `data_preproc`.
- **Livrable** : STEP watertight TL+FL+flap+paroi par patient, scripté de bout en bout.

### WP5 — Voie alternative haute fidélité (R&D, ≈ 3 sem., optionnel)
- Quadriflow/Instant Meshes sur la paroi externe [[86]], [[87]] → segmentation en patches → fitting geomdl moindres carrés → assemblage OCC.
- Comparaison Hausdorff vs approche A ; décision go/no-go pour remplacer le loft de paroi.
- Veille : Analysis Situs [[43]], Point2Brep [[24]].

### WP6 — Validation & intégration foampilot (≈ 1-2 sem.)
- Métriques : distance Hausdorff CAD↔STL, aires de sections le long de la centerline, volumes, watertightness.
- **Validation fonctionnelle** : remailler le STEP (snappyHexMesh déjà présent dans foampilot) et comparer avec `validation_windkessel.py` les courbes pression/débit actuelles — la géométrie CAD doit reproduire les résultats CFD à quelques % près.
- Intégration repo : `launch.sh`, config dataclass comme dans `data_preproc`, rapport auto via `generate_report.py`.

**Charge totale estimée** : ≈ 8-11 semaines homme, dont le WP4 est le chemin critique.

---

## 4. Risques principaux et parades

| Risque | Parade |
|---|---|
| Sections TBAD non convexes / TL en croissant → loft qui s'auto-intersecte | Fitting B-spline avec paramétrisation par arc length, sections adaptatives (plus denses aux zones courbées), vérification d'auto-intersection OCC |
| Twist entre sections | Alignement des seams via point de référence anatomique (côté flap) + optimisation de rotation minimale |
| Flap d'épaisseur nulle → booléens fragiles | Modéliser d'abord le volume sanguin TL∪FL comme **un seul solide**, le flap devenant une face interne ; éviter les offsets sur le flap |
| VMTK en maintenance incertaine (site expiré [[58]]) | Épingler la version via SlicerExtension-VMTK [[11]] ; wrapper isolé pour pouvoir migrer vers SimVascular paths |
| Perte de fidélité vs STL | WP5 en secours + tolérance Hausdorff contractualisée dans WP1 |

---

## 5. Recommandation finale

**Voie principale** : pipeline "VMTK (centerlines/sections) → geomdl (fitting B-spline) → OpenCASCADE via pyOCCT/CadQuery (lofts, booléens, STEP)", branché en aval de votre `data_preproc` existant. C'est la méthode éprouvée par la communauté (SimVascular fait exactement cela [[45]], [[48]], et des pipelines dissection open source existent [[84]], [[77]]), compatible avec votre philosophie open source et votre aval OpenFOAM.

**À écarter pour la production** : Point2CAD et les méthodes DL B-rep (formes mécaniques uniquement) [[26]], ainsi que le mesh→B-rep direct (B-rep non éditable) [[43]] — à conserver en veille.

Si vous voulez, je peux ensuite vous rédiger le squelette de code du WP4 (classes `CenterlineExtractor`, `SectionFitter`, `LoftBuilder` avec pyOCCT) prêt à être intégré dans `examples/coa/cad_reconstruction/`.