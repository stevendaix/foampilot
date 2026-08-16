# État des lieux — VoxCity / OpenFOAM Neighborhood Demo

## 1. Objectif global
Construire un exemple complet et fonctionnel d'urban CFD dans `examples/building_geo/neighborhood_demo/` en utilisant VoxCity, Gmsh, OpenFOAM et foampilot, sans fallback synthétique obligatoire.

## 2. Fichiers du dossier — rôle et périmètre

| Fichier / dossier | Rôle | À garder dans foampilot ? |
|-------------------|------|---------------------------|
| `neighborhood_demo/config.json` | Configuration AOI, solver, maillage | **Oui** — exemple de config utilisateur |
| `neighborhood_demo/generate.py` | Chargement VoxCity + pipeline complet | **Oui** — entry point principal |
| `neighborhood_demo/generate_synthetic.py` | Fallback synthétique (legacy) | **Non** — à supprimer après migration |
| `neighborhood_demo/postprocess.py` | Post-traitement standard | **Non** — remplacé par `voxcity_dedicated_postprocess.py` |
| `neighborhood_demo/run_full_voxcity_pipeline.py` | Pipeline unifié VoxCity→OpenFOAM→post | **Oui** — script principal |
| `neighborhood_demo/voxcity_dedicated_postprocess.py` | Post-traitement VoxCity-aware | **Oui** — module réutilisable |
| `neighborhood_demo/verify_geometry.py` | Vérification visuelle raw vs processed footprints | **Oui** — outil de vérification |
| `neighborhood_demo/rapport_calcul.md` | Rapport détaillé du calcul | **Non** — documentation de l'exemple |
| `neighborhood_demo/output/voxcity.h5` | Données VoxCity AOI Paris 15e | **Non** — données trop volumineuses, à télécharger |
| `neighborhood_demo/README.md` | Documentation utilisateur | **Oui** — à mettre à jour |
| `voxcity_export_work/src/vector_builder.py` | Construction Gmsh + export OpenFOAM | **Oui** — module core |
| `voxcity_export_work/src/voxcity_vector_example.py` | Exemple standalone | **Oui** — exemple minimal |
| `voxcity_export_work/src/voxcity_cached_example.py` | Exemple avec cache | **Oui** — exemple avec cache |
| `foampilot/src/foampilot/urban/readers/voxcity_reader.py` | Reader VoxCity→UrbanModel | **Oui** — intégré à foampilot |
| `foampilot/src/foampilot/mesh/direct_openfoam_exporter.py` | Export direct OpenFOAM | **Oui** — déjà intégré |

## 3. Corrections apportées

### 3.1 VoxCity reader (`voxcity_reader.py`)
- Ajout de `import numpy as np` manquant.
- Correction de `_extract_buildings()` pour reprojeter les footprints WGS84 vers EPSG:32631 avant de tester l'aire, sinon tous les bâtiments étaient rejetés.
- Gestion des `MultiPolygon` et garde-fous sur `ground_z` / `height`.

### 3.2 neighborhood_demo (`generate.py`)
- Suppression du fallback synthétique obligatoire ; l'exemple utilise maintenant VoxCity directement.
- Ajout de `--voxcity-h5` pour charger un HDF5 local sans re-téléchargement.
- Ajout de `--use-cache` pour réutiliser le cache VoxCity quand il existe.
- Chargement HDF5 via `voxcity.io.load_voxcity()` avec reprojection métrique et filtrage par surface projetée.
- Pipeline solver / BCs aligné sur `generate_wind_cases.py` avec profil log-wind `codedFixedValue`.

### 3.3 vector_builder.py
**Diagnostic critique du code initial :**
- Méthode `build()` dupliquée (lignes 63 et 97) — code mort source de confusion.
- Bug dans `_preprocess_geometry()` : après gap-filling, le tableau `used` conservait ses anciens indices alors que `valid` était remplacé par `filled` → `IndexError` ou comportement incorrect quand `--fill-gaps` est actif.
- Stratégie `cut()` fondamentalement fragile : `cut()` avec `removeObject=True, removeTool=True` sur des bâtiments se chevauchant produit des résultats incohérents (`fluid_tag=None`). Le parsing du résultat prenait arbitrairement le premier volume dim==3 trouvé, qui pouvait être un fragment de bâtiment.
- Fallback `fragment()` également fragile : il fragmente tout mais ne supprime rien, puis le code prenait arbitrairement le premier volume comme fluide.
- `_remove_building_volumes()` utilisait des tags périmés après les opérations Boolean.

**Corrections implémentées :**
- Suppression de la méthode `build()` dupliquée.
- Correction du bug `_preprocess_geometry()` : la suppression des chevauchements utilise maintenant un tableau `used` frais dimensionné pour la liste courante.
- **Stratégie Boolean** : `cut()` séquentiel reste la stratégie principale (plus robuste que `cut()` multiple). `fragment()` est utilisé en fallback si `cut()` échoue. Les bâtiments sont identifiés après fragmentation par triple test (bbox → Z → `Point.within()`).
- Ajout de `_identify_building_volumes()` : identifie les volumes bâtiments par triple test (bbox rapide → check Z → `Point.within()` sur la footprint).
- Suppression de `_remove_non_fluid_volumes()` et `_remove_building_volumes()` remplacée par une suppression ciblée des bâtiments identifiés.
- **Découverte clé** : `Mesh.Algorithm3D = 5` (Frontal-Delaunay) ne sait pas mailler des volumes avec des faces internes (trous). Passage à `Mesh.Algorithm3D = 4` (Delaunay) qui maille correctement les domaines avec bâtiments.
- Mise à jour de `self.fluid_tag` après suppression des bâtiments pour suivre le volume restant.

### 3.4 Pipeline complet (`run_full_voxcity_pipeline.py`)
**Nouveau script unifié** qui orchestre toute la chaîne :
1. Chargement VoxCity HDF5 (pas de téléchargement EE)
2. Construction géométrie Gmsh (cut séquentiel + Delaunay)
3. Export DirectOpenFOAM polyMesh
4. **Check qualité maillage** via `OpenFOAMQualityAnalyzer` + `checkMesh` automatique
5. Configuration solveur OpenFOAM avec BCs log-wind
6. Exécution simulation (option `--skip-run`)
7. Post-traitement dédié VoxCity
8. **Génération rapport PDF** via `generate_report.py`

### 3.5 Gestion individuelle des footprints (`_preprocess_geometry`)
**Changement majeur** : chaque bâtiment est maintenant traité individuellement au lieu d'être fusionné en un seul footprint géant.
- Suppression de la fusion de tous les bâtiments en un seul `merged_all`
- Chaque bâtiment valide est nettoyé indépendamment :
  - Filtrage : area < 1.0 m², height < 0.5 m
  - Correction géométrie invalide via `buffer(0)`
  - Simplification via `simplify(tolerance=mesh_size*0.5)`
- L'`UrbanModel` conserve tous les bâtiments individuels après nettoyage
- Avantage : chaque bâtiment peut être extrudé individuellement avec sa propre hauteur
- Nombre de bâtiments maintenus : tous les bâtiments valides (ex: ~30 pour l'AOI Paris 15e) au lieu de 1 merged

### 3.6 Bâtiments polygonaux (`_extrude_polygon`)
**Capacité conservée** : extrusion des footprints réels des bâtiments au lieu de boîtes axis-aligned.
- `_create_building_volume()` : priorité à `_extrude_polygon()`, fallback sur bbox si échec
- `_extrude_polygon()` : crée un volume Gmsh à partir du Polygon/MultiPolygon Shapely
- Résultat : formes de bâtiments conformes aux données VoxCity, chacun extrudé individuellement

**Avertissement** : les bâtiments polygonaux augmentent la complexité du maillage et peuvent causer des crashs solveur (SIGFPE) si la qualité est insuffisante. Pour ce cas, l'option bbox reste recommandée pour la robustesse.

### 3.7 Rapport de calcul détaillé (`generate_report.py`)
**Script Python** utilisant `foampilot.report.latex_pdf.LatexDocument` pour produire un PDF professionnel avec :
- Cartographie matplotlib + mention carte folium interactive
- Pipeline et données d'entrée
- Justification des hypothèses (loi log-wind, modèle kEpsilon, marges, fusion bâtiments)
- Configuration solveur et conditions aux limites
- Résultats : convergence, statistiques, distribution confort NEN
- Visualisations : slice vitesse, Cp bâtiments, wireframe maillage
- Export des données (CSV, JSON)

### 3.5 Post-traitement dédié (`voxcity_dedicated_postprocess.py`)
**Nouveau post-processeur VoxCity-aware** :
- Extraction métadonnées VoxCity depuis HDF5 (bâtiments, hauteurs, IDs)
- Cartes de confort éolien (NEN) à hauteur piétonne
- Intensité de turbulence à hauteur piétonne
- Distribution de confort (calm / comfortable / moderate / uncomfortable / dangerous)
- Statistiques par bâtiment VoxCity (Cp, vitesse, hauteur originale)
- Export JSON + CSV avec métadonnées VoxCity
- Vue map interactive folium + visualisations PyVista focusées sur la zone intéressante

## 4. Résultats actuels
- Le chargement HDF5 fonctionne : ~30 bâtiments chargés avec coordonnées métriques valides (dépend de l'AOI VoxCity).
- Le prétraitement de géométrie fonctionne :
  - Filtrage individuel de chaque bâtiment (area < 1.0 m², height < 0.5 m)
  - Simplification et correction de chaque footprint indépendamment
  - Tous les bâtiments valides conservés après nettoyage (pas de fusion)
- L'analyse de géométrie détecte correctement les bâtiments très proches/chevauchants.
- Le sizing par proximité est disponible via `--mesh-constraint proximity`.
- **Maillage 3D Gmsh fonctionne** avec la nouvelle stratégie `cut()` séquentiel + `Mesh.Algorithm3D=4` (Delaunay) :
  - Données VoxCity réelles (AOI Paris 15e) : ✅ ~35000 nœuds, ~150000 cellules, 7 patches
  - Marges automatiques : 4H/7.5H/2D/1.25H (divisées par 2 par rapport à building_aero)
- **Pipeline complet testé** de VoxCity HDF5 → mesh → case → solveur → post-traitement : ✅
- **Simulation convergée** : t=2000s, résidus U=5e-5, p=4e-3, continuity=2e-6, k_max=14.0
- **Cp corrigé** : valeurs physiques cohérentes (cp_mean=-0.36, cp_min=-1.54, cp_max=1.01)
- **Rapport PDF généré** : `test_full_pipeline/report/voxcity_cfd_report.pdf` (9 pages)
- Post-traitement dédié génère :
  - `slice_pedestrian_velocity.png`
  - `wind_comfort_map.png`
  - `buildings_cp.png`
  - `slice_pedestrian_ti.png`
  - `slice_vertical_velocity.png`
  - `mesh_wireframe.png`
  - `map_view.html`
  - `voxcity_case_statistics.json`
  - `cell_data.csv`
- **Bâtiments polygonaux** : implémentés (`_extrude_polygon`) mais causent un crash solveur SIGFPE (maillage trop déformé : aspect ratio 1017, non-orthogonalité 89.8°)
- **Vérification HDF5** : `plot_voxcity_h5.py` permet de comparer les 21 bâtiments bruts du HDF5
- **Vérification geometry** : `verify_geometry.py` permet de comparer raw vs processed footprints individuels

## 5. Diagnostic initial (résolu)
- Les bâtiments VoxCity de l'AOI Paris 15e ont des chevauchements résiduels après reprojection → **gérés par `cut()` séquentiel** (primaire) avec fallback `fragment()` si échec.
- Le `cut()` multiple échouait sur les chevauchements → **cut() séquentiel par bâtiment** résout ce problème.
- La boîte fluide était mal reconstruite après le Boolean → **identification par triple test (bbox → Z → Point.within()) résout ce problème**.
- `Mesh.Algorithm3D = 5` ne maillait pas les volumes avec trous → **passage à `Mesh.Algorithm3D = 4` (Delaunay)**.
- Marges de domaine trop grandes → **divisées par 2** (4H/7.5H/2D/1.25H).
- Crash solveur SIGFPE → **stabilisé par maillage adaptatif, paramètres solveur alignés sur building_aero, et check qualité automatique**.
- Cp=0 sur bâtiments → **résolu par extraction correcte du champ de pression sur les patches wall**.
- Compilation LaTeX rapport PDF → **résolu par échappement des caractères spéciaux et helper `_latex_safe()`**.

## 6. Propositions d'intégration à foampilot

### 6.1 Modules à intégrer

| Module | Action proposée |
|--------|-----------------|
| `foampilot/src/foampilot/urban/readers/voxcity_reader.py` | **Déjà intégré** — maintenir et étendre avec VoxCity overlap processor |
| `foampilot/src/foampilot/mesh/direct_openfoam_exporter.py` | **Déjà intégré** — ok |
| `examples/building_geo/voxcity_export_work/src/vector_builder.py` | **Déplacer vers** `foampilot/src/foampilot/mesh/gmsh_urban_builder.py` |
| `examples/building_geo/neighborhood_demo/run_full_voxcity_pipeline.py` | **Déplacer vers** `foampilot/src/foampilot/urban/pipeline.py` |
| `examples/building_geo/neighborhood_demo/voxcity_dedicated_postprocess.py` | **Déplacer vers** `foampilot/src/foampilot/postprocess/voxcity_urban.py` |
| `examples/building_geo/neighborhood_demo/verify_geometry.py` | **Nouveau** — vérification visuelle des footprints bruts vs traités |

### 6.2 Nettoyage du dossier `examples/building_geo/`

```
examples/building_geo/
├── README.md                          # À mettre à jour
├── etat_lieux.md                      # Ce fichier
├── plan_export.md                     # Plan d'export (legacy)
├── rapport_calcul.md                  # Rapport détaillé (nouveau)
├── neighborhood_demo/
│   ├── config.json                    # Garder
│   ├── generate.py                    # Garder
│   ├── generate_synthetic.py          # SUPPRIMER (legacy)
│   ├── postprocess.py                 # SUPPRIMER (remplacé)
│   ├── run_full_voxcity_pipeline.py   # Garder + intégrer foampilot
│   ├── voxcity_dedicated_postprocess.py # Garder + intégrer foampilot
│   ├── verify_geometry.py             # 🆕 vérification visuelle footprints
│   ├── README.md                      # Mettre à jour
│   └── output/
│       └── voxcity.h5                 # SUPPRIMER (données téléchargeables)
├── voxcity_export_work/
│   └── src/
│       ├── vector_builder.py          # Garder + intégrer foampilot
│       ├── voxcity_vector_example.py  # Garder
│       └── voxcity_cached_example.py  # Garder
├── posts/                             # Garder (exemples post-traitement)
└── voxcity_postprocess.py             # Garder (legacy mais utile)
```

### 6.3 Intégration proposée dans foampilot

```
foampilot/src/foampilot/
├── urban/
│   ├── readers/
│   │   ├── voxcity_reader.py          # ✅ déjà là
│   │   └── base_reader.py             # ✅ déjà là
│   ├── model/
│   │   ├── urban_model.py             # ✅ déjà là
│   │   └── terrain.py                 # ✅ déjà là
│   └── pipeline.py                    # 🆕 run_full_voxcity_pipeline.py déplacé
├── mesh/
│   ├── direct_openfoam_exporter.py    # ✅ déjà là
│   └── gmsh_urban_builder.py          # 🆕 vector_builder.py déplacé
└── postprocess/
    ├── openfoam_pyvista.py            # ✅ déjà là
    └── voxcity_urban.py               # 🆕 voxcity_dedicated_postprocess.py déplacé
```

## 7. Prochaines étapes
1. **Intégration foampilot** : déplacer les modules dans foampilot/src/
2. **Nettoyage** : supprimer les fichiers legacy (`generate_synthetic.py`, `postprocess.py`, `voxcity.h5`)
3. **Documentation** : mettre à jour `README.md` et `examples/building_geo/README.md`
4. **Tests** : ajouter tests unitaires pour `VectorGmshBuilder` et `VoxCityReader`
5. **Amélioration** : ajouter support multi-directions vent comme dans `building_aero`

## 8. Statut
- ✅ Chargement VoxCity HDF5
- ✅ Reprojection métrique
- ✅ Prétraitement géométrie (filtrage, gaps, chevauchements)
- ✅ Sizing proximité Gmsh
- ✅ Maillage 3D Gmsh (stratégie cut séquentiel + Delaunay)
- ✅ Marges automatiques (4H/7.5H/2D/1.25H)
- ✅ Pipeline complet VoxCity → OpenFOAM → solveur → post-traitement
- ✅ Post-traitement dédié VoxCity-aware (map folium + PyVista)
- ✅ Simulation convergée (t=2000s, U=5e-5, p=4e-3, continuity=2e-6)
- ✅ Cp corrigé (valeurs physiques cohérentes)
- ✅ Rapport détaillé (`rapport_calcul.md`)
- ✅ Rapport PDF généré (`generate_report.py` → `voxcity_cfd_report.pdf`)
- ⏳ Intégration modules dans foampilot/src/
- ⏳ Nettoyage dossier examples/building_geo/
