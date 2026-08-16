# Test Tracking — Détection Inlet/Outlet

| ID  | Description | Status | Agent | Patch View | À intégrer? | Notes |
|-----|-------------|--------|-------|------------|-------------|-------|
| P01 | Détection boucles de bord (vtkFeatureEdges) | ✅ | — | — | — | 1704 boucles, image générée |
| P02 | Centerline simulée + extrémités par PCA | ✅ | — | — | — | 2 extrémités, image générée |
| P03 | Angle normale / tangente locale (Section 3.5) | ✅ | — | — | — | Image et résultats générés |
| P04 | Détection caps plans (PCA / RANSAC / compacité) | ✅ | — | — | — | 2 caps détectés, image générée |
| P05 | Détection paroi cylindrique (RANSAC / normales radiales / courbure) | ✅ | — | — | — | Cylindre R=0.0682, 1057 faces paroi, image générée |
| P06 | Méthodes par courbure (moy/gauss/principale, seuil) | ✅ | — | — | — | 5131 cap, 6877 wall, image générée |
| P07 | Region growing par similarité de normales (Section 7.1) | ✅ | — | — | — | 430 régions, 137 cap candidates, image générée |
| P08 | Clustering KMeans/DBSCAN sur features de faces (Section 8) | ✅ | — | — | — | 10593 faces, KMeans=3 clusters, DBSCAN=1 cluster, image générée |
| P09 | PCA globale et locale (Section 9) | ✅ | — | — | — | Image et résultats générés |
| P10 | Squelette géométrique / medial axis simulé (Section 10) | ✅ | — | — | — | Image et résultats générés |
| P11 | Méthodes par champ de distance (Section 11) | ✅ | — | — | — | Image et résultats générés |
| P12 | Méthodes par coupes / slicing (Section 12) | ✅ | — | — | — | Image et résultats générés |
| P13 | Méthodes par graphe topologique (Section 13) | ✅ | — | — | — | Image et résultats générés |
| P14 | Méthodes par distance géodésique (Section 14) | ✅ | — | — | — | Image et résultats générés |
| P15 | Méthodes par formes primitives / features (Section 15) | ✅ | — | — | — | Plane=10039 faces, Sphere=9424 faces, Cyl=9799 faces, SOR=0.066, image générée |
| P16 | Méthodes par apprentissage automatique (Section 16) | ✅ | — | — | — | Image et résultats générés |
| P17 | Méthodes par sélection utilisateur / interactives (Section 17) | ✅ | — | — | — | Image et résultats générés |
| P18 | Méthodes par convention (Section 18) | ✅ | — | — | — | Image et résultats générés |
| P19 | Méthodes hybrides / vote (Section 19) | ✅ | — | — | — | 1384 opening, 135 wall, 9074 uncertain, conf=0.79, validation topo OK, image générée |
| P20 | Pipeline recommandé VTK/VMTK (Section 20) | ✅ | — | — | — | Surface fermée, 0 arêtes bord, 0 inlet, 1 outlet, image générée |
| P21 | Stratégie robuste (Section 21) | ✅ | — | — | — | Image et résultats générés |
| P22 | Critères numériques recommandés (Section 22) | ✅ | — | — | — | Surface fermée, 0 ouverture, image générée |
| P23 | Tests prioritaires (Section 23) | ✅ | — | — | — | 639 boucles, 2 caps filtrés, image générée |
| P24 | Conclusion (Section 24) | ✅ | — | — | — | 639 boucles, 2 caps, mean_angle=67°, validation topo OK, convention OK |
| T1  | Classification par angle de la normale | ⏳ | — | — | — | — |
| T2  | Titrage OpenFOAM vs géométrie | ⏳ | — | — | — | — |
| T3  | Vaisseau courbe | ⏳ | — | — | — | — |
| T4  | Centerline hors zone d'écoulement | ⏳ | — | — | — | — |
| T5  | Velocity profile valide | ⏳ | — | — | — | — |
| T6  | Caméra alignée | ⏳ | — | — | — | — |

Legend : ⏳ = en attente, ✅ = passé, ❌ = échoué, 🔄 = en cours
