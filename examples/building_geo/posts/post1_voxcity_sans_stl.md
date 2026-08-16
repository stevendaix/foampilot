# Post 1 : VoxCity vers OpenFOAM sans STL

## Le problème
L'approche classique VoxCity → STL → snappyHexMesh fonctionne, mais elle ajoute du bruit :
- téléchargement / export STL,
- réglage de la distance de snapping,
- gestion des angles de détection de features.

## La solution retenue
On garde les données vectorielles de VoxCity :
- `building_gdf` pour les bâtiments,
- DEM pour le terrain.

On construit directement la géométrie Gmsh du domaine fluide, on assigne les patches, et on exporte avec `DirectOpenFOamExporter`.

## Résultat
Un maillage tétraédrique valide dans `constant/polyMesh`, prêt pour `foamRun`.

## Fichier clé
`examples/building_geo/voxcity_vector_example.py`

## À venir
- tests avec données VoxCity réelles,
- benchmark de performance vs STL,
- visualisation 3D dans ParaView.
