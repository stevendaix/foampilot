# Conversion MakeHuman avec meshio

La conversion `convert_makehuman_meshio.py` lit directement `base.npz`, transforme les 18 486 quadrilatères MakeHuman en 36 972 triangles, applique l’échelle `0.1` et écrit :

- `human_body.vtk`, avec les `cell_data` `makehuman_group` et `makehuman_face_id` ;
- `human_body.obj`, pour inspection géométrique ;
- `human_body.stl`, pour snappyHexMesh ;
- `meshio_report.json`.

Le fichier VTK est le format de référence pour conserver les groupes et les identifiants de faces. OBJ et STL sont des formats d’échange géométrique et ne conservent pas ces métadonnées de manière fiable.

Le test meshio produit exactement la même topologie que l’export actuel : 36 972 triangles, 154 composantes, 784 arêtes ouvertes et surface non fermée. Meshio convertit proprement le format et conserve les données de cellules, mais ne reconstruit pas les faces manquantes. La cause du défaut est donc présente dans le maillage MakeHuman `base.npz` lui-même, et non dans la conversion STL précédente.

La méthode recommandée est d’utiliser meshio comme étape de conversion structurée et de traçabilité, puis de décider séparément si une reconstruction de surface MakeHuman est nécessaire avant snappyHexMesh.
