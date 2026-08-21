# Résultats de comparaison NetworkX — cas complexe

Le fichier analysé est `examples/medical_build/case_complex/analysis/centerlines.vtp`.

La lecture VTK donne 8 cellules et 1694 points. La reconstruction NetworkX avec une tolérance de regroupement des endpoints de 2,0 unités donne 9 nœuds, 8 arêtes, une seule composante connexe et un arbre.

Les nœuds terminaux sont N1 à N8, tous de degré 1. Le nœud N0 est l’unique nœud de jonction, de degré 8. Cela correspond à une topologie en étoile : un tronc commun et huit branches terminales dans le centerline complexe.

La vue 2D colore N0 en rouge comme jonction et N1–N8 en vert comme terminaux. La vue 3D confirme la présence de huit trajectoires qui partent d’un centre commun, avec huit endpoints noirs. Cette image est différente du petit fichier VMTK `aorta-centerline-branches.vtp`, qui possède 6 cellules mais seulement deux chaînes principales.

Fichiers générés :

- `complex_networkx_topology.json`
- `complex_networkx_graph_2d.png`
- `complex_networkx_overlay_3d.png`

## Comparaison géométrique strictement alignée

Une comparaison a été exécutée avec la surface complexe partitionnée `aorta_surface_patches.vtp` et les huit STL `case_complex/exports_complex/manual_stl/branch_00.stl` à `branch_07.stl`, tous dans le même repère. La référence a un volume de 61 304,78 et le candidat obtenu par simple append des huit branches a un volume de 250 835,55, soit une erreur relative de +309,16 %. Les deux objets sont connexes, mais cela ne valide pas la géométrie : les branches manuelles contiennent des volumes qui se recouvrent ou des extensions de jonction, et l’append n’est pas une union booléenne.

La vue superposée montre que la comparaison est maintenant spatialement cohérente, mais que le candidat rouge est plus volumineux autour du tronc et des jonctions. La comparaison précédente de volume n’était donc pas une comparaison de deux objets équivalents. Il faut tronquer les huit branches au nœud NetworkX, conserver les caps correspondant aux mêmes ouvertures, puis réaliser une vraie union/clean avant de comparer la surface globale.
