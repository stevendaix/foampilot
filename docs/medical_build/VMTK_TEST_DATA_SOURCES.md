# Sources VMTK test data vérifiées

Le dépôt officiel est : https://github.com/vmtk/vmtk-test-data

Révision inspectée localement : dépôt cloné sous `/tmp/vmtk-test-data` le 20 août 2026.

Les fichiers vasculaires officiels identifiés sont :

- `input/aorta-surface.vtp`
- `input/aorta-surface.stl`
- `input/aorta-surface-open-ends.stl`
- `input/aorta-surface-branch-split.vtp`
- `input/aorta-surface-connectivity-reference.stl`
- `input/aorta-surface-segment-2.stl`
- `input/aorta-centerline.vtp`
- `input/aorta-centerline-branches.vtp`
- `input/aorta-centerline-attribute-branches.vtp`
- `input/aorta-centerline-referencesystem.vtp`
- `input/aorta-mesh.vtu`
- `meshreference/aorta-mesh-external-layer.vtu`

Les références de tests de surface pertinentes comprennent notamment `surfacebooleanoperation`, `surfaceappend`, `surfaceconnectivity`, `surfacereconnection`, `surfacemeshing`, `surfacerepair`, `branchclipper` et `marchingcubes` sous `surfacereference/`.

Mesures VTK relevées : `input/aorta-surface.vtp` contient 6 468 points et 12 932 cellules ; `input/aorta-centerline.vtp` contient 409 points et 2 lignes ; `input/aorta-centerline-branches.vtp` contient 417 points et 6 lignes. Les copies présentes dans `foampilot/test/vmtk_test_data` ont les mêmes tailles, bornes et arrays.

La campagne complexe de foampilot est différente : son `centerlines.vtp` contient 1 694 points et 8 lignes, avec les arrays avancés `Abscissas`, `FrenetTangent`, `ParallelTransportNormals`, `ParallelTransportBinormals`, `TraceCellIds`, `TracePCoords`, `Curvature`, `Torsion` et `Tortuosity`. La surface source correspondante n’est pas dans le dépôt officiel VMTK test data inspecté.
