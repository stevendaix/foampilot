# Audit de la topologie du domaine air–homme

La boîte définie par `blockMeshDict` va de `x=-0.75` à `0.75`, `y=-1.10` à `1.10` et `z=-0.40` à `0.60`. Son volume géométrique est donc :

`V_box = 1.5 × 2.2 × 1.0 = 3.300 m³`.

La STL humaine a une aire de 4,563 m² et un volume trimesh estimé de 0,1984 m³. Si elle était fermée et complètement soustraite, le volume d’air attendu serait environ :

`V_air_attendu = 3.300 − 0.1984 = 3.1016 m³`.

`checkMesh` donne pour le maillage CFD actuel : `V_air_CFD = 3.2311102 m³`, soit environ 0,1295 m³ de plus que cette estimation, environ 4,2 % de l’air attendu. La différence ne prouve pas à elle seule une erreur de soustraction, car la STL n’est pas étanche et le volume trimesh est seulement une estimation; elle montre toutefois que la géométrie humaine ne forme pas une frontière volumique parfaitement fermée.

La configuration snappyHexMesh utilise bien `locationInMesh (0.0 -1.0 0.0)`, un point situé dans l’air à l’extérieur du corps. Les cellules conservées sont donc celles de la boîte connectées à cette région, et l’intérieur de la surface humaine est censé être retiré. Le patch `human` est créé avec 20 223 faces.

Cependant, `checkMesh` signale `human: multiply connected (shared edge)`. La STL MakeHuman est également signalée `watertight=False`. Le domaine est donc conceptuellement une boîte avec un volume humain retiré, mais la soustraction n’est pas topologiquement propre partout. Il faut réparer ou remailler la STL avant une validation quantitative des volumes et des flux.
