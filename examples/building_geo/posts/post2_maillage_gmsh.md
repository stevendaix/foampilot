# Post 2 : Maillage Gmsh pour CFD urbaine

## Ce qu'on veut
Un maillage unique, mono-fluide, avec :
- des patches `inlet`, `outlet`, `top`, `ground`, `side_left`, `side_right`, `buildings`,
- une résolution fine autour des bâtiments,
- des tetraèdres valides.

## Comment on fait
1. On crée un volume fluide englobant.
2. On y soustrait les volumes bâtiments.
3. On assigne des `Physical Groups` Gmsh → patches OpenFOAM.
4. On maillage avec `gmsh.model.mesh.generate(3)`.
5. On compacte les points et on écrit `points`, `faces`, `owner`, `neighbour`, `boundary`.

## Points de vigilance
- Les bâtiments peuvent être plus petits que le maillage : on vérifie `min volume`.
- La topologie doit être valide : `checkMesh` doit passer.
- On évite les `pressureOutlet` dans `boundary` : on utilise `patch` + BC dans les champs.

## CheckMesh actuel
- 11645 tetraèdres
- non-orthogonalité max ~53 %
- skewness max ~1.03
- OK
