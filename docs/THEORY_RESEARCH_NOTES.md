# Notes de recherche pour l’approfondissement théorique

## Sources consultées

La documentation YADE `FoamCoupling` décrit deux familles : le couplage ponctuel et le couplage basé sur la fraction volumique. Elle donne le noyau gaussien, la largeur `sigma = delta/(2 sqrt(2 ln 2))`, la fraction solide `epsilon_s,c = sum(V_p,i G_i,c)/V_c`, l’interpolation eulérienne vers une particule, la loi de traînée Schiller–Naumann, la force d’Archimède/ambiante et la masse ajoutée. Elle rappelle aussi l’usage de MPI et la condition de résolution du diamètre particulaire par rapport à la maille.

La documentation preCICE présente l’adaptateur OpenFOAM comme un plug-in/function object permettant à OpenFOAM de participer à un couplage externe générique. Cela sert de comparaison avec le MPI direct de YADE utilisé dans FoamPilot.

## Vérifications dans le code local

`FoamYade::locatePt()` utilise `meshSearch::New(mesh).findCell(pt)` en mode ponctuel, et `mshTree.nnearestCellsRange(pt, interpRange, true)` en mode gaussien. `calcInterpWeightGaussian()` calcule un poids exponentiel à partir de la distance entre le centre de cellule et la particule, puis renormalise les poids par leur somme.

`buildCellPartList()` accumule dans chaque cellule le volume particulaire pondéré et la vitesse particulaire pondérée. `setCellVolFraction()` calcule ensuite la fraction fluide `alpha = 1 - V_p/V_c`, applique une borne inférieure à `0.10` dans le code actuel et reconstruit `uParticle`.

En mode gaussien, `hydroDragForce()` interpole `U` et `alpha`, calcule la vitesse relative, `Re`, `Cd`, puis un coefficient de traînée. Pour `alpha_f > 0.8`, la branche Schiller–Naumann est utilisée; sinon le code utilise une fermeture dense de type Ergun-Wen-Yu avec termes visqueux et inertiels. La force particulaire est `pv*coeff*urelvel/alpha_p`, et la contribution fluide est accumulée avec le signe opposé dans `uSourceDrag`.

`archimedesForce()` utilise le gradient de pression et la divergence de contrainte visqueuse interpolés, puis ajoute l’opposé aux sources fluides. `addedMassForce()` existe dans le code mais n’est pas appelé par `calcHydroForce()` dans la version actuelle. Les couples hydrodynamiques sont évalués à partir de l’antisymétrie du gradient de vitesse.

En mode ponctuel, `stokesDragForce()` utilise l’interpolation OpenFOAM du champ `U` dans la cellule de la particule et applique `3*pi*d_p*nu*rhoF*(U_f-U_p)`, avec l’action opposée répartie dans la cellule.

## Références

[1]: https://yade-dem.org/doc/FoamCoupling.html "YADE FoamCoupling documentation"
[2]: https://precice.org/adapter-openfoam-overview "preCICE OpenFOAM adapter"
[3]: https://www.cfdem.com/media/CFDEM/docu/CFDEMcoupling_Manual.html "CFDEM coupling manual"
