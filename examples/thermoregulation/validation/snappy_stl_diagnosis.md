# Diagnostic STL et snappyHexMesh

## Résultat principal

Le problème dominant ne vient pas d’abord des paramètres de raffinement de `snappyHexMesh`, mais de la topologie de `human.stl`.

`surfaceCheck` OpenFOAM 13 rapporte :

- 36 972 triangles ;
- aucun triangle illégal ;
- 784 arêtes connectées à une seule face ;
- 0 arête connectée à plus de deux faces ;
- 154 parties non connectées ;
- 154 zones à orientation normale cohérente ;
- surface non fermée et orientations multiples.

Une STL composée de 154 parties et comportant 784 arêtes ouvertes ne peut pas définir proprement un volume solide à soustraire du cube. `snappyHexMesh` peut raffiner et snapper cette surface, mais `mergeTolerance`, `nSmoothPatch`, `tolerance` ou `nSolveIter` ne peuvent pas créer de manière fiable les faces manquantes ni décider quelles parties doivent être fusionnées.

## Configuration actuelle

La configuration conserve bien la région d’air avec `locationInMesh (0 -1 0)` et définit la surface humaine comme un patch `wall`. La boîte possède un volume de 3,300 m³. Le volume CFD obtenu est 3,2311 m³, alors que boîte moins volume STL estimé donnerait environ 3,1016 m³. Cet écart est cohérent avec une surface STL non fermée et ne doit pas être corrigé par un simple facteur numérique.

## Réglages recommandés après réparation de la STL

Pour une STL fermée et orientée, utiliser une configuration intermédiaire : `human { level (2 3); }`, `nCellsBetweenLevels 3`, `nSmoothPatch 5`, `tolerance 1.0`, `nSolveIter 80`, `nRelaxIter 8`, `maxGlobalCells` de 600 000 environ et `addLayers false` dans un premier temps. Le niveau 3 doit être réservé à la peau et éventuellement à une zone de distance de quelques centimètres autour de la surface; raffiner tout le volume de la boîte serait inutile et coûteux.

Les couches limites ne doivent être activées qu’après validation de la topologie et de la stabilité. Elles peuvent améliorer le calcul de `h`, mais rendent la génération et la convergence plus sensibles autour des doigts, mains, pieds et zones à faible épaisseur.

## Conclusion

La priorité est de produire une STL humaine fermée, avec un nombre limité de composantes correspondant réellement au corps, puis de relancer `surfaceCheck`, `snappyHexMesh` et `checkMesh`. L’optimisation des paramètres snappyHexMesh est secondaire tant que `surfaceCheck` signale des arêtes ouvertes et 154 parties séparées.
