# Post 5 : Prochaines étapes

## Ce qui fonctionne
- Pipeline VoxCity → Gmsh → OpenFOAM mono-fluide.
- Mise en données BCs réalistes pour vent urbain.
- Simulation convergée avec foampilot.

## Ce qui reste à faire
1. **Données réelles** : tester avec un plus grand quartier VoxCity, gérer les timeouts Earth Engine.
2. **Visualisation** : intégrer ParaView / PyVista pour vérifier les champs.
3. **Benchmark** : comparer voie vectorielle vs STL/snappyHexMesh en temps et qualité.
4. **Post-traitement** : extraction de profils de vitesse, coefficients de pression sur façades.
5. **Nettoyage** : factoriser les BCs dans `vector_builder.py` pour éviter les corrections manuelles.
6. **Documentation** : ajouter un exemple complet dans la doc foampilot.

## Une question ?
Ouvrir une issue ou une discussion dans le repo foampilot.
