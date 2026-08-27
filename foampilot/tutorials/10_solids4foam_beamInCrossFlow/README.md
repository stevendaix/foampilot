# solids4foam — beamInCrossFlow avec Foampilot et OpenFOAM Foundation 13

Ce tutoriel construit un cas fluide–structure à deux régions avec les API Foampilot. La géométrie est partitionnée avec Gmsh, les groupes physiques `FLUID`, `SOLID` et `interface` sont créés automatiquement, puis le maillage est exporté directement dans les répertoires OpenFOAM régionaux.

Depuis la racine du dépôt :

```bash
PYTHONPATH=src python tutorials/10_solids4foam_beamInCrossFlow/run.py
```

Le cas utilise le profil natif Foundation 13 : `physicalProperties`, `momentumTransport` avec modèle Stokes, solveur solide `implicitSegregated`, et `decomposeParDict` multi-région. Pour exécuter le smoke test sériel validé :

```bash
cd foampilot/tutorials/10_solids4foam_beamInCrossFlow/case
solids4Foam
```

Pour la validation MPI à deux processus :

```bash
decomposePar -allRegions
mpirun --allow-run-as-root --oversubscribe -np 2 solids4Foam -parallel
```

Le cas nécessite une installation fonctionnelle de Gmsh, OpenFOAM Foundation 13 et la bibliothèque `libsolids4FoamModels.so` compilée avec le profil minimal Foundation 13. Le script de génération n’appelle ni `gmshToFoam`, ni `RunFunctions`, ni un script shell externe.
