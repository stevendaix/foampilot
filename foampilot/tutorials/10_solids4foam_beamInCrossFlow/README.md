# solids4foam — beamInCrossFlow avec Foampilot

Ce tutoriel construit un cas fluide–structure à deux régions avec les API Foampilot. La géométrie est partitionnée avec Gmsh, les groupes physiques `FLUID`, `SOLID` et `interface` sont créés automatiquement, puis le maillage est exporté directement dans les répertoires OpenFOAM régionaux.

Depuis la racine du dépôt :

```bash
PYTHONPATH=src python tutorials/10_solids4foam_beamInCrossFlow/run.py
```

Pour exécuter réellement le solveur après génération, utiliser le workflow Foampilot retourné par `build_beam_in_cross_flow` avec `workflow.run()`. Le cas nécessite une installation fonctionnelle de Gmsh, OpenFOAM et solids4foam. Le script n’appelle ni `gmshToFoam`, ni `RunFunctions`, ni un script shell externe.
