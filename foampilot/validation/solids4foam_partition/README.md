# Validation de partition fluid-solid solids4foam

Ce cas vérifie la chaîne Foampilot suivante : géométrie CAD paramétrique, partition conforme en deux régions, création automatique des groupes physiques, export direct des maillages OpenFOAM et présence du patch `interface` dans les deux régions.

Depuis la racine du projet Foampilot :

```bash
PYTHONPATH=src python validation/solids4foam_partition/run.py
```

Le script utilise uniquement les fabriques Foampilot. Il ne lance pas le solveur `solids4Foam`; cette validation est volontairement déterministe et rapide. Elle doit être suivie d’un `checkMesh -region fluid`, d’un `checkMesh -region solid`, puis d’un calcul court avec les champs et schémas adaptés à la version locale de solids4foam.
