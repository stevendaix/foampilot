# Audit OF13 — multiRegion/CHT/VoFcoolingCylinder2D

Source locale : `/opt/openfoam13/tutorials/multiRegion/CHT/VoFcoolingCylinder2D`.

L’Allrun officielle exécute `blockMesh`, `splitMeshRegions -cellZones`, `setFields -region fluid`, touche les sorties ParaView, puis `foamMultiRun`. La mise en données crée deux zones cellulaires `fluid` et `solid` dans un maillage cylindrique 2D, avec une interface couplée `fluid_to_solid/solid_to_fluid`. Le contrôle régional définit `fluid compressibleVoF` et `solid solid`, avec `endTime=5`, `deltaT=1e-4`, pas ajustable, `maxCo=2`, `maxAlphaCo=1` et écriture toutes les `0,1` unités.

Le runner `169_multiRegion_CHT_VoFcoolingCylinder2D/run.py` importe les dictionnaires globaux et régionaux ainsi que les champs initiaux `0/fluid` et `0/solid` via FoamPilot, puis reproduit les étapes de l’Allrun. Une première validation a révélé que `import_reference_field` attend la racine du cas pour construire `0/<field_name>`; le runner a donc été corrigé pour transmettre des noms préfixés `fluid/...` ou `solid/...`, ce qui constitue l’usage correct et généralisable de l’API existante. Aucune nouvelle API n’a été nécessaire.

La validation OF13 corrigée réussit `blockMesh` avec `3264` cellules, sépare correctement les régions fluid/solid, initialise `alpha.water` avec `setFields` et exécute `foamMultiRun` jusqu’à `Time=5 s` et `End`. Les solveurs fluide et solide convergent, les nombres de Courant restent contrôlés malgré l’ajustement automatique du pas, et aucun `FOAM FATAL` n’est observé. Statut : validé OF13.
