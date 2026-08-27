# Audit OF13 — multiRegion/CHT/circuitBoardCooling

Source locale : `/opt/openfoam13/tutorials/multiRegion/CHT/circuitBoardCooling`.

L’Allrun officielle sélectionne par défaut la stratégie `extrudeFromInternalFaces`, exécute la préparation `Allmesh-extrudeFromInternalFaces`, puis `foamMultiRun`. Cette préparation exécute `blockMesh -region fluid`, `createZones -region fluid`, `extrudeToRegionMesh -region fluid -dict system/fluid/extrudeToRegionMeshDict.extrudeFromInternalFaces` et `createBaffles -region fluid -dict system/fluid/createBafflesDict.baffle1D`. Le cas comporte les régions `fluid` et `baffle3D`, des champs initiaux régionaux, une géométrie `baffle1D.stl` et `baffle3D.stl`, ainsi que des dictionnaires de baffle thermique compressible.

Le contrôle source définit `fluid fluid`, `baffle3D solid`, `endTime=5000`, `deltaT=1` et écriture toutes les 500 étapes. Le runner `170_multiRegion_CHT_circuitBoardCooling/run.py` importe les champs, dictionnaires, géométries et l’include racine `include/wallPatchFields` via FoamPilot. Cet include est requis par `createBafflesDict.baffle1D`; son absence a été détectée lors de la première validation puis corrigée par import d’actif, sans commande shell dans le runner.

La validation OF13 réussit la création du maillage fluide, des zones, l’extrusion de la région `baffle3D`, la création du baffle 1D et le calcul `foamMultiRun`. Le calcul couplé atteint `Time=5000 s` et `End`; les résidus fluide et solide restent traités, et aucun `FOAM FATAL` n’est observé. La variante `extrudeFromPatches` reste à traiter séparément si elle est requise comme cas de test; la stratégie officielle par défaut est validée. Aucune nouvelle API nécessaire.
