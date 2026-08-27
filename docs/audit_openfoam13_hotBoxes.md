# Audit OF13 — multiRegion/film/hotBoxes

La référence OpenFOAM 13 fournit une Allrun-parallel. Elle exécute `blockMesh -region fluid`, `decomposePar -region fluid -noFields`, `subsetMesh -region fluid -noFields` en parallèle, `extrudeToRegionMesh -region fluid` en parallèle, `decomposePar -fields -copyZero`, `foamMultiRun` parallèle, `reconstructPar -allRegions` et `paraFoam -touchAll`.

Le dictionnaire `subsetMeshDict` sélectionne l’intérieur de quatre boîtes : `(0.4 0.1 0.1)-(0.6 0.3 0.3)`, `(0.4 0.1 0.4)-(0.6 0.3 0.6)`, `(0.4 0.4 0.1)-(0.6 0.6 0.3)` et `(0.4 0.4 0.4)-(0.6 0.6 0.6)`, en conservant la surface extérieure comme patch `film`. L’extrusion produit une couche film de `0.001`.

Le runner `186_multiRegion_film_hotBoxes/run.py` importe tous les champs et dictionnaires OF13 et reproduit les étapes avec FoamPilot. Il conserve la décomposition hiérarchique à douze domaines et transmet explicitement `-parallel` aux applications MPI.

La validation confirme la sélection des boîtes par `subsetMesh`, la création du film et le démarrage du couplage `fluid multicomponentFluid`/`film film`. Les espèces `O2/H2O`, les équations d’énergie et de film convergent; les températures évoluent de 300 K jusqu’à environ 338 K et les transferts de parcelles vers le film sont observés. Aucun `FOAM FATAL` ni erreur de dictionnaire n’est observé.

Le cas est très coûteux avec douze domaines : après environ 300 secondes, il atteint `Time≈0,358 s` sur `2 s` et est interrompu par le plafond de validation. La reconstruction finale n’est donc pas confirmée.

Statut : **accepté avec réserve — limite de temps et pression mémoire élevée**.
