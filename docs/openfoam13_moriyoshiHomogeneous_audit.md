# Audit OpenFOAM 13 — XiFluid/moriyoshiHomogeneous

La référence normative est `/opt/openfoam13/tutorials/XiFluid/moriyoshiHomogeneous`. Le cas racine exécute deux variantes : le cas propane `moriyoshiHomogeneous/moriyoshiHomogeneous` et une copie hydrogène obtenue en remplaçant `physicalProperties` et `combustionPropertiesInclude` par leurs variantes hydrogen.

Le solver de référence est `foamRun` avec le module `XiFluid`. Le maillage est un `blockMeshDict` direct, avec les patches `left`, `right`, `top`, `bottom` de type `symmetryPlane` et `frontAndBack` de type `empty`. Les champs initiaux comprennent `p`, `U`, `T`, `Tu`, `Xi`, `alphat`, `b`, `ft`, `fu`, `k`, `epsilon` et `nut`.

Le dictionnaire `constant/fvModels` définit une ignition `constantbXiIgnition` dans la zone `ignition`, avec `start 0`, `duration 0.003`, `strength 2` et correction cylindrique `XiCorr`. Les propriétés de combustion propane utilisent le modèle de vitesse laminaire Gulder; la variante hydrogène utilise RaviPetersen et ses tables `alpha`/`beta`.

La séquence FoamPilot doit donc générer le cas propane, importer ou sérialiser ses dictionnaires OF13 via les APIs publiques, exécuter `blockMesh`, lancer `foamRun -solver XiFluid`, puis générer la variante hydrogène dans un second répertoire FoamPilot en conservant la même géométrie et les réglages système, avec uniquement les fichiers de propriétés de phase/combustion remplacés.
