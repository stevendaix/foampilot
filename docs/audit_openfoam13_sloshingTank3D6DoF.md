# Audit OF13 — incompressibleVoF/sloshingTank3D6DoF

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank3D6DoF`.

L’`Allrun` officiel exécute séquentiellement `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/sloshingTank3D`, `setFields`, puis `foamRun`. La référence contient `constant/6DoF.dat` pré-généré (6456 octets) et un générateur auxiliaire `gen6DoF/gen6DoF.C`; le runner FoamPilot importe directement le fichier de données officiel et n’exécute pas de compilation auxiliaire.

Les paramètres de `system/controlDict` sont `solver incompressibleVoF`, `endTime 40`, `deltaT 0.01`, `writeInterval 0.05`, `adjustTimeStep yes`, `maxCo 0.5`, `maxAlphaCo 0.5`, `maxDeltaT 1`. Le maillage officiel `blockMesh/sloshingTank3D` produit initialement 25 840 cellules. `system/decomposeParDict` prévoit 16 domaines hiérarchiques, mais l’Allrun de référence reste séquentiel. `setFieldsDict` initialise `alpha.water` dans la zone `water`.

`constant/dynamicMeshDict` utilise `solidBody` et `sixDoFMotion`, avec deux lectures du fichier `$FOAM_CASE/constant/6DoF.dat`, ainsi qu’un champ de raffinement dynamique piloté par `alpha.water`. Le code `gen6DoF.C` génère 100 échantillons sur 40 s : amplitudes de translation `(2,3,2) m`, fréquences `(0.5,0.8,0.4) rad/s`, amplitudes de rotation `(30,10,10) deg` et fréquences `(0.4,0.7,0.5) rad/s`.

Runner créé : `foampilot/tutorials/140_incompressibleVoF_sloshingTank3D3DoF/run.py`. API utilisées : `BaseSolver.import_reference_asset`, `CaseFieldsManager.import_reference_field`, `BaseSolver.run_command`; aucune nouvelle API.
