# Intégration multiphysique OpenFOAM 13

Cette extension ajoute à Foampilot un contrat d’intégration explicite pour `sediFoam`, `openHFDIB-DEM` et `libAcoustics`. Les sources portées d’openHFDIB-DEM sont conservées dans `third_party/openHFDIB-DEM` afin de rendre le build reproductible depuis Foampilot.

## Architecture

| Module | Physique | Référence amont | État dans cette branche |
| --- | --- | --- | --- |
| `sediFoam` | CFD–DEM avec LAMMPS et transport sédimentaire | [dépôt sediFoam][1] | Profil Foampilot ; portage LAMMPS à traiter séparément |
| `openHFDIB-DEM` | CFD–DEM immersed-boundary | [dépôt openHFDIB-DEM][2] | Bibliothèque portée et compilée avec OpenFOAM Foundation 13 ; solveurs encore à adapter |
| `libAcoustics` | Sources acoustiques et FW-H | [branche v2512][3] | Profil Foampilot ; convention OpenFOAM+ à isoler du backend Foundation |

Les deux backends DEM sont mutuellement exclusifs dans un même cas. L’acoustique est modélisée comme une physique de mesure ou de post-traitement et ne concurrence pas le solveur DEM.

## Portage openHFDIB-DEM validé

La bibliothèque `libHFDIBDEM.so` compile avec OpenFOAM Foundation 13 après les adaptations suivantes :

| Ancienne API | Adaptation OF13 |
| --- | --- |
| `fvCFD.H` | Shim local minimal incluant `fvMesh.H`, `volFields.H`, `surfaceFields.H`, `fvc.H`, `fvm.H`, `dimensionedTypes.H`, `IOdictionary.H`, `Pstream.H`, `randomGenerator.H`, `plane.H`, `fvMeshSubset.H`, `scalarMatrices.H` et `uniformDimensionedFields.H`. |
| `triSurfaceMesh` | Alias local vers `triSurface`, avec inclusions explicites de `triSurfaceSearch`, `treeDataTriSurface`, `indexedOctree` et `triangleFuncs`. |
| `triSurfaceMesh::movePoints` | Mise à jour contrôlée du `pointField` de `triSurface`, car l’API OF13 n’expose plus cette méthode. |
| `triSurfaceMesh::getNormal` | Calcul de la normale par `triFace::normal(pointField)` à partir de l’index retourné par `triSurfaceSearch`. |
| `Random` | Alias vers `randomGenerator`, avec conversion des constructeurs de graine historiques. |
| `meshSearch(mesh_)` | `meshSearch::New(mesh_)`, le constructeur direct étant protégé en OF13. |
| `cellZones().findZoneID` | `cellZones().findIndex`. |
| `scalarRectangularMatrix` et `scalarSquareMatrix` | Inclusion explicite de `scalarMatrices.H`. |
| `unallocLabelList` | `labelUList`. |
| `Time::timeName()` | `Time::timeName(mesh_.time().value())`. |
| Bibliothèques legacy `dynamicMesh`/`dynamicFvMesh` | Retrait des chemins et bibliothèques absents de Foundation 13 pour la bibliothèque compilée. |

Les modifications sont conservées dans les sources vendorisées sous `third_party/openHFDIB-DEM/src/HFDIBDEM` et dans les options de build correspondantes. Les répertoires `lnInclude` restent générés par `wmakeLnInclude` et ne sont pas versionnés.

## Vérifications réalisées

La validation doit être exécutée avec une installation OpenFOAM Foundation 13 fournie par l’utilisateur via `FOAM_BASHRC` ou `WM_PROJECT_DIR`. Le script portable suivant construit la bibliothèque portée sans chemin machine codé en dur :

```bash
export FOAM_BASHRC=/chemin/vers/OpenFOAM-13/etc/bashrc
./tools/build_openhfdib_of13.sh
```

La bibliothèque `libHFDIBDEM.so` est le composant actuellement porté et compilable. Les solveurs doivent être considérés séparément : le premier essai historique a rencontré des API supprimées (`dynamicFvMesh.H`, `createDynamicFvMesh.H`, `fvOptions.H`). La variante statique de `HFDIBDEMFoam` a ensuite reçu un portage spécifique, tandis que `pimpleHFDIBFoam` conserve des adaptations métier non entièrement qualifiées.

Cette distinction est volontaire : la bibliothèque est portée et compilable, mais la compatibilité complète des deux solveurs et la validation scientifique d’un cas DEM ne sont pas déclarées terminées sans une relance dans un environnement Foundation 13 propre.

## Utilisation Foampilot

```python
from foampilot import MultiphysicsConfiguration

config = MultiphysicsConfiguration(("openhfdib_dem", "libacoustics"))
config.write_case_assets("./case")
```

La commande produit `system/foampilotMultiphysics.json` pour l’audit et `system/foampilotMultiphysics` comme dictionnaire lisible. Les propriétés indispensables, notamment `nu`, doivent continuer à être écrites explicitement dans `constant/transportProperties`.

## Références

[1]: https://github.com/xiaoh/sediFoam "Dépôt sediFoam"

[2]: https://github.com/techMathGroup/openHFDIB-DEM "Dépôt openHFDIB-DEM"

[3]: https://github.com/unicfdlab/libAcoustics/tree/v2512 "Branche v2512 de libAcoustics"

[4]: https://openfoam.org/download/13-ubuntu/ "Installation officielle OpenFOAM 13 pour Ubuntu"

## État du portage des solveurs après modification directe

Lors d’une validation précédente, la variante `HFDIBDEMFoam` à maillage fixe a été modifiée directement puis compilée avec succès sous OpenFOAM Foundation 13. L’exécutable produit était `HFDIBDEMFoam` dans `$FOAM_USER_APPBIN`; cette validation doit être reproduite avec l’environnement portable documenté. Le portage remplace `dynamicFvMesh` par `fvMesh`, crée `fvModels` et `fvConstraints`, supprime les mises à jour de maillage et conserve un refus explicite de toute configuration `dynamicRefineFvMesh`.

Le portage statique de `pimpleHFDIBFoam` est engagé mais n’est pas encore compilable. Les erreurs restantes ne sont plus des includes omnibus : elles concernent notamment la collision de nom entre l’ancien `physicalProperties` DEM et la classe Foundation 13, `setRefCell`, `setFluxRequired`, `moveMeshOuterCorrectors`, les méthodes MRF supprimées (`correctBoundaryVelocity`, `zeroFilter`), `constrainHbyA`, `constrainPressure` et la variable `laminarTransport`. Ces éléments nécessitent une adaptation métier et ne doivent pas être remplacés par des aliases aveugles.

Les journaux de compilation associés doivent être conservés avec la PR. La commande reproductible pour le solveur DEM, lorsque l’environnement Foundation 13 et les dépendances du cas sont disponibles, est :

```bash
export FOAM_BASHRC=/chemin/vers/OpenFOAM-13/etc/bashrc
cd third_party/openHFDIB-DEM/applications/solvers/pureDEM/HFDIBDEMFoam
wclean
wmake -j1
```

## Validation runtime OF13

Lors d’une validation précédente, le cas `examples/01_LIGGGHTSVerificationTests/test01_normalForceTest/openHFDIB-DEM/RestituionCoeff-0.06` a été copié dans `validation/normalForce_OF13`. Après génération du maillage par `blockMesh`, le solveur `HFDIBDEMFoam` a initialisé les champs, lu les propriétés de fluide, créé les deux corps sphériques et exécuté cinquante pas temporels de `0.0001` à `0.005` s sans erreur fatale. Le journal contenait les séquences `Time = ...`, `updated HFDIBDEM` et `ExecutionTime = ...`; cette exécution doit être reproduite dans un environnement Foundation 13 disponible.

Le portage de `pimpleHFDIBFoam` reste à qualifier après migration de `viscosityModel`, `incompressibleMomentumTransportModels`, `fvModels`, `fvConstraints`, `constrainHbyA`, `constrainPressure`, `findRefCell` et des interfaces MRF Foundation 13. La variante visée est statique : les mécanismes de mouvement et de raffinement topologique ne doivent pas être exposés comme fonctionnels tant qu’une compilation et un cas runtime dédiés ne sont pas reproduits.

## Suite de validation approfondie

Une suite reproductible est fournie par `run_deep_validation.py`. Elle vérifie les tests unitaires Foampilot, la génération du manifeste pour la combinaison openHFDIB-DEM/libAcoustics, la validité JSON, le parsing du dictionnaire Foundation 13, la présence de la bibliothèque HFDIBDEM, `-help` pour les deux solveurs, `checkMesh`, la configuration `staticFvMesh` et l’exécution d’un cas DEM prolongé.

Le rapport historique de validation indiquait **14 contrôles réussis sur 14 contrôles applicables**, avec 150 marqueurs `Time =` et 50 mises à jour `updated HFDIBDEM` dans le cas DEM prolongé. Ce résultat doit être considéré comme une référence à reproduire, et non comme un contrôle exécuté dans le checkout courant tant qu’une installation Foundation 13 n’est pas disponible.

La combinaison de profils `openhfdib_dem` et `libacoustics` est acceptée par Foampilot et son manifeste est valide. Cette validation reste une validation de contrat et de génération, pas une validation de la bibliothèque acoustique elle-même : le clone `libAcoustics` contient bien des sources `.C`/`.H` sous ses sous-arbres, mais aucun orchestrateur de build Foundation 13 directement exécutable n’a été identifié dans la PR. La production de `libAcoustics.so` doit donc rester une étape séparée et explicitement qualifiée.

De même, la tentative de build direct de `sediFoam/lammpsFoam` échoue avant compilation, car le répertoire racine ne contient pas `Make/options`. Le module nécessite son orchestration de build dédiée ainsi que l’interface LAMMPS ; cette suite ne prétend donc pas valider un backend sediFoam compilé. Le résultat est classé comme **limitation d’artefact upstream**, et non comme échec du solveur openHFDIB-DEM.
