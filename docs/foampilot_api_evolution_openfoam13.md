# Historique de l’évolution de l’API FoamPilot pour OpenFOAM 13

> Ce document est la source dédiée au suivi des changements de l’API FoamPilot nécessaires à l’intégration des tutoriels OpenFOAM 13. Il doit être complété à chaque ajout ou modification de fonction. La matrice [`openfoam13_foampilot_integration.md`](openfoam13_foampilot_integration.md) conserve en parallèle la liste des fonctions utilisées par tutoriel.

## Règle de traçabilité

Chaque évolution de l’API doit être documentée avant que le tutoriel qui la nécessite soit marqué comme validé. L’entrée doit préciser le module et la méthode, le besoin OpenFOAM couvert, le comportement générique fourni, le ou les tutoriels utilisateurs, ainsi que la validation effectuée sous OpenFOAM 13. Une fonction ajoutée pour un seul cas doit être formulée comme une capacité réutilisable et non comme un contournement codé en dur.

## Évolutions enregistrées

| Référence | Module / fonction | Type | Besoin OpenFOAM 13 couvert | Tutoriels concernés | Validation / remarques |
|---|---|---|---|---|---|
| API-001 | `Solver.solver_name` | Extension d’attribut et sélection du solveur | Sélectionner explicitement le module `foamRun` attendu par un tutoriel OF13 | `fluid/pitzDaily`, cas `fluid`, `XiFluid`, `compressibleVoF` | Utilisé avec `foamRun -solver ...` lors des validations OF13. |
| API-002 | `Solver.setup_case` | Adaptation de préparation de cas | Initialiser un cas FoamPilot avant import ou écriture des dictionnaires de référence | Tous les runners intégrés | Utilisé systématiquement avant import des fichiers OF13. |
| API-003 | `Solver.run_simulation` | Extension d’exécution | Exécuter les solveurs transitoires via FoamPilot, en série ou en parallèle | Tous les cas transitoires intégrés | Validations jusqu’aux temps de fin indiqués dans la matrice. |
| API-004 | `Solver.run_parallel` | Extension d’exécution parallèle | Reproduire `runParallel foamRun` avec `mpirun`, nombre de processus et journal FoamPilot | `compressibleVoF/depthCharge3D` | Validation OF13 à 4 processus jusqu’à `End=0.5 s`, suivie de `reconstructPar`. |
| API-005 | `SystemDirectory.run_utility` | Nouvelle fonction générique | Exécuter une utilité OpenFOAM depuis FoamPilot avec journalisation uniforme | `XiFluid/engine2Valve2D` et autres pipelines multi-utilités | Utilisé pour les utilités de préparation et de maillage. |
| API-006 | `SystemDirectory.import_reference_file` | Nouvelle fonction générique | Importer sans perte un dictionnaire OF13 complet dans `system/`, y compris les dictionnaires complexes | Tous les cas utilisant des fichiers de référence | Permet de conserver exactement la syntaxe et les entrées OF13 originales. |
| API-007 | `SystemDirectory.update_dictionary_entries` | Nouvelle fonction générique | Modifier des entrées ciblées d’un dictionnaire importé | `XiFluid/moriyoshiHomogeneous`, `XiFluid/engine2Valve2D` | Utilisé pour les variantes et paramètres dépendant du cas. |
| API-008 | `SystemDirectory.rename_dictionary_entries` | Nouvelle fonction générique | Renommer des entrées ou patches sans réécriture destructive du dictionnaire | `XiFluid/engine2Valve2D` | Validé dans le pipeline des maillages temporels. |
| API-009 | `SystemDirectory.remove_dictionary_entries` | Nouvelle fonction générique | Supprimer proprement des entrées incompatibles d’un dictionnaire de référence | `XiFluid/engine2Valve2D` | Utilisé dans les transformations de configuration OF13. |
| API-010 | `SystemDirectory.import_reference_file` et import `.gz` associé | Extension d’import de ressources | Importer des ressources officielles, y compris les assets compressés | `compressibleVoF/ballValve`, `incompressibleFluid/motorBike` | Assets officiels OF13 importés par FoamPilot. |
| API-011 | `BlockMesher.import_reference_dict` | Nouvelle fonction générique | Importer un `blockMeshDict` complet ou un dictionnaire de ressource OF13 | `fluid/cavity`, `compressibleVoF`, `XiFluid` | Validé sur plusieurs familles de solveurs. |
| API-012 | `BlockMesher.import_reference_asset` | Nouvelle fonction générique | Importer une géométrie ou ressource de maillage officielle, avec support `.gz` | `compressibleVoF/ballValve`, `incompressibleFluid/motorBike` | Utilisé pour le tore officiel et la géométrie motorBike. |
| API-013 | `BlockMesher.copy_mesh` | Nouvelle fonction générique | Copier un maillage entre temps ou cas FoamPilot | `XiFluid/engine2Valve2D` | Utilisé dans la génération des 24 maillages temporels. |
| API-014 | `BlockMesher.write_mesh_times` | Nouvelle fonction générique | Écrire les temps de maillage nécessaires aux cas à maillages multiples | `XiFluid/engine2Valve2D` | Validé dans le pipeline OF13 jusqu’à `End=3600 CAD`. |
| API-015 | `BlockMesher.create_non_conformal_couples` | Nouvelle fonction générique | Créer les couples non conformes requis par les cas OF13 | `compressibleVoF/ballValve`, `XiFluid/engine2Valve2D` | Validé sans `FOAM FATAL` dans les cas concernés. |
| API-016 | `CaseFieldsManager.register_field` | Extension de gestion des champs | Enregistrer des champs spécifiques avec leurs valeurs initiales et propriétés | `fluid/pitzDaily`, `fluid/buoyantCavity` | Utilisé pour reproduire les champs et valeurs de référence. |
| API-017 | `CaseFieldsManager.custom_initial_values` | Extension de gestion des champs | Définir des valeurs initiales non couvertes par les valeurs par défaut | `fluid/pitzDaily` | Reproduction de la mise en données OF13. |
| API-018 | `CaseFieldsManager.import_reference_field` | Nouvelle fonction générique | Importer un champ OF13 complet, préserver ses conditions aux limites et renommer automatiquement les fichiers `.orig` | `XiFluid`, `compressibleVoF`, `fluid` | Utilisé notamment pour `alpha.water.orig`, `alpha.liquid.orig`, `T.orig`, `p.orig` et `p_rgh.orig`. |
| API-019 | `CaseFieldsManager.set_vof_primary_phase` | Nouvelle fonction générique | Définir la phase primaire d’un cas VoF et le champ de fraction associé | `compressibleVoF/ballValve` | Validé avec `alpha.vapour`. |
| API-020 | `Boundary.set_patch_type` et `Boundary.write_boundary_conditions` avec overrides | Extension de conditions aux limites | Reproduire fidèlement les types de patches et les valeurs OF13 | `fluid/pitzDaily` et plusieurs cas VoF | Validé sur les configurations multi-patches. |
| API-021 | `PhysicalPropertiesFile.configure_reference` | Nouvelle fonction générique | Conserver les blocs `thermoType` et `mixture` d’une référence OF13 | `fluid/cavity`, `fluid/pitzDaily` | Utilisé pour les modèles thermodynamiques compressibles. |
| API-022 | `PhasePhysicalPropertiesFile` : support complet de `thermo_type` et `mixture` | Extension thermophysique | Écrire les blocs complets `thermoType` / `mixture` des phases compressibles OF13 | `compressibleVoF/ballValve`, `compressibleVoF` | Validé avec les phases `air`, `water`, `vapour`, `liquid` et les cas multi-phases. |
| API-023 | `PhasePropertiesFile` : support d’un `sigma` dictionnaire | Extension VoF | Préserver les modèles de tension superficielle OF13, notamment `liquidProperties` et `constant` | `compressibleVoF/ballValve`, `compressibleVoF/damBreak` | Validé dans les cas avec tension superficielle scalaire ou dictionnaire. |
| API-024 | `ConstantDirectory.configure_vof` | Extension de configuration VoF | Configurer phases, propriétés de phase et fichiers OF13 VoF | `compressibleVoF/ballValve` et cas VoF antérieurs | Le comportement conserve `pRef` lorsque la référence l’exige. |
| API-025 | `ConstantDirectory.import_reference_file` | Nouvelle fonction générique | Importer sans perte les fichiers `constant/`, notamment `physicalProperties.<phase>`, `phaseProperties`, `momentumTransport` et `fvModels` | Tous les cas VoF intégrés | Validé sur les tutoriels VoF jusqu’à leurs temps de fin. |
| API-026 | `ConstantDirectory.remove_files` | Nouvelle fonction générique | Supprimer explicitement les fichiers par défaut générés par `setup_case` lorsqu’ils entrent en conflit avec une référence OF13 importée | `compressibleVoF/damBreak`, `depthCharge2D`, `depthCharge3D`, `sloshingTank2D`, `throttle` | Ajout motivé par la divergence thermique de `damBreak` lorsque `pRef`, `transportProperties` et `turbulenceProperties` coexistaient avec les dictionnaires OF13. Validation rétablie jusqu’à `End=1 s`. |
| API-027 | Import d’include de maillage par `SystemDirectory.import_reference_file` | Usage générique documenté | Rendre disponibles les fichiers inclus par une ressource `blockMeshDict`, par exemple `#include "sloshingTank"` | `compressibleVoF/sloshingTank2D` | Include partagé importé dans `system/`; validation jusqu’à `End=40 s`. |
| API-028 | `ChtSolver` : gestion régionale des champs et conditions aux limites | Extension multi-région | Générer les répertoires `fluid`, `metal`, `heater`, leurs champs et leurs fichiers système OF13 | `CHT/heatedDuct` | Validation multi-région OF13 jusqu’à `t=20 s`. |

## Convention pour les prochaines intégrations

Pour chaque nouveau tutoriel, le runner doit utiliser uniquement des appels FoamPilot. Si une capacité manque, elle doit être ajoutée au module générique approprié, puis enregistrée ici avec un nouvel identifiant `API-xxx`. La ligne du tutoriel dans la matrice doit reprendre le nom exact de la fonction et résumer la validation OpenFOAM 13 correspondante. Une entrée de matrice sans fonction nouvelle doit tout de même indiquer explicitement que seules les fonctions existantes ont été utilisées.

## État au 27 août 2026

Les tutoriels validés dans la tranche courante sont `compressibleMultiphaseVoF/damBreak4phaseLaminar`, `compressibleVoF/angledDuct`, `compressibleVoF/climbingRod`, `compressibleVoF/damBreak`, `compressibleVoF/depthCharge2D`, `compressibleVoF/depthCharge3D`, `compressibleVoF/sloshingTank2D` et `compressibleVoF/throttle`. Le cas `throttle` a atteint `End=0.001 s` sous OF13 avec `Solver.run_parallel` à quatre processus, sans `FOAM FATAL`.

| API-029 | `BaseSolver.import_reference_asset` | Nouvelle fonction générique | Copier un asset de référence non dictionnaire vers n’importe quel chemin relatif du cas, en conservant les permissions exécutables | `fluid/externalCoupledCavity` | Ajoutée pour importer le script `externalSolver` dans le cas sans appel shell dans le runner; validée sous OpenFOAM 13 jusqu’à `End=100 s`. |
| API-030 | `BaseSolver.run_command_async` et `BaseSolver.wait_command` | Nouvelles fonctions génériques | Démarrer une commande FoamPilot en arrière-plan, journaliser sa sortie et attendre son code de retour | `fluid/externalCoupledCavity` | Ajoutées pour coordonner `foamRun` et le processus externe de couplage; validées sous OpenFOAM 13 jusqu’à `End=100 s`. |

| API-031 | `SystemDirectory.replace_file_text` | Nouvelle fonction générique | Appliquer un remplacement textuel déterministe à un fichier généré par une utilité OpenFOAM lorsqu’un parser de dictionnaire n’est pas adapté | `fluid/nacaAirfoil` | Ajoutée pour reproduire la transformation OF13 `symmetry` → `empty` dans `constant/polyMesh/boundary` après `star3ToFoam`; validée dans le runner OF13, avec le cas `nacaAirfoil` accepté avec réserve pour dépassement du budget de calcul. |

| API-032 | `BaseSolver.run_parallel(..., force_decompose=False)` | Extension générique | Autoriser une nouvelle décomposition `decomposePar -force` entre deux phases parallèles successives sans modifier les dictionnaires physiques | `fluid/roomHeating` | Ajoutée pour reproduire les phases steady puis transient du tutoriel OF13; le comportement par défaut reste inchangé. Validée dans le runner OF13; `roomHeating` est accepté avec réserve car le transitoire n’a pas atteint son temps final dans le budget disponible. |

| API-033 | `CaseFieldsManager.import_reference_field` | Extension générique d’import | Créer automatiquement les répertoires parents lorsqu’un champ OF13 est importé sous un chemin imbriqué, par exemple `0/Lagrangian/cloud/*` | `fluid/stackPlume` | Ajoutée après l’échec d’import du champ `Lagrangian/cloud/d`; permet l’import récursif des champs Lagrangian sans opération de fichier directe dans le runner. Validée avec la chaîne de préparation OF13 de `stackPlume`. |

| API-034 | `CaseFieldsManager.import_reference_field` | Extension générique d’import | Décompresser automatiquement les champs de référence OpenFOAM terminés par `.gz` avant leur écriture sous leur nom actif | `incompressibleFluid/channel395` | Ajoutée après l’échec de `decomposePar` sur les champs compressés `p.gz`, `U.gz`, `k.gz`, `nuTilda.gz` et `nut.gz`; permet leur import correct et leur décomposition parallèle sous OpenFOAM 13. |
| API-035 | `BaseSolver.copy_case_tree` | Nouvelle fonction générique | Copier un fichier ou un sous-arbre entre cas FoamPilot pour les workflows de continuation et de mapping multi-cas | `incompressibleFluid/wingMotion2D_transient` | Ajoutée pour transférer `constant/polyMesh` du cas steady vers le cas transient et installer `pointDisplacement` après `mapFields`; validée sous OpenFOAM 13 avec quatre processeurs et mouvement sixDoF actif. |

| API-036 | `SystemDirectory.merge_reference_dictionary` | Nouvelle fonction générique | Reproduire la fusion de blocs de dictionnaires effectuée par `foamMergeCase` sans commande shell dans un runner | `incompressibleVoF/damBreak` et futures continuations OF13 | Fusionne les blocs de niveau supérieur d’un fichier différentiel avec le fichier cible en conservant les entrées existantes et en ajoutant les entrées de référence. Utilisée pour fusionner le bloc `RAS` de `momentumTransport.orig` et le bloc `divSchemes` de `fvSchemes.orig` tout en conservant les schémas alpha et vitesse du cas laminaire. Validée sous OpenFOAM 13 avec `damBreak` jusqu’à `End=1 s`, sans `FOAM FATAL`. |
| API-037 | `BaseSolver.run_command(..., environment=...)` | Extension générique d’exécution | Fusionner un environnement contrôlé avec l’environnement parent avant l’exécution d’une commande, afin de sélectionner explicitement `PATH`, `LD_LIBRARY_PATH`, MPI et bibliothèques ThirdParty d’une version OpenFOAM | `multiRegion/film/rivuletPanel` et futurs runners OF13 multi-région/parallèles | Ajoutée après la différence entre l’environnement hôte et le `bashrc` officiel OF13 : nécessaire pour charger Scotch/OpenMPI; validée avec `blockMesh`, `decomposePar`, `extrudeToRegionMesh -parallel`, `foamMultiRun -parallel` jusqu’à `End=5 s` et reconstruction finale sans `FOAM FATAL`. |
| API-038 | `BaseSolver.update_mesh_patch_types(...)` | Nouvelle fonction générique de maillage | Modifier de manière déterministe le type de patches dans `constant/polyMesh/boundary` après une utilité de maillage, en préservant le contenu des autres patches et sans commande shell de transformation | `shockFluid/biconic25-55Run35` et futurs workflows post-maillage | Ajoutée pour remplacer la transformation OF13 `sed` de `wedge1/wedge2` après `collapseEdges`; méthode limitée aux patches explicitement demandés et conçue pour les conversions `patch` → `wedge`, `symmetry`, etc. Validée dans la préparation du runner `biconic25-55Run35`, dont le calcul est accepté avec réserves de temps et d’avertissement de planéité. |
| API-039 | `BaseSolver.merge_mesh_points(...)` | Nouvelle fonction générique de maillage | Fusionner une liste temporaire de points produite par une utilité OF13 avec le fichier `constant/polyMesh/points`, en préservant l’en-tête OpenFOAM sans commande shell de concaténation | `shockFluid/biconic25-55Run35` et futurs workflows `datToFoam` | Ajoutée pour remplacer la séquence OF13 de découpage des 17 premières lignes, concaténation de `points.tmp` et suppression du fichier temporaire; validée dans la préparation du runner `biconic25-55Run35`, dont le calcul est accepté avec réserves de temps et d’avertissement de planéité. |
| API-040 | `BaseSolver.check_solver_module_exists` | Extension générique de détection | Reconnaître les modules FoamRun OF13 installés comme bibliothèques partagées `lib<solver>.so` ou `.dylib`, en plus d’un fichier sans préfixe | `incompressibleVoF/damBreakPorousBaffle` et futurs solveurs FoamRun | Corrige le rejet préalable du module OF13 `libincompressibleVoF.so` alors que `foamRun` pouvait le charger via `LD_LIBRARY_PATH`; validée avec `damBreakPorousBaffle` jusqu’à `Time=1 s` sans `FOAM FATAL`. |
| API-041 | `BaseSolver.import_reference_asset` | Extension générique d’import | Décompresser automatiquement un asset de référence `.gz` lorsque la destination demandée ne porte pas `.gz`, tout en conservant la copie directe pour les destinations compressées | `mesh/spiralPipe` et futurs assets géométriques compressés | Ajoutée pour matérialiser `curve.obj` à partir de `curve.obj.gz` conformément à `extrudeMeshDict`; permet de gérer les ressources gzip sans commande shell dans les runners. Validée sous OpenFOAM 13 avec `blockMesh` puis `extrudeMesh`. |
| API-042 | `OpenFOAMEnvironment` | Nouvelle façade générique d’environnement | Résoudre une installation OpenFOAM à partir de `etc/bashrc`, produire un environnement isolé et appliquer des overrides sans modifier le shell parent | Tous les runners OF13 et futurs cas multi-version | Centralise la résolution de `PATH`, `LD_LIBRARY_PATH`, variables `WM_*`, bibliothèques OpenFOAM et paramètres MPI. Testée avec une erreur explicite lorsque le `bashrc` est absent. |
| API-043 | `RunWorkflow` | Retirée / supersédée | La façade ajoutait une couche au-dessus de `BaseSolver` sans usage démontré dans les runners | Aucun | Retirée après audit: les primitives `BaseSolver.run_command`, `run_simulation` et `run_parallel` restent l’API d’exécution actuelle. Une façade ne sera réintroduite qu’après migration concrète de runners et preuve de valeur utilisateur. |

### API-044 — `SnappyMesher.mergeTolerance`

Ajout d’un paramètre générique `mergeTolerance` à `SnappyMesher` et écriture systématique de cette entrée dans `snappyHexMeshDict`. OpenFOAM Foundation 13 l’exige au niveau racine du dictionnaire. Cette évolution a été découverte et validée lors de la réécriture de `02_simpleCar_turbulent`; elle est indépendante du tutoriel et s’applique à tout maillage `snappyHexMesh`.

Validation: `snappyHexMesh` termine, tous les fichiers principaux et champs sont générés par FoamPilot, et le calcul turbulent atteint `Time=300 s`/`End` sans `FOAM FATAL`.

### Statut de réécriture

Les runners `01_cavity_laminar` et `02_simpleCar_turbulent` sont les premières réécritures complètes validées: ils ne copient pas de dictionnaires ni de champs de référence. Les autres runners restent classés selon la matrice complète jusqu’à leur migration effective.

### API-045 — `BlockMesher.definitions`
Ajout d’un registre déclaratif de définitions OpenFOAM brutes dans `BlockMesher`, écrit entre `vertices` et `blocks`. Cette capacité permet de représenter les listes nommées et variables de dictionnaire utilisées par `simpleGrading` et `edgeGrading`, sans importer un `blockMeshDict` de référence. L’écriture du dictionnaire a été structurée pour préserver l’ordre et la syntaxe OF13 des sections `vertices`, définitions, `blocks`, `edges`, `defaultPatch`, `boundary` et `mergePatchPairs`.
Tutoriel concerné: `05_scalarTransport`; validation OF13: maillage pitzDaily généré par FoamPilot, champs `U`, `p`, `T`, `k`, `epsilon`, `nut` générés et conditions turbulence valides, calcul `functions` lancé sans `FOAM FATAL` jusqu’à la fin du cas.
### Statut de réécriture
`05_scalarTransport` rejoint les réécritures complètes validées: aucun `import_reference_case`, `import_reference_field`, `copy_reference_fields` ou ressource de maillage de référence n’est utilisé dans son runner.

### API-046 — `SnappyMesher.add_searchable_box` et raffinement volumique scalaire
Ajout d’une primitive générique `add_searchable_box(name, minimum, maximum)` pour déclarer une géométrie `searchableBox` directement dans `snappyHexMeshDict`. Extension de `add_refinement_region` et de son writer pour accepter soit une paire `levels (min max)`, soit un niveau scalaire `level n`, conformément aux deux syntaxes OF13. Cette évolution remplace l’import d’un `snappyHexMeshDict` complet dans les cas d’aérodynamique urbaine.
Tutoriel concerné: `06_buildingAero`; validation OF13: background mesh, extraction de features, `snappyHexMesh`, propriétés, champs `U/p/k/epsilon/nut` et calcul incompressible jusqu’à `Time=500 s` puis `End`, sans `FOAM FATAL`.

### API-047 — géométrie et sommets avancés déclaratifs dans `BlockMesher`
`BlockMesher` accepte désormais `geometry` pour écrire des objets de géométrie nommés (`sphere`, `triSurface`, etc.), des sommets représentés par des fragments OpenFOAM bruts (`name ...`, `project ...`) et des faces de patch brutes. Les coordonnées numériques restent supportées. Cette capacité est destinée aux topologies avancées comme ballValve, avec arcs et projections, sans importer un `blockMeshDict` de référence.

### API-048 — faces top-level avancées dans `BlockMesher`
`BlockMesher` accepte désormais `faces`, une liste de faces OpenFOAM brutes écrites dans la section top-level `faces`. Cela permet de séparer correctement les faces `project` de la topologie des patches `boundary`, conformément à la grammaire OF13 des maillages non conformes.

### API-049 — sous-dictionnaire `potentialFlow` dans `FvSolutionFile`
`FvSolutionFile` expose désormais `potentialFlow` dans son constructeur, sa sérialisation et `from_dict`. Les cas utilisant `potentialFoam` peuvent déclarer `nNonOrthogonalCorrectors` sans importer `fvSolution`.


## API-048 — Directives `#include` déclaratives dans `OpenFOAMDictAddFile`

Le writer accepte l’attribut `includes`, sérialisé en directives natives `#include`, pour composer des dictionnaires OpenFOAM sans recopier leurs fichiers sources. Utilisé pour les mélanges thermodynamiques XiFluid.

## API-049 — Géométrie avancée et faces top-level dans `BlockMesher`

`BlockMesher` sait déclarer les objets de géométrie, les sommets nommés/projetés et les faces top-level nécessaires aux maillages ballValve et aux géométries multi-blocs avancées.

## API-050 — Dimensions et champs thermiques XiFluid

`OpenFOAMFile.FIELD_DIMENSIONS` reconnaît `Tu` comme température. La génération compressible conserve la pression dynamique et permet l’initialisation déclarative des champs XiFluid (`T`, `Tu`, `Xi`, `b`, `egr`, `ft`, `fu`, `k`, `epsilon`, `omega`, `p`) sur les patches produits par baffles et couples NCC.

## API-051 — Configuration XiFluid avancée

Les writers déclaratifs existants sont utilisés conjointement pour produire les propriétés thermophysiques incluses, `combustionProperties`, les solveurs `fvSolution` (`rhoFinal`, `epsilonFinal`, `pFinal`, `MeshPhi`, PIMPLE) et les schémas de convection XiFluid. Validation OF13: #11 atteint `Time=5000 s` puis `End`; #10 atteint `End` après 1412 s.


## API-052 — Génération déclarative de mélanges homogènes XiFluid

Les runners peuvent désormais construire sans import les propriétés `homogeneousMixture` (`thermoType`, `reactants`, `products`, coefficients JANAF, transport `mu/Pr`), les modèles de combustion Gulder, les schémas XiFluid composés et les champs initiaux de combustion. Validation OF13: les variantes propane et hydrogène de `12_XiFluid_moriyoshiHomogeneous` atteignent chacune `Time=0,015 s` puis `End`.
