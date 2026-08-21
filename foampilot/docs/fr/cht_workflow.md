# Conjugate heat transfer: data setup, execution, and validation

Le transfert de chaleur conjugé (CHT) couple la conduction thermique dans les solides au transport de chaleur dans une ou plusieurs régions fluides. FoamPilot fournit des objets Python pour les définitions de régions, les dictionnaires thermophysiques, les champs de température, les interfaces, les dictionnaires de contrôle, l'exécution et le post-traitement thermique. OpenFOAM reste responsable de la résolution des équations volumes finis couplées.

## 1. Physical problem

Un cas CHT contient au moins une région fluide et une région solide. Dans le fluide, le solveur résout la quantité de mouvement, la continuité et l'énergie. Dans le solide, il résout la conduction thermique ; il n'existe pas de champ de vitesse de fluide dans le solide.

Pour un fluide compressible, l'équation d'énergie est écrite dans une forme thermodynamiquement consistante en utilisant la formulation par enthalpie ou énergie interne sélectionnée. Une forme simplifiée en température est :

$$
\rho c_p\left(\frac{\partial T}{\partial t}+\mathbf{U}\cdot\nabla T\right)
=\nabla\cdot(k_f\nabla T)+S_T,
$$

où $\rho$ est la densité, $c_p$ la capacité calorifique, $k_f$ la conductivité du fluide, et $S_T$ représente les sources. Dans un solide au repos :

$$
\rho_s c_{p,s}\frac{\partial T_s}{\partial t}
=\nabla\cdot(k_s\nabla T_s)+S_s.
$$

À une interface fluide-solide parfaite :

$$
T_f=T_s,
$$

et

$$
-k_f\nabla T_f\cdot\mathbf{n}
=-k_s\nabla T_s\cdot\mathbf{n}.
$$

La première condition exprime la continuité de la température. La seconde exprime la continuité du flux de chaleur. Si la résistance de contact, le rayonnement, la rugosité, ou un revêtement mince sont physiquement importants, le modèle d'interface doit être modifié plutôt que masqué par un ajustement de maillage.

## 2. Data model in FoamPilot

L'API CHT est organisée autour de quatre concepts :

| Objet | Rôle |
| --- | --- |
| `FluidRegion` | Température et vitesse initiales, modèle de turbulence, modèle thermo, équation d'état, mélange et modèle de transport pour une région fluide. |
| `SolidRegion` | Température initiale et propriétés matérielles solides pour une région conductrice. |
| `CoupledInterface` | Appariement des patches fluide et solide et dictionnaire d'interface. |
| `ChtSolver` | Chemin du cas, exécutable du solveur, régions, interfaces, mappage solveur-région, configuration et exécution série/parallèle. |

Le patron de configuration minimal est :

```python
from foampilot.cht import (
    ChtSolver,
    FluidRegion,
    SolidRegion,
    CoupledInterface,
)

fluid = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

solid = SolidRegion(
    name="solid",
    temperature=350.0,
)

interface = CoupledInterface(
    name="fluid_to_solid",
    fluid_region="fluid",
    solid_region="solid",
)

solver = ChtSolver(
    case_path="case",
    solver_name="chtMultiRegionFoam",
    regions=[fluid, solid],
    interfaces=[interface],
)
solver.setup_case()
solver.run_simulation(nb_proc=1)
```

Les paramètres matériels concrets doivent être renseignés pour le problème ciblé. N'utilisez pas les valeurs par défaut comme données matérielles validées.

## 3. Case directory and region data

Un cas multi-région n'est pas un cas mono-région avec un champ de température supplémentaire. La disposition des répertoires porte une signification physique :

```text
case/
├── 0/
│   ├── fluid/
│   │   ├── T
│   │   ├── U
│   │   ├── p
│   │   ├── p_rgh
│   │   └── turbulence fields
│   └── solid/
│       └── T
├── constant/
│   ├── fluid/
│   │   ├── thermophysicalProperties
│   │   └── turbulenceProperties
│   ├── solid/
│   │   ├── thermophysicalProperties
│   │   └── transportProperties
│   └── regionInterfaces/
│       └── fluid_to_solid.dict
└── system/
    ├── controlDict
    ├── regionProperties or region solver mapping
    ├── fvSchemes
    ├── fvSolution
    └── createZonesDict
```

Les noms précis des dictionnaires varient selon la version d'OpenFOAM et la famille de solveurs. Le tutoriel du dépôt `09_CHT_heatedDuct` est la référence pour la version visée. Inspectez toujours les dictionnaires générés car `foamSetupCHT`, `splitMeshRegions` et les conventions spécifiques aux régions diffèrent entre les versions.

## 4. Geometry and region definition

Le tutoriel heated-duct utilise un maillage de fond et une définition en cell-zone pour distinguer le fluide du solide chauffé. La séquence conceptuelle est :

```text
blockMesh
→ createZones
→ splitMeshRegions -cellZones
→ create region dictionaries and fields
→ run chtMultiRegionFoam
```

La définition des cell-zones doit couvrir le volume solide prévu sans trous ni recouvrements. Un split de région raté peut créer un cas qui semble contenir un répertoire solid mais ne représente pas le domaine physique voulu.

Validez la topologie des régions avec :

```bash
checkMesh -case case -region fluid
checkMesh -case case -region solid
splitMeshRegions -cellZones -overwrite -case case
```

Utilisez la commande appropriée à la distribution OpenFOAM installée et n'exécutez pas une opération destructive `-overwrite` sur la seule copie d'un cas source.

## 5. Fluid-region data

Les données du fluide doivent spécifier suffisamment d'informations pour fermer les équations de quantité de mouvement et d'énergie :

| Donnée | Signification | Choix typiques |
| --- | --- | --- |
| Density law | Relation entre densité, pression et température. | Perfect gas, constant density, Boussinesq where supported. |
| Heat capacity | Stockage d'énergie. | Constant or temperature-dependent $c_p$. |
| Conductivity | Diffusion moléculaire de chaleur. | Constant or temperature-dependent $k_f$. |
| Viscosity | Diffusion de quantité de mouvement. | Constant or temperature-dependent dynamic viscosity. |
| Equation of state | Relation pression-densité. | Ideal gas or incompressible approximation. |
| Turbulence | Fermeture pour l'écoulement non résolu. | Laminar, $k$–$\epsilon$, $k$–$\omega$ SST, or version-specific RAS model. |
| Initial velocity | Champ d'écoulement initial. | Zero or prescribed bulk velocity. |
| Initial temperature | Champ thermique initial. | Inlet temperature, wall temperature, or a uniform estimate. |

Pour un conduit d'air, la région fluide est couramment traitée comme compressible en CHT parce que la température affecte la densité et l'énergie. Si la plage de température est faible et que le solveur visé le supporte, une formulation thermique incompressible peut être plus appropriée, mais cela doit être physiquement justifié.

## 6. Solid-region data

La région solide requiert au minimum :

- la densité $\rho_s$ ;
- la capacité calorifique $c_{p,s}$ ;
- la conductivité thermique $k_s$ ;
- la température initiale ;
- les conditions limites thermiques externes et d'interface.

Pour une paroi en cuivre, une conductivité élevée produit un gradient de température relativement faible à travers le solide comparé à une couche isolante à faible conductivité. Cela ne signifie pas que le solide peut être supprimé : son épaisseur et sa résistance thermique déterminent toujours le flux de chaleur.

Le modèle solide peut être étendu avec des propriétés dépendantes de la température, du rayonnement, de la génération de chaleur, ou une conductivité anisotrope si le solveur et les dictionnaires générés les supportent.

## 7. Boundary-condition data

Les conditions aux limites doivent être définies séparément pour chaque région. Conditions fluides communes :

| Frontière | Champs du fluide |
| --- | --- |
| Inlet | Velocity, temperature, pressure, turbulence quantities. |
| Outlet | Pressure and compatible velocity/temperature outflow conditions. |
| Fluid-solid interface | Coupled temperature and heat-flux condition. |
| Symmetry | `symmetryPlane` for all compatible fields. |
| Adiabatic wall | Zero heat flux, with no-slip or the selected wall velocity treatment. |

Conditions solides communes :

| Frontière | Champ solide |
| --- | --- |
| Coupled interface | Temperature and heat flux coupled to the fluid region. |
| Fixed-temperature wall | Prescribed `T`. |
| Heat-flux wall | Prescribed normal heat flux. |
| Adiabatic wall | Zero gradient or solver-specific insulated condition. |
| Radiation wall | Radiation-coupled condition when radiation is included. |

FoamPilot expose des usines (`factories`) telles que `get_coupled_temperature_bc`, `get_fixed_temperature_bc`, `get_heat_flux_bc`, `get_inlet_outlet_bc`, et `get_symmetry_bc`. La condition générée doit néanmoins être vérifiée par rapport à la version d'OpenFOAM installée.

## 8. Solver execution

Le solveur CHT est un exécutable multi-région autonome dans le flux de travail OpenFOAM classique. Le chemin sériel de FoamPilot lance le solveur avec le chemin du cas. Le chemin parallèle décompose toutes les régions, exécute MPI, et reconstruit toutes les régions.

```python
solver.run_simulation(
    nb_proc=1,
    log_filename="log.chtMultiRegionFoam",
)
```

Pour l'exécution parallèle :

```python
solver.run_simulation(nb_proc=8)
```

Avant un calcul parallèle, confirmez que MPI, `decomposePar`, et `reconstructPar` sont compatibles avec la compilation d'OpenFOAM. Sauvegardez la référence sérielle car elle est nécessaire pour distinguer un problème de décomposition d'un problème physique ou numérique.

## 9. Convergence et conservation

Une exécution CHT ne doit être considérée comme convergée que lorsque plusieurs conditions sont satisfaites :

1. les résidus décroissent jusqu'aux tolérances sélectionnées ;
2. les températures et flux de chaleur des régions atteignent des tendances stables ;
3. la continuité de la température à l'interface est acceptable ;
4. la chaleur totale entrant et sortant du système couplé est équilibrée ;
5. le résultat est insensible à une réduction raisonnable du pas de temps ou à une augmentation des itérations ;
6. la grandeur d'ingénierie d'intérêt est stable.

Les helpers `foampilot.cht.postprocess` incluent des calculs pour le flux de chaleur de région, le flux de chaleur d'interface, le nombre de Nusselt, l'épaisseur de la couche limite thermique, le coefficient de transfert de chaleur, le bilan thermique total, les contours de température, et la résistance thermique.

## 10. Heated-duct example data

Le tutoriel du dépôt utilise une région fluide initialement proche de 300 K et une région solide chauffée proche de 350 K. Ses données d'entrée incluent un `block_mesh.json`, un constructeur de cas Python, un script de post-traitement, des fichiers de champs spécifiques aux régions, et des dictionnaires matériaux. Les sorties attendues incluent des statistiques de température par région, des profils de température du fluide, des profils combinés, et des contours de température fluide/solide.

Les valeurs du tutoriel sont des données de démonstration. Pour un conduit réel, remplacez-les par des valeurs mesurées ou de conception pour :

- le débit massique ou la vitesse d'entrée ;
- la température et la pression d'entrée ;
- l'épaisseur et le matériau de la paroi ;
- la source de chaleur externe ou la température paroi ;
- la composition du fluide et les corrélations de propriétés ;
- la résistance de contact, si applicable ;
- une longueur de domaine suffisante pour la région de développement thermique.

## 11. Heat-transfer interpretation

Le coefficient convectif local est souvent défini par :

$$
 h=\frac{q''}{T_w-T_b},
$$

où $q''$ est le flux de chaleur à la paroi, $T_w$ est la température de paroi, et $T_b$ est une température de fluide en masse adaptée. Le nombre de Nusselt est :

$$
 Nu=\frac{hL}{k_f}.
$$

La longueur caractéristique $L$ doit être indiquée : diamètre hydraulique, hauteur du conduit, distance locale depuis l'entrée, ou une autre échelle physiquement significative. Un nombre de Nusselt sans son échelle de longueur et sa condition aux limites est incomplet.

## 12. Common CHT failure modes

| Symptôme | Cause probable |
| --- | --- |
| Missing region fields | Region directories were not created or the setup was interrupted. |
| Solver cannot find a material property | Region-specific `thermophysicalProperties` is incomplete or named differently for the OpenFOAM release. |
| Interface temperature jumps | Wrong patch pairing, non-conformal mapping, or incompatible interface conditions. |
| Heat balance is not closed | Insufficient convergence, incorrect flux sign, missing boundary heat loss, or an unintended source. |
| Fluid temperature is unstable | Time step, energy relaxation, thermo model, or boundary condition is inconsistent. |
| Solid temperature is uniform when it should not be | Conductivity, thickness, source, or region geometry is wrong; check the generated solid mesh. |
| Parallel case differs from serial case | Decomposition, reconstruction, processor boundary treatment, or insufficient convergence. |

## Références

[1]: https://doc.openfoam.com/2306/tools/processing/solvers/rtm/heat-transfer/chtMultiRegionFoam/ "Documentation OpenFOAM: chtMultiRegionFoam"

[2]: https://openfoamwiki.net/index.php/Getting_started_with_chtMultiRegionSimpleFoam_-_planeWall2D "OpenFOAM Wiki: démarrage avec chtMultiRegionSimpleFoam - planeWall2D"
