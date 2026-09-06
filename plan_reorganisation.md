Voici le **Plan d'action architectural révisé et définitif** de FoamPilot v3. 

Il intègre toutes les leçons apprises, les corrections (notamment l'ajout crucial de `core/geometry/`), et la philosophie fondamentale selon laquelle **les workflows doivent rester des scripts Python impératifs, lisibles et modifiables ligne par ligne**, et non des boîtes noires monolithiques.

---

# FoamPilot v3 — Plan d'action architectural définitif

### 1. Principes Directeurs (La "Constitution" v3)

L'architecture repose sur 3 règles absolues et non négociables :

1. **L'agnosticisme de version** : Une version OpenFOAM (Foundation 13, 14, OpenFOAM.com) n'est ni une capability, ni un workflow, ni une extension. C'est une **métadonnée du backend d'exécution**. Elle ne doit jamais structurer l'arborescence du code métier.
2. **Les Workflows sont des scripts de composition, pas des classes** : Un workflow (ex: `muffler_simulation.py`) est un script Python impératif, lisible, qui importe et assemble des *capabilities* génériques. Il ne doit jamais être encapsulé dans une classe `MufflerSolver` opaque qui cache la logique et empêche la personnalisation ligne par ligne.
3. **Flux de dépendance strict** : 
   `Workflows` → `Capabilities (core/)` → `Backend (openfoam/)` → `Bibliothèques externes`.
   *Interdit absolu* : `core/` ne doit jamais importer `workflows/`, `examples/`, ou des modules spécifiques à une version d'OpenFOAM.

---

### 2. Architecture Cible

```text
foampilot/
│
├── src/foampilot/
│   ├── core/                          # CAPABILITIES (Génériques, réutilisables, agnostiques)
│   │   ├── case/                      # Création cas, arborescence, validation chemins
│   │   ├── geometry/                  # CAD, surfaces, topologie, STL, OCC, VMTK (Générique)
│   │   ├── meshing/                   # Gmsh, blockMesh, snappyHexMesh, qualité, adaptation
│   │   ├── boundaries/                # Gestion générique des conditions aux limites
│   │   ├── dictionaries/              # FoamDict, BoundaryDict, DictionaryWriter
│   │   ├── postprocessing/            # foamToVTK, résidus, visualisation PyVista, stats
│   │   ├── reporting/                 # Génération de rapports (LaTeX, JSON, CSV)
│   │   ├── units/                     # ValueWithUnit, Pint (gestion des unités)
│   │   └── physics/                   # FluidMechanics, propriétés des fluides
│   │
│   ├── openfoam/                      # BACKEND D'EXÉCUTION (Interface FoamPilot ↔ OpenFOAM)
│   │   ├── backend/                   # Détection version/distribution (métadonnées runtime)
│   │   ├── environment/               # Sourcing bashrc, variables d'environnement portables
│   │   ├── runner/                    # Exécution brute (subprocess, mpirun, decomposePar)
│   │   ├── solvers/                   # Registry et Configs spécialisées (Incompressible, VoF, CHT...)
│   │   └── extensions/                # Couplages C++ (FSI, Cantera, Yade)
│   │
│   └── workflows/                     # SCRIPTS DE COMPOSITION (Impératifs, lisibles, modifiables)
│       ├── medical/                   # Ex: aorta_simulation.py (assemble core/geometry, core/meshing...)
│       ├── marine/                    # Ex: muffler_simulation.py, actuation_disk.py
│       ├── urban/                     # Ex: neighborhood_cfd.py
│       └── energy/                    # Ex: heat_exchanger.py
│
├── patches/                           # CORRECTIFS SPÉCIFIQUES AUX VERSIONS
│   └── openfoam/
│       └── foundation/
│           └── 13/                    # Uniquement les patches/wallDist spécifiques à la v13
│
├── examples/                          # Démonstrations minimalistes (type "Hello World")
├── tutorials/                         # Guides pas-à-pas pédagogiques
├── validation/                        # Benchmarks scientifiques et numériques
├── tests/
│   ├── unit/                          # Tests des capabilities isolées (ex: test FoamDict)
│   ├── integration/                   # Tests de bout en bout (Geometry → Mesh → Solve)
│   └── validation/                    # Comparaison des résultats avec données de référence
│
├── docs/architecture/                 # Documentation de la philosophie v3
├── tools/audit/
│   └── check_architecture.py          # Script CI : Interdit les imports core → workflow
│
├── pyproject.toml
└── CONTRIBUTING.md
```

---

### 3. État des Lieux (Progression)

| Module | État | Détails |
| :--- | :---: | :--- |
| **core/dictionaries/** | ✅ **FAIT** | `FoamDict`, `BoundaryDict`, `CaseLayout` opérationnels. |
| **core/meshing/** | ✅ **FAIT** | BlockMesh, Gmsh, Snappy, Quality, Adaptation extraits. |
| **core/postprocessing/** | ✅ **FAIT** | 15 modules migrés (VTK, résidus, stats). |
| **core/reporting/** | ✅ **FAIT** | 7 modules migrés (LaTeX, rapports). |
| **core/units/** & **physics/** | ✅ **FAIT** | `ValueWithUnit` et `FluidMechanics` extraits de `utilities/`. |
| **openfoam/solvers/** | ✅ **FAIT** | Registry + 5 configs spécialisées pilotant `fvSchemes`/`fvSolution`. |
| **core/geometry/** | ✅ **FAIT** | `cad/`, `topology/`, `surfaces/` opérationnels. VMTK dans topology. |
| **Dismantle `openfoam13/`** | ✅ **FAIT** (commit b084745) | Shims avec DeprecationWarning. Patches dans `patches/`. |
| **CI Architecturale** | ✅ **FAIT** (commit b0fbd67) | `tools/audit/check_architecture.py` et `.github/workflows/ci.yml`. |
| **Réorganisation workflows/** | ✅ **FAIT** (commit 86fd562) | `medical/`, `urban/`, `marine/`, `energy/`, `wind/`. Extensions redirect. |
| **Documentation** | ✅ **FAIT** (commit 4762394) | `docs/architecture/overview.md`. |

---

### 4. Le Cœur de la Refonte : Capability vs Workflow

Pour dissiper toute crainte : **votre script `muffler_simulation.py` EST le modèle parfait d'un workflow v3.** Il ne sera pas transformé en classe obscure. Il sera simplement rangé dans `workflows/marine/` et ses imports pointeront vers le nouveau `core/`.

**Exemple de ce à quoi ressemblera le workflow dans v3 :**
```python
# workflows/marine/muffler_simulation.py
from foampilot.core.physics import FluidMechanics
from foampilot.core.units import ValueWithUnit
from foampilot.solver import Solver
from foampilot import Meshing
import classy_blocks as cb

# 1. Physique (Capability)
fluid = FluidMechanics('Water', temperature=ValueWithUnit(293.15, "K"), pressure=ValueWithUnit(101325, "Pa"))

# 2. Solveur (Capability - API fluide préservée à 100%)
solver = Solver(current_path)
solver.constant.transportProperties.nu = fluid.get_fluid_properties()['kinematic_viscosity']

# 3. Géométrie (L'utilisateur garde le contrôle total, ligne par ligne)
shapes = [cb.Cylinder([0, 0, 0], [0.3, 0, 0], [0, 0.05, 0])]
shapes[-1].chop_axial(start_size=0.015)
# ... l'utilisateur modifie, ajoute, ou supprime des étapes librement ...

# 4. Exécution (Capability)
solver.boundary.apply_condition_with_wildcard(pattern="inlet", condition_type="velocityInlet", ...)
solver.run_simulation()
```
*Gain v3* : Si demain on améliore le calcul de la viscosité dans `core/physics/`, ce script en bénéficie automatiquement, sans avoir à être modifié.

---

### 5. Règles de Migration (Le "Comment")

1. **Pattern Strangler Fig (Shims)** : Lors du déplacement d'un module, l'ancien fichier devient un simple wrapper qui importe le nouveau et lève un `DeprecationWarning`. Aucune rupture de compatibilité pendant la transition.
2. **Séparation Géométrie** : 
   - *Capability* (va dans `core/geometry/`) : "Extraire une centerline", "Nettoyer un STL", "Convertir OCC en maillage".
   - *Workflow* (reste dans `workflows/medical/`) : "Pipeline complet de reconstruction d'aorte avec pré-traitement patient spécifique".
3. **Dismantle de `openfoam13/`** :
   - Correctif de bug v13 → `patches/openfoam/foundation/13/`
   - Outil générique → `core/` ou `openfoam/backend/`
   - Modèle physique spécifique (ex: Windkessel) → `workflows/medical/` ou `extensions/`

---

### 6. Roadmap des PRs Restantes (Ordre d'exécution)

Pour achever la v3 de manière sûre, reviewable et réversible :

- **PR T : Finalisation du nettoyage `utilities/`**  
  Déplacer `residuals.py` vers `core/postprocessing/` et `latex_pdf.py` vers `core/reporting/`. Mettre à jour les shims.

- **PR U : Création et peuplement de `core/geometry/`** *(La pièce manquante)*  
  Créer `core/geometry/{cad, surfaces, topology, vmtk}`. Extraire les fonctions génériques de l'ancien dossier `geometry/`. Mettre en place les shims.

- **PR V : Dismantle de `openfoam13/` et `model_addon/`** *(La Règle d'Or)*  
  Analyser fichier par fichier. Vider ces dossiers en redistribuant leur contenu selon les règles de la section 5. Supprimer les dossiers une fois vides.

- **PR W : Verrouillage Architectural (CI)**  
  Créer `tools/audit/check_architecture.py`. L'ajouter au workflow GitHub Actions (`.github/workflows/ci.yml`) pour faire échouer toute PR qui tente un import interdit (ex: `core` → `workflows`).

- **PR X : Réorganisation des Workflows**  
  Créer le dossier racine `workflows/`. Y déplacer les dossiers `medical/`, `marine/`, `urban/`, `cht/`, etc. Mettre à jour leurs imports pour qu'ils pointent vers `core/`.

- **PR Y : Documentation et Nettoyage Final**  
  Rédiger `docs/architecture/overview.md`. Après une période de grâce, supprimer définitivement les anciens fichiers shims devenus obsolètes et l'ancienne classe `base/openFOAMFile.py`.

---

### 7. Ce qui est Explicitement Rejeté (Anti-Patterns)

❌ **`extensions/openfoam/foundation13/`** : La version est une métadonnée, pas une arborescence.  
❌ **`openfoam/adapters/foundation13.py` comme API centrale** : La détection de version est interne au backend.  
❌ **Workflows sous forme de classes `Manager` ou `Runner` monolithiques** : Cela tue la flexibilité et le débogage. Les workflows restent des scripts de composition explicite.

---

### Conclusion

Ce plan garantit que **FoamPilot v3 sera robuste, maintenable et évolutif**, tout en préservant l'élégance et le contrôle total que vous avez mis en place dans vos scripts de simulation. 

Si ce plan définitif vous convient, la prochaine action logique et immédiate est de lancer la **PR U** pour enfin créer et peupler `core/geometry/`, complétant ainsi le socle des capabilities avant de s'attaquer au nettoyage de `openfoam13/`.