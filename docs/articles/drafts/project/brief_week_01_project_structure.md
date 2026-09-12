# Brief — Article projet : "La structure de foampilot : architecture d'un wrapper OpenFOAM"

**Objectif** : Expliquer comment foampilot est structuré en interne, comment les modules interagissent, et pourquoi cette architecture rend le projet maintenable et extensible.
**Angle** : Architecture logicielle, retour d'expérience, open-source.

---

## Structure proposée

### Introduction — "Ce n'est pas qu'un script, c'est un projet"

Storytelling : Quand j'ai commencé foampilot, c'était un script de 200 lignes. Aujourd'hui, c'est une plateforme avec plus de 30 modules, une documentation complète, des tests, et une communauté naissante. Voici comment j'ai structuré le projet pour qu'il survive à son créateur.

**Promesse** : Dans cet article, je vous emmène dans les coulisses de l'architecture de foampilot : comment les modules interagissent, pourquoi j'ai choisi une structure en couches, et comment vous pouvez contribuer.

### Section 1 — Vue d'ensemble de l'architecture

Présenter le schéma d'architecture :

```
foampilot/
├── base/               # Abstractions de base (OpenFOAMFile, Meshing)
├── solver/             # Gestion des solveurs OpenFOAM
├── boundaries/         # Conditions aux limites
├── mesh/               # Maillage (blockMesh, Gmsh, snappyHexMesh)
├── constant/           # Fichiers constants (transport, turbulence)
├── system/             # Fichiers système (controlDict, fvSchemes)
├── postprocess/        # Post-processing (PyVista, readers)
├── report/             # Génération de rapports (LaTeX, Typst)
├── cht/                # Modules spécifiques CHT
├── utilities/          # Utilitaires (fluides, unités, fonctions)
└── commons/            # Code partagé
```

### Section 2 — Les couches d'abstraction

#### Couche 1 : OpenFOAMFile (base)

```python
class OpenFOAMFile:
    # Traducteur Python → OpenFOAM
    # Gestion des unités, headers, syntaxe
```

#### Couche 2 : Solver / Meshing / Report (orchestrateurs)

```python
class Solver:
    # Orchestrateur principal
    # Propriétés Python pour configuration déclarative

class Meshing:
    # Factory pour blockMesh / Gmsh / snappyHexMesh

class CFDReportGenerator:
    # Agrégation des résultats → rapports
```

#### Couche 3 : Spécialisations (boundaries, cht, postprocess)

```python
class Boundary:
    # Conditions aux limites par turbulence model

class ChtSolver:
    # Spécialisation CHT multi-région

class FoamPostProcessing:
    # Extraction de quantités physiques
```

### Section 3 — Les principes de design

#### 3.1 Séparation des responsabilités

Chaque module a un rôle clair :
- `base/` : abstractions génériques
- `solver/` : logique de configuration et d'exécution
- `mesh/` : génération et export de maillages
- `report/` : production de documents

#### 3.2 Dependency inversion

Les modules supérieurs ne dépendent pas des modules inférieurs, mais d'abstractions :

```python
# Solver dépend de abstractions, pas d'implémentations
from foampilot.base import Meshing  # abstraction
from foampilot.solver import Solver  # orchestration
```

#### 3.3 Composition over inheritance

```python
# Solver est composé de SystemDirectory + ConstantDirectory + Boundary
# Pas d'héritage complexe, juste de la composition
```

### Section 4 — Les choix techniques et leurs conséquences

#### 4.1 Pourquoi Python ?

- **Lisibilité** : la CFD est déjà complexe, le code ne doit pas l'être
- **Écosystème** : NumPy, PyVista, pandas, Matplotlib
- **Intégration** : OpenFOAM est accessible via subprocess
- **Adoption** : Python est le langage des ingénieurs CFD modernes

#### 4.2 Pourquoi une structure modulaire ?

- **Testabilité** : chaque module peut être testé indépendamment
- **Maintenabilité** : une modification dans `boundaries/` n'impacte pas `solver/`
- **Extensibilité** : ajouter un nouveau mesher = ajouter un fichier dans `mesh/`
- **Documentation** : chaque module a sa propre doc

#### 3.3 Pourquoi des propriétés Python ?

```python
# Configuration déclarative vs impérative
solver.compressible = True  # déclaratif
# vs
solver.set_compressible(True)  # impératif
```

Les propriétés permettent une API fluide et auto-documentée.

### Section 5 — Comment contribuer au projet

#### 5.1 Structure des contributions

```
1. Fork et clone
2. Créer une branche feature/
3. Ajouter des tests dans test/
4. Mettre à jour la documentation
5. Ouvrir une Pull Request
```

#### 5.2 Standards de code

- PEP 8 obligatoire
- Type hints sur toutes les signatures
- Docstrings Google style
- Tests unitaires pour toute nouvelle fonctionnalité

#### 5.3 Documentation

- `README.md` : vue d'ensemble
- `docs/` : documentation détaillée
- `examples/` : cas d'usage concrets
- `AGENTS.md` : conventions de développement

### Section 6 — Les leçons apprises

1. **Commencer simple, structurer plus tard** : le premier script de 200 lignes est devenu une plateforme grâce à la refactorisation progressive.
2. **La documentation est du code** : chaque module est documenté, chaque exemple est testé.
3. **Les tests sont une preuve de rigueur** : dans le monde scientifique, la reproductibilité est reine.
4. **Communiquer, c'est coder** : un README clair vaut mieux qu'une architecture parfaite mais invisible.

### Conclusion — Un projet vivant

foampilot n'est pas fini. C'est un projet vivant, qui évolue avec ses utilisateurs et ses contributeurs. Si vous voulez comprendre comment une idée simple — "Python devrait éditer les fichiers OpenFOAM à ma place" — peut devenir une plateforme complète, plongez dans le code.

Et si vous voulez contribuer : forkez, codez, testez, documentez. L'architecture est prête à accueillir de nouvelles fonctionnalités.

**Ressources :**
- Dépôt : [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)
- Contribuer : [github.com/stevendaix/foampilot/blob/main/CONTRIBUTING.md](https://github.com/stevendaix/foampilot/blob/main/CONTRIBUTING.md)

---

*Article en cours d'amélioration — version 1.0*
