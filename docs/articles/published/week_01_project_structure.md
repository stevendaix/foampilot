# J'ai construit un wrapper Python pour OpenFOAM : architecture, design et leçons d'un projet open-source

*Un retour d'expérience entre architecture logicielle, mécanique des fluides et présentation de projet.*

---

## Introduction : L'ingénieur qui en avait assez des dictionnaires

Je suis ingénieur CFD. Si vous avez déjà utilisé OpenFOAM, vous savez de quoi je parle : ce mélange étrange de puissance brute et de friction cognitive. Vous réglez votre cas, tout est prêt, et puis… il faut éditer `controlDict`. Puis `fvSchemes`. Puis `fvSolution`. Puis chaque champ dans `0/`. Et si vous oubliez un point-virgule, c'est le crash assuré.

Après des années à copier-coller des répertoires et à éditer des fichiers texte à la main, j'ai fini par me poser une question : *et si Python pouvait faire ce travail à ma place ?*

C'est de cette frustration qu'est né **foampilot**. Mais au-delà du code, ce projet m'a appris deux choses essentielles :

1. **Le design d'outils scientifiques repose sur la réduction de la charge cognitive, pas sur la minimisation du code.**
2. **Un projet open-source ne vit pas seulement par sa publication — il vit par son outil.**

Dans cet article, je vous emmène dans les coulisses de foampilot : comment j'ai conçu l'architecture, pourquoi j'ai fait des choix de design contre-intuitifs, et comment j'ai structuré le projet pour qu'il soit adopté — pas seulement utilisé.

---

## Partie 1 : Le problème — Pourquoi OpenFOAM mérite un wrapper

OpenFOAM est une boîte à outils formidable. Mais son modèle de configuration par dictionnaires texte, bien que puissant, est source de trois maux chroniques :

### 1.1 La répétition fastidieuse

Chaque simulation nécessite la création de **plusieurs dizaines de fichiers** :
- `system/controlDict`, `fvSchemes`, `fvSolution`, `decomposeParDict`
- `constant/transportProperties`, `turbulenceProperties`, `physicalProperties`
- `0/U`, `0/p`, `0/k`, `0/epsilon`, `0/nut`, `0/T`, etc.

Chaque fichier suit une syntaxe stricte : blocs, points-virgules, guillemets, en-têtes `FoamFile`. Une erreur de frappe suffit pour faire échouer `blockMesh` ou `foamRun`.

### 1.2 La non-reproductibilité

Dans un workflow manuel, reproduire une étude signifie copier un répertoire complet, espérer ne pas avoir oublié un fichier caché, puis éditer manuellement les paramètres modifiés. Résultat : deux simulations "identiques" peuvent donner des résultats différents parce qu'un fichier a été modifié à la main entre les deux.

### 1.3 L'absence de tests

Quand vos cas sont construits à la main, impossible d'écrire un test unitaire qui vérifie la cohérence des fichiers générés. La moindre régression passe inaperçue jusqu'à l'exécution.

**La solution ?** Transformer OpenFOAM en un **générateur de cas**, pas un éditeur de fichiers. C'est exactement ce que fait foampilot.

---

## Partie 2 : Architecture — Comment foampilot orchestre OpenFOAM

foampilot est structuré autour de trois modules principaux : **Solver**, **Meshing** et **Report**. Chacun joue un rôle d'orchestrateur, et leur interaction forme une architecture en couches propre et extensible.

### 2.1 Le Solver : configuration par propriétés

Le cœur de foampilot est la classe `Solver` (`foampilot/solver/solver.py`). Au lieu de demander à l'utilisateur de sélectionner un solveur et d'éditer des fichiers, elle utilise des **propriétés Python** pour piloter la configuration automatiquement.

Voici un extrait concret :

```python
# foampilot/solver/solver.py
class Solver:
    def __init__(self, case_path: str | Path):
        self.case_path = Path(case_path)
        self._solver: Optional[BaseSolver] = None
        self._compressible = False
        self._with_gravity = False
        self._is_vof = False
        self._is_solid = False
        self._transient = False
        self._turbulence_model = "kEpsilon"
        self._update_solver()

    @property
    def compressible(self) -> bool:
        return self._compressible

    @compressible.setter
    def compressible(self, value: bool):
        self._compressible = value
        self._update_solver()
```

Chaque fois que vous modifiez une propriété (`solver.compressible = True`, `solver.transient = True`, etc.), la méthode `_update_solver()` est appelée. Elle sélectionne automatiquement le bon solveur OpenFOAM et reconstruit toute la chaîne de dépendances : gestionnaire de champs, conditions aux limites, propriétés physiques.

**L'effet pour l'utilisateur ?** Une configuration déclarative qui ressemble à du Python standard :

```python
from foampilot import Solver

solver = Solver(case_path="./mon_cas")
solver.transient = True
solver.turbulence_model = "kOmegaSST"
solver.compressible = True
solver.with_gravity = True
solver.boundary.set_condition("inlet", "velocityInlet", velocity=(10, 0, 0))
solver.write_case()
solver.run_simulation(nb_proc=4)
```

Tout est généré automatiquement. Les fichiers `0/`, `system/`, `constant/` sont créés sans qu'un seul dictionnaire soit édité à la main.

### 2.2 Le Meshing : une factory pour les stratégies de maillage

La classe `Meshing` (`foampilot/base/meshing.py`) est une **factory** qui unifie l'accès aux trois moteurs de maillage supportés : `blockMesh`, `gmsh` et `snappyHexMesh`.

```python
from foampilot.base import Meshing

meshing = Meshing(case_path="./mon_cas", mesher="gmsh")
meshing.add_file("surfaceFeatureExtractDict", {...})
meshing.write()
```

Derrière cette interface uniforme, chaque mesher sait écrire ses propres fichiers et lancer son propre pipeline. La factory ne fait que choisir l'implémentation appropriée — c'est le **design pattern Strategy** appliqué à la CFD.

### 2.3 Le Report : du post-processing au PDF automatisé

Le module `report` (`foampilot/report/report_generator.py`) complète la chaîne en générant des documents professionnels à partir des résultats de simulation. Il supporte trois backends :

- **LaTeX/PDF** via `LatexDocument`
- **Typst** via `ScientificDocument`
- **HTML interactif** avec Plotly

Même flux de données, trois formats de sortie. L'utilisateur choisit son support sans changer son code.

---

## Partie 3 : Simplicité par abstraction — Comment masquer la complexité sans la perdre

Le plus grand défi d'un wrapper OpenFOAM est de **générer des fichiers valides** sans empêcher l'utilisateur d'accéder à la puissance brute d'OpenFOAM quand il en a besoin.

### 3.1 OpenFOAMFile : le traducteur Python → OpenFOAM

Tous les dictionnaires OpenFOAM dans foampilot passent par la classe `OpenFOAMFile` (`foampilot/base/openFOAMFile.py`). Elle connaît la syntaxe OpenFOAM : headers `FoamFile`, blocs imbriqués, listes, dimensions, et même la conversion d'unités.

Voici ce qu'elle fait pour vous :

```python
# foampilot/base/openFOAMFile.py
class OpenFOAMFile:
    DEFAULT_UNITS = {
        "nu": "m^2/s", "mu": "Pa.s", "rho": "kg/m^3",
        "k": "m^2/s^2", "epsilon": "m^2/s^3", "omega": "1/s",
        "U": "m/s", "p": "Pa", "T": "K",
    }

    def _format_value(self, key: str, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return format(value, ".15g")
        if ValueWithUnit and isinstance(value, ValueWithUnit):
            unit = self.DEFAULT_UNITS.get(key)
            val = value.get_in(unit) if unit else value.magnitude
            return f'{val:.15g}'
        return str(value)
```

Un `True` devient `true`. Un flottant est formaté en 15 chiffres significatifs. Un `ValueWithUnit(10, "m/s")` devient `10` dans le bon contexte. Le développeur n'a plus à se soucier de la syntaxe OpenFOAM.

### 3.2 CaseFieldsManager : la logique de configuration dynamique

La classe `CaseFieldsManager` (`foampilot/base/cases_variables.py`) détermine quels champs initiaux sont nécessaires en fonction de la physique :

```python
# foampilot/base/cases_variables.py
def _generate_fields(self) -> None:
    self.fields.clear()
    pressure_name = "p_rgh" if self.with_gravity and not self.compressible else "p"
    self.fields[pressure_name] = {"value": ValueWithUnit(0, "Pa")}
    if not self.is_solid:
        self.fields["U"] = {"value": ValueWithUnit(0, "m/s")}
    if self.energy_activated or self.compressible:
        self.fields["T"] = {"value": ValueWithUnit(300, "K")}
    if self.turbulence_model:
        self._generate_turbulence_fields()
```

Cinq lignes de logique remplacent la création manuelle de 3 à 6 fichiers `0/` avec les bons noms, dimensions et valeurs par défaut. Pour la CHT, la méthode `_generate_region_fields()` étend cette logique par région : les solides ne reçoivent que `T`, les fluides reçoivent `U`, `p`, `T` et les champs de turbulence.

---

## Partie 4 : Le design pattern clé — La répétition intentionnelle

C'est le point le plus contre-intuitif de foampilot : **j'ai volontairement répété des motifs de code** là où la plupart des architectes chercheraient à factoriser.

### 4.1 Le `write()` universel

Dans foampilot, presque tous les objets majeurs exposent une méthode `write()` :

| Classe | Ce que fait `write()` |
|---|---|
| `OpenFOAMFile` | Écrit un fichier dictionnaire |
| `SystemDirectory` | Écrit `controlDict`, `fvSchemes`, `fvSolution` |
| `ConstantDirectory` | Écrit `transportProperties`, `turbulenceProperties`, etc. |
| `Meshing` | Écrit `blockMeshDict`/`snappyHexMeshDict` + fichiers système |
| `BlockMesher` | Écrit `blockMeshDict` |
| `GmshMesher` | Exporte le maillage vers OpenFOAM |
| `SnappyMesher` | Écrit `snappyHexMeshDict` |
| `ChtSolver` | Écrit tous les fichiers région-spécifiques |

Cette uniformité a un effet profond : **vous n'avez jamais à vous demander "comment j'écris cet objet ?"**. La réponse est toujours la même.

```python
for component in [solver.system, solver.constant, meshing]:
    component.write()
```

En programmation orientée objet, on appelle ça le **Command Pattern** dans sa forme la plus pure : chaque objet encapsule l'action de se persister lui-même. La répétition du nom `write()` n'est pas un défaut — c'est un **contrat d'interface** qui rend l'API prévisible.

### 4.2 `apply_condition_with_wildcard` : une signature constante

La classe `Boundary` (`foampilot/boundaries/boundaries_dict.py`) expose une méthode unique pour appliquer des conditions aux limites :

```python
def apply_condition_with_wildcard(self, pattern: str, condition_type: str, **kwargs):
```

Que vous appliquiez une `velocityInlet`, une `pressureOutlet` ou un `wall` avec friction, l'entrée est toujours la même : un pattern regex, un type, et des arguments nommés.

### 4.3 Pourquoi la répétition est une vertu ici

Dans la plupart des projets, la duplication de code est un anti-pattern. Dans foampilot, elle sert trois objectifs :

1. **Découvrabilité** : Quand chaque objet a un `write()`, vous n'avez pas à consulter la documentation pour trouver le bon nom de méthode.
2. **Composabilité** : Des interfaces uniformes permettent des algorithmes génériques.
3. **Réduction de la charge cognitive** : Une API où tout se fait de la même façon élimine une variable mentale.

J'aime à dire que foampilot **répète pour simplifier**. La factorisation à tout prix crée des abstractions invisibles ; la répétition intentionnelle crée une interface **apprenable en 10 minutes**.

---

## Partie 5 : La démo technique — L'export direct Gmsh → polyMesh

Si je devais ne citer qu'une prouesse technique du projet, ce serait l'**export direct depuis Gmsh vers le format polyMesh d'OpenFOAM**, sans passer par `gmshToFoam`.

Le module `direct_openfoam_exporter.py` écrit directement les fichiers `points`, `faces`, `owner`, `neighbour`, `boundary` et `cellZones` en interrogeant l'API Python de Gmsh. Il gère :

- Les tétraèdres (type 4) et hexaèdres (type 5)
- L'orientation des faces par vérification géométrique (centroïde × normale)
- Le tri des faces internes par `(owner, neighbour)` avec rotation cyclique pour satisfaire l'ordre upper-triangular d'OpenFOAM
- La compaction des points inutilisés par région
- Les maillages multi-régions pour la CHT

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("case")
# ... construction de la géométrie, maillage ...
exporter = DirectOpenFOAMExporter("/path/to/case")
exporter.export_single_region()  # ou export_multi_region() pour la CHT
gmsh.finalize()
```

Cette fonctionnalité n'est pas qu'un gadget : elle supprime une dépendance externe, élimine les erreurs de conversion et donne un contrôle total sur le pipeline de maillage.

---

## Partie 6 : Comment présenter un projet open-source scientifique

Construire l'outil n'est que la moitié du travail. L'autre moitié, c'est de **faire en sorte que d'autres l'utilisent**.

### 6.1 Un README comme vitrine, pas comme documentation

Le README de foampilot n'est pas un manuel d'utilisation. C'est une **promesse** : un titre clair, un elevator punch, des fonctionnalités avec des verbes d'action, et une section "Ce que foampilot n'est pas" pour éviter les malentendus. L'objectif : en 30 secondes de lecture, un visiteur sait si l'outil est fait pour lui.

### 6.2 La documentation multilingue comme stratégie d'adoption

Le README existe en trois versions : anglais, français et chinois. Ce n'est pas un hasard. OpenFOAM est utilisé mondialement, mais beaucoup d'utilisateurs se heurtent à la barrière de l'anglais technique. Une documentation multilingue envoie un message fort : *"ce projet est fait pour vous, quel que soit votre langue"*.

### 6.3 Les exemples comme argument de vente

Le répertoire `examples/` contient des cas complets : électronique CHT, chauffage de canal, aéroacoustique, etc. Chaque exemple est un **argument de vente vivant**. Un ingénieur qui se demande "est-ce que foampilot peut gérer mon cas ?" peut cloner le dépôt et exécuter l'exemple en 5 minutes.

### 6.4 La documentation technique comme mémoire du projet

Le site de documentation (`docs/` avec MkDocs) sert deux publics : les **nouveaux utilisateurs** qui cherchent à démarrer, et les **contributeurs** qui cherchent à comprendre l'architecture. Un bon projet open-source doit être navigable par les deux.

### 6.5 Les tests comme preuve de maturité

Le répertoire `test/` et les commandes de test documentées dans `AGENTS.md` envoient un signal important : *"ce projet est testé, donc fiable"*. Dans le monde scientifique, la reproductibilité est reine.

---

## Partie 7 : Conclusion — L'outil comme produit de communication

Si je devais résumer ce que foampilot m'a appris, ce serait ceci :

**Un projet scientifique open-source est un produit à deux faces :**
- **La face technique** : le code, les algorithmes, la documentation de référence.
- **La face communication** : le README, les exemples, la documentation utilisateur, la démo.

La plupart des chercheurs excellent sur la première face. Peu investissent suffisamment la seconde.

Construire foampilot m'a forcé à penser comme un ingénieur *et* comme un chef de produit. Chaque méthode `write()` est un choix de design. Chaque exemple dans `examples/` est un argument d'adoption. Chaque ligne du README est une promesse tenue — ou rompue.

Si vous êtes chercheur ou ingénieur et que vous hésitez à open-sourcer votre outil, sachez ceci : **votre outil mérite d'être utilisé, mais il doit aussi être compréhensible**. Investissez dans la présentation autant que dans le code. Écrivez des exemples. Documentez les concepts. Faites des tutoriels.

Et si vous avez passé trop de temps à éditer des dictionnaires OpenFOAM à la main… peut-être que votre prochain projet devrait commencer par un `pip install foampilot`.

---

**Ressources :**
- Dépôt GitHub : [https://github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation : [https://stevendaix.github.io/foampilot/](https://stevendaix.github.io/foampilot/)
- Licence MIT

---

*Article en cours d'amélioration — version 1.0*
