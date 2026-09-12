# Automatiser OpenFOAM avec Python : guide pas-à-pas pour des workflows reproductibles

*Comment transformer des heures d'édition manuelle de dictionnaires en 20 lignes de Python reproductibles et testables — avec validation physique intégrée.*

---

## Introduction : Le réveil à 2h du matin

Si vous avez déjà utilisé OpenFOam, cette scène vous est familière : une simulation critique doit tourner avant une réunion. Vous copiez un cas de référence, éditez `controlDict`, puis `fvSchemes`, puis `fvSolution`. Vous créez les champs dans `0/` — `U`, `p`, `k`, `epsilon`, `nut`. Vous ajustez les conditions aux limites, vérifiez la syntaxe, corrigez un point-virgule oublié, et relancez. Deux heures plus tard, la simulation démarre enfin.

Ce n'est pas de la CFD. C'est de la **gestion de fichiers texte**.

Dans cet article, je vous montre comment remplacer ce workflow manuel par un script Python reproductible, testable et versionnable — avec **foampilot**. Et pour ne pas rester dans l'abstraction, je prends un cas physique concret : l'écoulement de Poiseuille plan.

---

## Le problème : 7 étapes, 7 opportunités d'erreur

Un workflow OpenFOAM classique ressemble à ça :

1. **Copier** un cas de référence (`cp -r base_case my_case`)
2. **Éditer** `system/controlDict` (application, temps, schémas)
3. **Éditer** `system/fvSchemes` et `system/fvSolution`
4. **Créer** les champs initiaux dans `0/` (U, p, k, epsilon, nut, T…)
5. **Configurer** les conditions aux limites pour chaque patch et chaque champ
6. **Vérifier** la syntaxe de chaque fichier
7. **Lancer** la simulation et croiser les doigts

Chaque étape est une opportunité d'erreur. Un `;` oublié, un nom de champ incorrect, une condition aux limites mal copiée — et c'est le crash assuré. Pire : reproduire cette simulation signifie recommencer tout le processus manuellement, avec le risque d'oublier une modification.

**La conséquence** : pas de tests, pas de versionnement, pas de reproductibilité.

---

## La solution : 20 lignes de Python (avec physique)

Avec foampilot, le même cas se crée en **20 lignes de code**. Pas de copier-coller, pas d'édition manuelle. Voici un exemple complet basé sur l'écoulement de Poiseuille plan :

```python
from foampilot import Solver, Meshing, FluidMechanics, ValueWithUnit

# 1. Définir le fluide et calculer ses propriétés physiques
fluid = FluidMechanics(
    fluid_name=FluidsList.Air,
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)
props = fluid.get_fluid_properties()
nu = props['kinematic_viscosity']  # 1.52e-5 m²/s

# 2. Définir le solveur
solver = Solver(case_path="./poiseuille")
solver.transient = False
solver.turbulence_model = "laminar"

# 3. Injecter la physique
solver.constant.transportProperties.nu = nu

# 4. Conditions aux limites
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    "inlet", "velocityInlet", velocity=(1, 0, 0)
)
solver.boundary.apply_condition_with_wildcard(
    "outlet", "pressureOutlet"
)

# 5. Écrire et lancer
solver.write_case()
solver.boundary.write_boundary_conditions()
solver.run_simulation(nb_proc=2)
```

**Ce qui change** : la viscosité n'est pas copiée d'un tableur. Elle est calculée par `pyfluids` en fonction du fluide, de la température et de la pression. Si vous changez de fluide (eau, huile, sang), la valeur est recalculée automatiquement.

---

## Ce qui se passe sous le capot

### Configuration par propriétés

Le `Solver` utilise des **propriétés Python** (`@property`) pour déclencher des mises à jour automatiques. Quand vous faites `solver.turbulence_model = "laminar"`, le solveur sélectionne automatiquement le module OpenFOAM approprié et régénère la liste des champs nécessaires.

```python
# foampilot/solver/solver.py
class Solver:
    @property
    def turbulence_model(self) -> str:
        return self._turbulence_model

    @turbulence_model.setter
    def turbulence_model(self, value: str):
        self._turbulence_model = value
        if self.boundary:
            self.boundary.turbulence_model = value
        self._update_solver()
```

C'est ce qu'on appelle la **configuration déclarative** : vous déclarez ce que vous voulez, et le système s'organise pour le réaliser.

### Génération dynamique des champs

La classe `CaseFieldsManager` inspecte la configuration et détermine quels champs initiaux sont nécessaires :

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
```

Trois lignes de logique remplacent la création manuelle de 3 à 6 fichiers `0/` avec leurs dimensions et valeurs par défaut.

### Validation : comparer à la solution analytique

Pour Poiseuille plan, la solution analytique est :

```
u(y) = (1 / 2ν) × (dp/dx) × (h² - y²)
```

Avec foampilot, vous pouvez valider vos résultats directement en Python :

```python
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
import numpy as np

fp = FoamPostProcessing(case_path="./poiseuille")
mesh = fp.read_direct(time_step=100)

# Extraire le profil de vitesse sur la ligne médiane
centreline = mesh.slice(normal="y", origin=(0, 0, 0))
y_points = centreline.points[:, 2]

# Solution analytique
nu = 1.52e-5
dpdx = -5.0
H = 1.0
U_analytical = (1 / (2 * nu)) * dpdx * (H**2 - y_points**2)

# Comparaison
U_numerical = centreline.point_data["U"][:, 0]
error = np.mean(np.abs(U_numerical - U_analytical))
print(f"Erreur moyenne sur le profil : {error:.4f} m/s")
```

**Résultat attendu** : erreur < 5% pour un maillage suffisamment raffiné.

---

## Les bénéfices concrets

### Reproductibilité

Le script est la documentation. Deux exécutions du même script produisent le même cas. Pas de "ça marchait la dernière fois" — le versionnement Git trace chaque modification.

```bash
git add poiseuille.py
git commit -m "feat: add laminar Poiseuille case with pyfluids validation"
```

### Versionnement

Chaque modification de configuration est tracée. Vous savez exactement quand vous avez changé `solver.turbulence_model` de `"laminar"` à `"kEpsilon"`, et pourquoi.

### Études paramétriques

Une boucle Python remplace des dizaines de copier-coller :

```python
for Re_target in [100, 500, 1000]:
    for fluid_name in ["Air", "Water"]:
        fluid = FluidMechanics(
            fluid_name=FluidsList[fluid_name],
            temperature=ValueWithUnit(293.15, "K"),
            pressure=ValueWithUnit(101325, "Pa"),
            velocity=ValueWithUnit(Re_target * nu / H, "m/s")
        )
        case = Solver(case_path=f"./cases/Re_{Re_target}_{fluid_name}")
        # ... config ...
        case.run_simulation()
```

### Tests

Parce que le cas est généré par du code, vous pouvez écrire des tests :

```python
def test_poiseuille_case_generates_correct_files():
    solver = Solver(case_path="./test_poiseuille")
    solver.transient = False
    solver.turbulence_model = "laminar"
    solver.write_case()
    
    assert (Path("./test_poiseuille/system/controlDict")).exists()
    assert (Path("./test_poiseuille/0/U")).exists()
    assert (Path("./test_poiseuille/0/p")).exists()
```

---

## Étendre le workflow

### Ajouter un maillage

```python
from foampilot.base import Meshing

meshing = Meshing(case_path="./poiseuille", mesher="blockMesh")
meshing.write()
```

### Générer un rapport PDF avec statistiques physiques

```python
from foampilot.report import CFDReportGenerator

report = CFDReportGenerator(
    case_path="./poiseuille",
    title="Écoulement de Poiseuille plan — Validation"
)
report.add_statistic("Re", 65800, "-", "Nombre de Reynolds")
report.add_statistic("nu", 1.52e-5, "m²/s", "Viscosité cinématique (air à 20°C)")
report.add_statistic("U_max", 1.0, "m/s", "Vitesse maximale")
report.save_latex_report(compile_pdf=True)
```

---

## Conclusion : La CFD reproductible, avec physique

OpenFOAM est un outil puissant, mais sa configuration manuelle est un frein à la reproductibilité. foampilot ne cache pas OpenFOAM — il l'encapsule dans un workflow Python moderne, testable et automatisable.

Mais au-delà de l'automatisation, foampilot intègre la **physique** directement dans le code : propriétés de fluides, nombres sans dimension, validation analytique. Ce n'est pas un gadget — c'est la garantie que vos simulations sont fondées sur des principes physiques vérifiables.

Dans le prochain article, je vous emmène dans les coulisses du design de foampilot : pourquoi j'ai volontairement répété du code pour rendre l'API plus prévisible, et ce que ça change pour l'expérience développeur.

**Ressources :**
- Dépôt : [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)
- Exemples : [github.com/stevendaix/foampilot/tree/main/examples](https://github.com/stevendaix/foampilot/tree/main/examples)

---

*Article en cours de amélioration — version 1.0*
