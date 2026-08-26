# Utiliser solids4foam avec Foampilot

## Ce que montre le dépôt officiel

L’utilisation de solids4foam repose sur un **solveur unique nommé `solids4Foam`**. Le choix de la physique se fait dans `constant/physicsProperties`; pour un couplage fluide–solide, la valeur attendue est `type fluidSolidInteraction`. Le couplage est ensuite configuré dans `constant/fsiProperties`, le modèle mécanique dans `constant/solid/solidProperties` et la loi matériau dans `constant/solid/mechanicalProperties`.

Foampilot intègre maintenant cette couche de configuration sans réimplémenter le solveur solids4foam. Cela permet de conserver Foampilot pour la géométrie, le maillage, les propriétés et le post-traitement, tout en laissant solids4foam résoudre le problème fluide–structure.

## Génération des dictionnaires

```python
from foampilot.solids4foam import Solids4FoamCase, SolidMaterial

case = Solids4FoamCase(
    case_path="cases/beamInCrossFlow",
    fluid_patch="interface",
    solid_patch="interface",
    coupling="IQNILS",
    material=SolidMaterial(
        name="rubber",
        law="neoHookeanElastic",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.4,
    ),
    solid_model="nonLinearGeometryTotalLagrangianTotalDisplacement",
    solution_algorithm="PETScSNES",
)
case.write()
```

La méthode écrit les éléments suivants :

| Fichier | Rôle |
|---|---|
| `constant/physicsProperties` | Sélection de `fluidSolidInteraction` |
| `constant/fsiProperties` | Interface, relaxation, tolérance et nombre d’itérations FSI |
| `constant/solid/solidProperties` | Modèle solide et algorithme de résolution |
| `constant/solid/mechanicalProperties` | Densité, module d’Young, coefficient de Poisson et loi mécanique |
| `system/functions` | Efforts fluides et déplacement solide pour le suivi des résultats |

Les champs `0/fluid/*` et `0/solid/*`, les deux maillages régionaux et les fichiers `fvSchemes`/`fvSolution` ne sont pas écrasés. Cette séparation est volontaire : les conditions limites et les schémas numériques dépendent fortement de la version OpenFOAM, du modèle fluide et de la géométrie.

## Exécution

Le plan de commandes généré par `case.run_plan()` suit la logique du tutoriel officiel `beamInCrossFlow` : création du maillage de la région solide et de la région fluide, puis lancement de `solids4Foam`. En série :

```python
for command in case.run_plan(parallel=False):
    print(" ".join(command))
```

En parallèle, le plan ajoute `decomposePar` pour chaque région, `solids4Foam -parallel`, puis la reconstruction de chaque région. L’exécution effective reste à faire dans un environnement où solids4foam et OpenFOAM sont compilés et initialisés.

## Correspondance avec un tutoriel officiel

La configuration générée correspond à la structure observée dans le tutoriel officiel `beamInCrossFlow` : deux régions nommées `fluid` et `solid`, un patch d’interface homonyme, une interface `IQNILS` ou `Aitken`, un modèle solide non linéaire et une loi matériau telle que `neoHookeanElastic`. Le tutoriel officiel utilise aussi des fonctions de suivi comme `forces` côté fluide et `solidPointDisplacement` côté solide.

> **Point de vigilance.** Le nom du solver fluide (`pimpleFluid`, `interFluid`, etc.), les suffixes de compatibilité OpenFOAM et les bibliothèques optionnelles comme PETSc doivent être alignés sur la version de solids4foam installée. Foampilot génère la configuration, mais ne compile pas automatiquement solids4foam.

## Limite actuelle

Cette intégration prépare les dictionnaires standards, mais elle ne fournit pas encore un constructeur automatique complet des champs `0/fluid` et `0/solid`, ni une conversion universelle de maillage vers deux régions. L’étape suivante recommandée est d’ajouter un adaptateur de maillage régional à partir des objets de géométrie Foampilot et un validateur qui vérifie la cohérence des patchs d’interface avant lancement.

## Références

[1]: https://www.solids4foam.com/documentation/overview.html "solids4foam — Overview"
[2]: https://www.solids4foam.com/tutorials/tutorial3.html "solids4foam — beamInCrossFlow"
[3]: https://github.com/solids4foam/solids4foam "solids4foam — dépôt officiel"

## Connexion automatique avec un modèle Gmsh Foampilot

Lorsque le modèle Gmsh actif contient deux volumes physiques nommés `FLUID` et `SOLID`, ainsi qu’une surface physique `interface`, la génération peut exporter directement les deux régions :

```python
import gmsh
from foampilot.solids4foam import Solids4FoamCase

# Le modèle peut être construit avec les helpers géométriques Foampilot.
gmsh.initialize()
gmsh.model.add("beamInCrossFlow")
# ... création des volumes et groupes physiques FLUID, SOLID, interface ...
gmsh.model.mesh.generate(3)

case = Solids4FoamCase("cases/beamInCrossFlow", fluid_patch="interface", solid_patch="interface")
result = case.prepare_from_gmsh(
    fluid_volume="FLUID",
    solid_volume="SOLID",
    interface_surface="interface",
)
gmsh.finalize()
```

Cette méthode effectue trois contrôles avant l’export : les deux volumes physiques 3-D existent, le groupe de surface d’interface existe, et la carte des régions associe bien les volumes aux répertoires `fluid` et `solid`. Elle appelle ensuite l’exporteur direct déjà présent dans Foampilot et produit :

```text
constant/fluid/polyMesh/*
constant/solid/polyMesh/*
```

Il ne s’agit pas d’un simple découpage arbitraire du maillage : la continuité géométrique et le nommage des faces d’interface doivent être préparés dans le modèle Gmsh. Une surface commune doit être associée au même nom de patch dans les deux régions. Le contrôle final doit être effectué avec `checkMesh -region fluid` et `checkMesh -region solid`, puis avec un cas solids4foam court avant toute étude paramétrique.

L’exporteur direct sait déjà écrire des maillages multi-régions à partir de groupes physiques 3-D. La nouvelle méthode fournit donc le lien manquant entre les objets Gmsh de Foampilot et la configuration solids4foam, sans ajouter de convertisseur externe.
