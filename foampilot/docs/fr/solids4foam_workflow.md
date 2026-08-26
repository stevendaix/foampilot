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
