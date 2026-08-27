# Couplage fluide–structure natif dans Foampilot

## Objectif

Foampilot fournit désormais `foampilot.fsi`, une couche de génération de cas pour le **couplage fluide–corps rigide** réalisé entièrement par OpenFOAM. Le mécanisme repose sur `sixDoFRigidBodyMotion` : le solveur fluide calcule les efforts sur le patch du corps, OpenFOAM intègre la translation et la rotation du corps, puis le maillage est déplacé à l’itération suivante.

Cette approche ne lance ni preCICE, ni MOOSE, ni un solveur solide séparé. Elle est adaptée à une plaque, un volet, une membrane modélisée comme corps rigide, une pièce immergée ou un flotteur dont les six degrés de liberté sont suffisants.

> **Limite importante.** Ce module ne résout pas la déformation élastique ou plastique d’un solide. Pour une structure réellement déformable, il faut un solveur de mécanique des solides compatible avec OpenFOAM, tel que solids4foam, ou une stratégie de couplage externe. L’API native rend cette distinction explicite afin d’éviter de générer un cas physiquement trompeur.

## Exemple minimal

```python
from foampilot.fsi import NativeRigidFSI, RigidBody

fsi = NativeRigidFSI(
    case_path="cases/flap_fsi",
    body=RigidBody(
        name="flap",
        patch="flap",
        mass=0.25,
        centre_of_mass=(0.1, 0.0, 0.0),
        moment_of_inertia=(0.002, 0.002, 0.002, 0.0, 0.0, 0.0),
    ),
    variant="foundation13",
    restraints={
        "torsionalSpring": {
            "type": "linearSpring",
            "body": "flap",
            "anchor": (0.0, 0.0, 0.0),
            "stiffness": 2.0,
            "damping": 0.02,
        }
    },
)
paths = fsi.write()
```

La méthode `write()` produit `constant/dynamicMeshDict` et un fragment de function object dans `system/flapForces.functionObject`. Le fragment doit être inclus dans le `controlDict` du cas selon la convention déjà utilisée par la version d’OpenFOAM installée. Foampilot ne remplace pas les fichiers `0/U`, `0/p`, le maillage, `fvSchemes` ou `fvSolution`, car ces fichiers dépendent du cas fluide et du solveur choisi.

## Variantes supportées

| Variante | Dictionnaire généré | Usage |
|---|---|---|
| `foundation13` | bloc `mover` et `rigidBodyMotion` | OpenFOAM Foundation récent, notamment les workflows OpenFOAM 13 du dépôt |
| `legacy` | `dynamicMotionSolverFvMesh` et `sixDoFRigidBodyMotionCoeffs` | Installations OpenFOAM utilisant le format classique |

La syntaxe exacte des bibliothèques et des dictionnaires doit être vérifiée avec la version installée. Il est recommandé de comparer le fichier produit avec un tutoriel livré par cette version d’OpenFOAM avant une campagne de calcul.

## Organisation recommandée d’un cas

Le cas reste organisé comme un cas OpenFOAM standard :

```text
flap_fsi/
├── 0/
│   ├── U
│   ├── p
│   └── ...
├── constant/
│   ├── polyMesh/
│   └── dynamicMeshDict       # généré par NativeRigidFSI
└── system/
    ├── controlDict
    ├── fvSchemes
    ├── fvSolution
    └── flapForces.functionObject # fragment généré
```

Le patch `body.patch` doit être cohérent entre le maillage, les conditions limites et `RigidBody.patch`. Le choix du maillage mobile, des distances `innerDistance` et `outerDistance`, du pas de temps et des critères de qualité de maillage reste spécifique à la géométrie.

## Vérifications avant calcul

Il faut d’abord confirmer que le solver choisi accepte la dynamique de maillage et que la bibliothèque `rigidBodyMeshMotion` est disponible. Ensuite, il faut vérifier l’unité et la cohérence de la masse, du centre de masse et du tenseur d’inertie. Une masse ou une inertie erronée peut produire une réponse dynamique numériquement stable mais physiquement fausse.

Pour un calcul instationnaire, il est conseillé d’utiliser un pas de temps suffisamment petit pour résoudre la fréquence propre du système fluide–structure, de surveiller les résidus et de suivre les forces, moments, déplacement et rotation dans le temps. Une vérification de conservation et une étude de sensibilité au pas de temps restent nécessaires.

## Évolution possible

La prochaine extension naturelle est d’ajouter une couche de validation de cas : contrôle de l’existence du patch, contrôle de `dynamicMeshDict`, vérification de la présence de `nu` dans `transportProperties` lorsque ce fichier est nécessaire, et détection des incompatibilités entre variante OpenFOAM et solver. La mécanique des solides déformables doit rester un module séparé, car elle ne peut pas être remplacée par six degrés de liberté rigides.
