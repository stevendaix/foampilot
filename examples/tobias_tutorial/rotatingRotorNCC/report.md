# Rotating Rotor NCC

## Objet

Ce tutoriel de Tobias Holzmann simule un rotor en rotation en utilisant la condition aux limites Non-Conformal-Coupled (NCC) qui remplace l'ancienne interface AMI (Arbitrary Mesh Interface) dans OpenFOAM. Il démontre la génération du maillage dynamique et la simulation de l'écoulement induit par la rotation [1].

## Portage FoamPilot

`run.py` écrit les dictionnaires, copie le maillage UNV et les surfaces STL. Le workflow de maillage utilise `ideasUnvToFoam`, `snappyHexMesh`, `createPatch` et `createNonConformalCouples`. Comme pour les autres cas NCC, l'utilitaire `renumberMesh` a été retiré car il provoque une erreur avec les maillages dynamiques couplés sous OpenFOAM 13.

Le temps de simulation a été réduit à `endTime 0.001` et le `deltaT` a été ajusté à `0.0005` pour permettre un smoke run rapide qui valide la dynamique du maillage et le solveur incompressible. Le script original demandait à l'utilisateur de choisir entre le solveur et le mouvement de maillage uniquement ; FoamPilot exécute la simulation complète (solveur `incompressibleFluid` avec `solidBody` motion).

L’audit de l’API n’a identifié aucune méthode FoamPilot manquante.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Conversion UNV | Le maillage de fond est converti avec succès. |
| Maillage | `snappyHexMesh` génère le maillage du rotor. |
| Couplage NCC | `createNonConformalCouples` génère les couples entre `AMI1` et `AMI2` avec succès. |
| Simulation dynamique | `foamRun` charge le solveur de mouvement `solidBody` et `rotatingMotion`. |
| Calcul | Le calcul incompressible transitoire avance dans le temps et atteint `End` sans erreur. |

Le cas est **validé**. La validation démontre la configuration correcte du maillage dynamique et du couplage NCC, ainsi que l'exécution de la simulation d'écoulement induit par la rotation.

## Limites

Le temps de simulation a été considérablement réduit. La simulation ne permet pas d'observer le développement complet de l'écoulement autour du rotor, mais valide l'intégrité du workflow de maillage et du setup dynamique.

## Référence

[1]: https://holzmann-cfd.com/community/training-cases/rotating-rotor-ami — Tobias Holzmann, *Rotating Rotor AMI*.
