# DLBFoam hydrogen planar flame

Cet exemple porte dans **FoamPilot** le tutoriel [DLBFoam-Hydrogen-Tutorials](https://github.com/Aalto-CFD/DLBFoam-Hydrogen-Tutorials/tree/main/2D_planar_flame), cas 2D de flamme prémélangée pauvre hydrogène-air à propagation libre. Le cas de référence utilise un domaine de `200 x 200` épaisseurs de flamme, une perturbation sinusoïdale initiale et des conditions périodiques latérales [1].

## Reproduction

Depuis la racine du dépôt Foampilot, installer le paquet en mode editable puis sourcer une installation **OpenFOAM 13** qui contient les bibliothèques DLBFoam, FickianTransportFoam et le mécanisme PyJac compilé :

```bash
pip install -e foampilot
# Exemple : source /opt/openfoam13/etc/bashrc
cd examples/dlfoam_hydrogen_planar
python run.py
python run.py --run --np 1
# Smoke test borné : maillage et une seule itération
python run.py --run --nx 2 --ny 2 --end-time 1.1e-6 --write-interval 1
# Calcul de référence : 2000 x 2000 cellules et 100 temps de flamme
python run.py --run --np 1
```

Le premier appel crée un dossier `case/` propre. Le runner ré-écrit chaque fichier texte de `case_template/` avec `OpenFOAMDictAddFile`, normalise les en-têtes en version 13 et conserve les données initiales publiées, les fichiers `.dat`, le mécanisme et les includes `codeStream`. Le second appel exécute `blockMesh`, le solveur choisi et, pour un calcul MPI, `decomposePar` puis `reconstructPar` via l’API de commande de FoamPilot.

Sous OpenFOAM 13, le solveur recommandé est `foamRun -solver multicomponentFluid`, car `reactingFoam` est désormais un wrapper obsolète. Le runner conserve `reactingFoam` comme option explicite, mais utilise `foamRun` par défaut. Le runner compile automatiquement le mécanisme PyJac local (`constant/mech/Allwmake`) avant le maillage et arrête explicitement la reproduction lorsque `blockMesh` ou le solveur demandé ne sont pas disponibles.

Les options `--nx`, `--ny`, `--end-time` et `--write-interval` sont uniquement destinées aux smoke tests bornés. Sans ces options, le maillage et le temps de calcul restent ceux de la référence publiée.

## Validation

La validation automatisée vérifie que le cas est complet, que le journal du solveur existe et qu’un répertoire de temps numérique est produit. La comparaison quantitative des profils de température, vitesse et espèces doit être réalisée avec les bibliothèques DLBFoam/FickianTransportFoam effectivement compilées et avec le même nombre de processeurs que la référence. Le dépôt source indique qu’au-delà d’environ 150 temps de flamme, le champ doit être remappé pour maintenir la flamme dans le domaine ; cette opération reste une étape physique manuelle du tutoriel original [1].

> Les fichiers `case_template/` sont des données d’entrée scientifiques de référence, et non un résultat de calcul. Les sorties générées dans `case/` sont ignorées par Git.

## Références

[1]: https://github.com/Aalto-CFD/DLBFoam-Hydrogen-Tutorials/tree/main/2D_planar_flame "Aalto-CFD — DLBFoam Hydrogen Tutorials, 2D planar flame"
