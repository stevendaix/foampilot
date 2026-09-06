# Turbine flottante et OpenFOAM 13

Cette extension apporte une couche déclarative Foampilot autour des physiques du dépôt [thesis-FloatingTurbine](https://github.com/fronterapp/thesis-FloatingTurbine). Elle couvre la source actuator-line `axialFlowTurbineALSource`, le mouvement de corps rigide sixDoF et les lignes d’amarrage caténaires `mooringLine`.

## État du portage C++

Le dépôt source annonce OpenFOAM v2012 et recopie un ancien cœur sixDoF. OpenFOAM 13 fournit une implémentation native sous `libsixDoFRigidBodyMotion`, avec des interfaces différentes pour les dictionnaires, les pointeurs et la gestion du temps. Le portage ne recopie donc pas le cœur sixDoF historique : le plugin `third_party/openfoam13/floatingSixDoFRigidBodyMotion` réutilise l’implémentation native OpenFOAM 13 et porte les extensions `mooringLine`, `catenaryShape` et `constantLoad` comme restraints runtime.

Ce plugin a été compilé avec succès contre OpenFOAM 13. Son Makefile lie explicitement `-lsixDoFRigidBodyMotion` et `-lfiniteVolume`. Le script `Allwmake` est reproductible après chargement de `/opt/openfoam13/etc/bashrc`.

L’actuator-line `floatingTurbinesFoam` demeure une conversion séparée. Le code upstream est basé sur `fvOptions` et `cellSetOption`, alors que l’API OpenFOAM 13 attend `fvModel`/`fvModels`, le constructeur `(name, modelType, mesh, dict)`, `addSupFields()` et les callbacks de topologie. Le générateur Python conserve donc un choix explicite entre `fvOptions` historique et `fvModels` cible ; il ne prétend pas que la bibliothèque actuator-line est déjà compilée OpenFOAM 13.

## Utilisation Python

```python
from foampilot.solver import Solver
from foampilot.wind import FloatingTurbine, MooringLine

solver = Solver("cases/floating_turbine")
turbine = FloatingTurbine(
    rotor_diameter=178.0,
    position=(0.0, 0.0, 90.0),
    mooring_lines=(MooringLine(
        name="line1",
        anchor=(-837.6, 0.0, -200.0),
        attachment_point=(-20.4, 0.0, -14.0),
        mass_per_length=108.63,
        line_length=865.5,
    ),),
)
turbine.configure_solver(solver)
solver.write_case()
turbine.write(solver.case_path, cell_zone="rotor")
```

Pour compiler le plugin sixDoF porté :

```bash
. /opt/openfoam13/etc/bashrc
cd foampilot/third_party/openfoam13/floatingSixDoFRigidBodyMotion
./Allwmake
```

Avant l’exécution, il faut vérifier que le maillage contient la zone `rotor` et le patch `floater`, charger la bibliothèque portée dans le `controlDict` ou le dictionnaire de mouvement, puis exécuter `checkMesh`. La validation Python ne remplace ni la compilation actuator-line, ni `checkMesh`, ni un test transitoire de convergence.

## Vérifications recommandées

| Étape | Contrôle | Résultat attendu |
| --- | --- | --- |
| 1 | `pytest foampilot/test/wind/test_floating_turbine.py` | Rendu et validations Python cohérents |
| 2 | `git diff --check` | Aucun espace parasite ni erreur de patch |
| 3 | `./third_party/openfoam13/floatingSixDoFRigidBodyMotion/Allwmake` | Plugin sixDoF compilé avec OpenFOAM 13 |
| 4 | `checkMesh` | Zone rotor et patch floater présents, maillage valide |
| 5 | Petit cas transitoire | Restraints chargées, forces et mouvement sixDoF écrits |
| 6 | Portage actuator-line | `floatingTurbinesFoam` compilé comme `fvModel` et chargé par `fvModels` |
| 7 | Étude physique | Sensibilité au pas de temps, au maillage et aux coefficients d’amarrage documentée |

## Provenance et licence

Les fichiers du plugin proviennent des extensions physiques du dépôt source fourni et conservent leurs en-têtes de licence. Le cœur sixDoF OpenFOAM 13 reste celui distribué par OpenFOAM Foundation ; seules les extensions nécessaires aux cas de turbine flottante sont compilées dans le plugin Foampilot.
