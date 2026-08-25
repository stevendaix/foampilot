# Turbine flottante et OpenFOAM 13

Cette extension apporte une couche déclarative Foampilot autour des physiques du dépôt [thesis-FloatingTurbine](https://github.com/fronterapp/thesis-FloatingTurbine). Elle couvre la source **actuator-line** `axialFlowTurbineALSource`, le mouvement de corps rigide **sixDoF** et les lignes d’amarrage caténaires `mooringLine`.

## Choix d’architecture

Les cas historiques du dépôt source ciblent OpenFOAM v2012 et copient des arbres de cas complets. Foampilot suit au contraire son principe de dictionnaires générés : `FloatingTurbine` ne copie pas les anciens cas et produit uniquement les fichiers appartenant à la physique (`constant/fvOptions` par défaut et, lorsqu’il y a des lignes d’amarrage, `constant/dynamicMeshDict`). Les bibliothèques C++ restent des dépendances runtime explicites, enregistrées dans `system/controlDict` par `configure_solver()`.

Le code C++ fourni par le dépôt source s’enregistre dans l’API historique `fvOptions`. La méthode `render_fv_models()` permet de produire la variante `fvModels` attendue par un portage OpenFOAM 13 de cette bibliothèque, mais elle n’est pas sélectionnée silencieusement : `write(..., source_container="fvModels")` doit être demandé explicitement. Cela évite de générer un dictionnaire moderne que la bibliothèque historique ne saurait pas charger.

## Exemple

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
turbine.write(solver.case_path, cell_zone="rotor")  # fvOptions pour la bibliothèque historique
# Après portage C++ vers l’API OpenFOAM 13 :
# turbine.write(solver.case_path, cell_zone="rotor", source_container="fvModels")
```

Avant l’exécution, il faut compiler les bibliothèques compatibles avec OpenFOAM 13 et vérifier que le maillage contient la zone `rotor` et le patch `floater`. La validation Python contrôle les vecteurs unitaires, les dimensions positives et la présence des entrées physiques essentielles ; elle ne remplace pas `checkMesh`, la compilation C++ ni un test de convergence.

## Vérifications recommandées

| Étape | Contrôle | Résultat attendu |
| --- | --- | --- |
| 1 | `pytest foampilot/test/wind/test_floating_turbine.py` | Rendu et validations Python cohérents |
| 2 | Vérification des fichiers générés | `fvOptions` (ou `fvModels` après portage), `dynamicMeshDict` et `controlDict` complets |
| 3 | Compilation des bibliothèques | `libfloatingTurbinesFoam.so` et `libfloatingSixDoFRigidBodyMotion.so` chargées |
| 4 | `checkMesh` | Zone rotor et patch floater présents, maillage valide |
| 5 | Petit cas transitoire | Résidus finis, forces et mouvement sixDoF écrits |
| 6 | Étude physique | Sensibilité au pas de temps, au maillage et aux coefficients d’amarrage documentée |

## Provenance et limites

Les noms des modèles, paramètres et bibliothèques sont alignés sur le dépôt source fourni et sur ses README. Celui-ci annonce une base OpenFOAM v2012 et une diffusion à visée éducative ; cette PR fournit donc l’intégration déclarative et les contrôles de cas dans Foampilot, mais ne prétend pas transformer automatiquement le code C++ historique en bibliothèque OpenFOAM 13 validée. La compilation effective doit être faite dans un environnement OpenFOAM 13 avec les sources C++ portées et leurs dépendances disponibles.
