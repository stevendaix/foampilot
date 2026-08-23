# Tutoriel Wolf Dynamics : CounterFlow Flame (OpenFOAM 13)

Ce dossier contient l'intégration FoamPilot du tutoriel de flamme à contre-courant (Day 1) issu de la formation "Overview of Chemical Processes with OpenFOAM" (Wolf Dynamics).

## Objectifs

- Démontrer l'utilisation du module `multicomponentFluid` dans OpenFOAM 13.
- Gérer les espèces chimiques et la combustion avec FoamPilot.
- Exécuter le cas de manière reproductible sans dépendre des scripts `Allrun` externes.

## Fichiers

- `adapter.py` : Adaptateur FoamPilot spécifique à ce cas (injection des contrôles, validation structurelle).
- `run_tutorial.py` : Lanceur exécutable du cas.
- `report/` : Résultats de l'exécution de référence (maillage, résidus, logs).
