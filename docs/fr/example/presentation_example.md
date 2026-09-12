# Exemples FOAMPilot

FOAMPilot est une bibliothèque Python conçue pour faciliter la création, la configuration et l'exécution de simulations OpenFOAM. Elle propose une approche modulable et intuitive pour gérer les cas CFD, la génération de maillages, les conditions aux limites, les fonctions objets, ainsi que le post-traitement des résultats.

Cette section présente différents exemples illustrant les avantages et la flexibilité de FOAMPilot pour automatiser les workflows CFD et faciliter l'apprentissage d'OpenFOAM avec Python.

## Objectifs des exemples

Les exemples permettent de :

- Montrer comment initialiser un cas OpenFOAM depuis Python.
- Démontrer la génération et la modification de maillages à partir de fichiers JSON.
- Illustrer la définition des propriétés des fluides et des conditions aux limites.
- Mettre en place des `functionObjects` pour le suivi des grandeurs physiques (forces, pressions, moyennes de champ…).
- Créer et gérer des dictionnaires spécifiques OpenFOAM (`topoSetDict`, `createPatchDict`, etc.).
- Exécuter la simulation et automatiser le post-traitement.
- Fournir des exemples reproductibles pour l'apprentissage et le prototypage.

## Liste des exemples

Cette partie sera complétée au fur et à mesure de la conception des tests.  

- [Muffler](muffler/detailled_example_muffler.md) : Exemple détaillé d'un silencieux de moteur, montrant la génération de maillage complexe, les conditions aux limites, et l'analyse des résultats acoustiques et fluidiques.  
- [SimpleCar](simplecar/detailled_example.md) : Exemple basé sur le tutoriel officiel OpenFOAM [SimpleCar](https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar), illustrant la simulation d'un écoulement autour d'une voiture simple avec génération de maillage via JSON, application des conditions aux limites et suivi des forces aérodynamiques.

## Notes

Chaque exemple est fourni avec un script Python autonome qui :

1. Définit le chemin du cas (`current_path`).
2. Initialise les propriétés du fluide (densité, viscosité, pression, température…).
3. Initialise le solver FOAMPilot et les dossiers système/constant.
4. Génère le maillage à partir d'un fichier JSON.
5. Ajoute les `functionObjects` nécessaires (moyenne de champ, pression de référence, contrôle de temps, etc.).
6. Manipule les dictionnaires OpenFOAM pour la création de patchs et la définition des zones.
7. Applique les conditions aux limites avec l'API moderne.
8. Exécute la simulation.
9. Post-traite automatiquement les résultats et génère des exports CSV, JSON, PNG et HTML.

Ces exemples sont conçus pour être modulables et facilement adaptables à différents cas d'étude CFD.

