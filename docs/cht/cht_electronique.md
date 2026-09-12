Nouveau Tutoriel : Refroidissement d'Électronique CHT avec Gmsh
Ce tutoriel démontre comment utiliser l'API Python de Gmsh intégrée à foampilot pour générer une géométrie complexe de dissipateur thermique à ailettes (heatsink) et réaliser une simulation de transfert thermique conjugué (CHT) multi-régions.
Concept du Tutoriel
L'objectif est de modéliser un composant électronique (source de chaleur) monté sur un dissipateur thermique en aluminium, le tout refroidi par un flux d'air forcé. Contrairement aux méthodes basées sur des fichiers STL, ce tutoriel utilise le noyau OpenCASCADE de Gmsh pour construire la géométrie de manière paramétrique directement en Python.
Composant	Matériau	Type de Région	Rôle
Puce (Chip)	Silicium	Solide	Source de chaleur volumique
Dissipateur (Heatsink)	Aluminium	Solide	Conduction et dissipation par ailettes
Air	Air	Fluide	Convection forcée
Points Forts du Workflow foampilot + Gmsh
1. Géométrie Paramétrique et Opérations Booléennes
L'utilisation de `GmshMesher` permet de définir des paramètres tels que le nombre d'ailettes, leur épaisseur et leur hauteur. Le script utilise des opérations booléennes (`fuse`, `fragment`) pour garantir une parfaite continuité du maillage aux interfaces solide-fluide, ce qui est crucial pour la précision du CHT.
2. Raffinement de Maillage Sélectif
Grâce à l'API Gmsh, nous appliquons un raffinement local automatique :
Maillage très fin aux interfaces solide/fluide pour capturer les gradients thermiques.
Couches limites (boundary layers) générées par extrusion pour une résolution précise de la couche limite convective.
Maillage plus lâche en amont et en aval pour optimiser le temps de calcul.
3. Export Direct Multi-Régions
Le tutoriel met en avant la méthode `export_to_openfoam_direct` de foampilot. Cette fonctionnalité permet de passer directement de la mémoire Gmsh aux répertoires `constant/fluid/polyMesh` et `constant/solid/polyMesh` d'OpenFOAM sans passer par des fichiers intermédiaires `.msh`, évitant ainsi les erreurs de conversion courantes dans les cas multi-régions.
Structure du Tutoriel Proposé
Le tutoriel sera structuré autour d'un script `run.py` unique orchestrant les étapes suivantes :
Définition des Paramètres : Variables Python pour les dimensions du dissipateur et les conditions de flux.
Construction Gmsh : Appel aux méthodes `add_box`, `add_cylinder` et `boolean_fragment` pour créer les domaines imbriqués.
Configuration du Solveur : Initialisation de `Solver` avec `is_solid=False` et activation du CHT via le module `cht`.
Conditions aux Limites : Utilisation des `Physical Groups` nommés dans Gmsh pour appliquer les conditions `compressible::turbulentTemperatureCoupledBaffleMixed` aux interfaces.
Exécution et Post-traitement : Lancement de `chtMultiRegionFoam` et génération automatique de rapports incluant la résistance thermique calculée du système.
> "L'intégration de Gmsh dans foampilot transforme la création de maillages CHT complexes, autrefois fastidieuse, en un processus scriptable, reproductible et facilement optimisable."
Améliorations suggérées pour foampilot
Pour parfaire ce tutoriel, il est recommandé d'ajouter une méthode `add_heat_source` dans le module `fvModels` de foampilot, permettant de définir une puissance en Watts sur une région solide spécifique directement depuis le script principal.