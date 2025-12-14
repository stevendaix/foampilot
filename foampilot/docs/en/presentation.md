# Vue d'Ensemble Conceptuelle et Théorique du Module foampilot

Le module `foampilot` est conçu comme une surcouche Python orientée objet pour la plateforme de simulation de mécanique des fluides numérique (CFD) **OpenFOAM**. Son objectif principal est d'abstraire la complexité de la configuration des cas OpenFOAM, qui repose traditionnellement sur des fichiers de dictionnaire textuels, en offrant une interface de programmation (API) Python intuitive et robuste.

L'architecture de `foampilot` reflète fidèlement la structure d'un cas OpenFOAM, chaque sous-module gérant un aspect fondamental de la simulation.

## 1. foampilot.solver : Le Cœur de la Physique et de la Numérique

Le sous-module `solver` est le centre de contrôle de la simulation. Il ne se contente pas d'exécuter un solveur OpenFOAM ; il gère la **sélection dynamique du solveur** en fonction des propriétés physiques et numériques définies par l'utilisateur.

| Concept OpenFOAM | Rôle dans foampilot | Description Théorique |
| :--- | :--- | :--- |
| **Solveur** (e.g., `simpleFoam`, `pimpleFoam`) | **Classe `Solver`** | La classe `Solver` agit comme un **gestionnaire de solveur intelligent**. En modifiant des propriétés booléennes (ex: `compressible`, `transient`, `is_vof`), elle sélectionne automatiquement le solveur OpenFOAM approprié qui correspond aux équations physiques à résoudre (Navier-Stokes pour fluide incompressible, équations d'état pour compressible, etc.). |
| **Configuration Physique** | **Propriétés du `Solver`** | Permet de définir la nature du problème (stationnaire vs. transitoire, mono-phase vs. multi-phase VOF, avec ou sans gravité/énergie). Cette abstraction garantit que l'utilisateur n'a pas à se soucier du nom exact du solveur OpenFOAM, mais uniquement de la physique du problème. |

## 2. foampilot.constant : La Définition du Milieu Physique

Le sous-module `constant` est dédié à la gestion du répertoire `constant` d'OpenFOAM, qui contient les propriétés du milieu et du maillage.

| Concept OpenFOAM | Rôle dans foampilot | Description Théorique |
| :--- | :--- | :--- |
| **Propriétés du Fluide** | **Classes `transportProperties`, `physicalProperties`** | Ces classes permettent de définir les propriétés essentielles du fluide (viscosité cinématique $\nu$, densité $\rho$, chaleur spécifique $C_p$, etc.). Elles sont cruciales pour la fermeture des équations de Navier-Stokes et la modélisation des transferts. |
| **Modèle de Turbulence** | **Classe `turbulenceProperties`** | Gère la sélection et la configuration des modèles de turbulence (ex: $k-\epsilon$, $k-\omega$ SST). Théoriquement, ces modèles ajoutent des équations de transport pour modéliser les effets de la turbulence sur l'écoulement, en fermant le système d'équations RANS (Reynolds-Averaged Navier-Stokes). |
| **Gravité** | **Classe `gravityFile`** | Permet d'activer et de définir le vecteur de gravité $\mathbf{g}$, essentiel pour les simulations de flottabilité (Boussinesq) ou les écoulements à surface libre. |

## 3. foampilot.system : Le Contrôle Numérique et Temporel

Le sous-module `system` gère le répertoire `system` d'OpenFOAM, qui dicte la manière dont les équations sont discrétisées et résolues.

| Concept OpenFOAM | Rôle dans foampilot | Description Théorique |
| :--- | :--- | :--- |
| **Contrôle de la Simulation** | **Classe `controlDictFile`** | Définit les paramètres temporels (pas de temps $\Delta t$, temps de début/fin), les fréquences d'écriture des résultats, et les fonctions d'exécution (ex: `runTimeControl` pour l'arrêt automatique). |
| **Schémas Numériques** | **Classe `fvSchemesFile`** | Gère la discrétisation des termes de l'équation (dérivées temporelles, termes de convection, termes de diffusion). Le choix des schémas (ex: Euler pour le temps, `upwind` ou `Gauss linear` pour la convection) impacte directement la stabilité et la précision de la solution numérique. |
| **Solveurs Algébriques** | **Classe `fvSolutionFile`** | Configure les solveurs matriciels utilisés pour résoudre les systèmes d'équations linéaires résultant de la discrétisation (ex: `PCG` pour la pression, `BiCGStab` pour la vitesse). Il définit également les critères de convergence (tolérance) et les stratégies de sous-relaxation. |

## 4. foampilot.mesh : Le maillage

Le sous-module `mesh` est responsable de la création du maillage, la discrétisation spatiale du domaine de calcul.

| Concept OpenFOAM | Rôle dans foampilot | Description Théorique |
| :--- | :--- | :--- |
| **Maillage Structuré** | **Classe `BlockMeshFile`** | Abstraction de l'utilitaire `blockMesh` d'OpenFOAM. Il permet de définir la géométrie du domaine par blocs hexaédriques, une méthode efficace pour les géométries simples ou paramétrables. |
| **Maillage Non-Structuré** | **Classes `gmsh_mesher`, `snappymesh`** | Ces classes gèrent l'intégration avec des outils de maillage plus avancés (`Gmsh`, `snappyHexMesh`) pour les géométries complexes importées (ex: fichiers STL). Elles préparent les fichiers de configuration nécessaires à ces utilitaires. |

## 5. foampilot.boundaries : Les Conditions aux Limites Physiques

Le sous-module `boundaries` est essentiel pour définir l'interaction du fluide avec son environnement.

| Concept OpenFOAM | Rôle dans foampilot | Description Théorique |
| :--- | :--- | :--- |
| **Conditions aux Limites** | **Classe `Boundary`** | Gère la définition des conditions aux limites pour chaque champ physique ($\mathbf{U}$, $p$, $k$, $\epsilon$, etc.) sur les patchs du maillage. Théoriquement, ces conditions sont nécessaires pour fournir les informations manquantes aux frontières du domaine, permettant la résolution unique des équations aux dérivées partielles. |
| **Types de Conditions** | **Méthodes de `Boundary`** | Offre des méthodes spécifiques pour les conditions physiques courantes : `set_velocity_inlet` (Dirichlet pour la vitesse), `set_pressure_outlet` (Neumann pour la pression), `set_wall` (non-glissement ou glissement), etc. |
| **Fonctions de Paroi** | **Intégration automatique** | La classe `Boundary` intègre la logique pour appliquer les fonctions de paroi (Wall Functions) appropriées pour les modèles de turbulence, permettant de modéliser la couche limite sans nécessiter un maillage extrêmement fin près des parois. |

## 6. foampilot.postprocess et foampilot.report : L'Analyse des Résultats

Ces sous-modules gèrent la phase finale de la simulation : l'extraction, l'analyse et la présentation des données.

| Sous-module | Rôle Conceptuel | Description Théorique |
| :--- | :--- | :--- |
| **`postprocess`** | **Visualisation et Analyse** | Utilise des bibliothèques comme **PyVista** pour charger les résultats OpenFOAM (fichiers VTK) et effectuer des opérations de post-traitement courantes (tranches, contours, vecteurs, lignes de courant). Il permet de visualiser les champs physiques et d'extraire des quantités dérivées (ex: critère Q, vorticité). |
| **`report`** | **Génération de Rapports** | Automatise la création de rapports de simulation structurés (ex: PDF). Il agrège les données clés (paramètres d'entrée, résidus de convergence, images de post-traitement) pour garantir la traçabilité et la reproductibilité des résultats. |

## 7. foampilot.utilities et foampilot.commons : Les Outils Transversaux

Ces modules fournissent des fonctionnalités de support essentielles à l'ensemble du framework.

| Sous-module | Rôle Conceptuel | Description Théorique |
| :--- | :--- | :--- |
| **`utilities.manageunits`** | **Gestion des Unités** | Utilise la classe `Quantity` pour garantir la cohérence dimensionnelle des entrées. C'est une pratique essentielle en physique et en ingénierie pour éviter les erreurs de conversion et rendre le code indépendant du système d'unités utilisé (SI, impérial, etc.). |
| **`utilities.dictonnary`** | **Manipulation des Dictionnaires OpenFOAM** | Fournit des outils pour créer et manipuler des structures de données complexes qui correspondent aux fichiers de dictionnaire OpenFOAM (ex: `topoSetDict`, `createPatchDict`). |
| **`commons`** | **Utilitaires Génériques** | Contient des fonctions pour la sérialisation des classes, la lecture des fichiers de maillage (`polyMesh/boundary`), et d'autres opérations de bas niveau nécessaires à l'interfaçage avec le format de fichier OpenFOAM. |

Ce document fournit une vue d'ensemble de la manière dont `foampilot` structure et gère les concepts fondamentaux de la simulation OpenFOAM, en les traduisant en une architecture Python modulaire et intuitive.