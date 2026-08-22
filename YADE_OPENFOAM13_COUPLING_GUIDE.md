# Couplage CFD–DEM YADE–OpenFOAM 13 dans FoamPilot

**Auteur : Manus AI**
**Plateforme validée : Ubuntu 24.04, OpenFOAM Foundation 13, YADE `yadedaily`, Python 3.12**

## 1. Objet et périmètre

Ce document décrit l’intégration du couplage Euler–Lagrange entre **YADE**, utilisé comme solveur discret (DEM), et **OpenFOAM 13**, utilisé comme solveur fluide (CFD), dans le dépôt FoamPilot. Le code n’est pas une approximation logicielle de démonstration : les sources `FoamCoupling`, `libYadeComm`, `libMeshTree`, `libYadeFoam`, ainsi que les solveurs `icoFoamYade` et `pimpleFoamYade`, sont compilés et exécutés dans les cas de validation versionnés.

La documentation YADE décrit ce couplage comme un échange MPI dans lequel YADE diffuse la position, la vitesse, la vitesse angulaire et le rayon des particules aux processus OpenFOAM; chaque processus recherche les particules dans son maillage local, calcule les forces hydrodynamiques, puis retourne les forces à YADE et réinjecte l’action opposée dans l’équation de quantité de mouvement du fluide [1].

| Composant | Rôle | Version validée |
|---|---|---|
| YADE | Intégration DEM, contacts, matériaux, mouvement des particules | paquet officiel `yadedaily` pour Noble |
| OpenFOAM | Maillage, discrétisation volumes finis, pression et vitesse | Foundation 13 |
| `FoamCoupling` | Échange des données et calcul des forces fluide–particule | port local OF13 |
| `icoFoamYade` | Couplage ponctuel, écoulement incompressible transitoire | validé |
| `pimpleFoamYade` | Couplage moyenné en volume avec interpolation gaussienne | validé |
| FoamPilot | Orchestration et cas d’exemple | branche `feat/yade-openfoam13-coupling` |

## 2. Architecture d’exécution

Le couplage utilise un communicateur MPI intercommunicateur entre le monde YADE et les processus OpenFOAM. Le script Python YADE construit la scène, instancie `FoamCoupling`, définit le solveur OpenFOAM et appelle `mp.mpirun(NSTEPS)`. Le solveur OpenFOAM est démarré par le moteur de couplage; il décompose son domaine, charge le maillage et participe ensuite à l’échange à chaque pas.

La séquence logique d’un pas est la suivante :

1. YADE calcule les contacts et prépare les propriétés cinématiques des particules.
2. Les données des particules sont transmises aux processus OpenFOAM.
3. Chaque rang OpenFOAM localise les particules dans son sous-maillage.
4. Le champ fluide est interpolé à la position des particules.
5. OpenFOAM calcule la traînée et, selon le modèle, les contributions de pression, de contrainte visqueuse et de masse ajoutée.
6. Les forces hydrodynamiques sont renvoyées à YADE et ajoutées au conteneur de forces des particules.
7. L’opposé de l’action exercée sur les particules est distribué sur les cellules fluides comme source de quantité de mouvement.
8. OpenFOAM résout pression et vitesse; YADE avance vers le pas suivant.

L’ordre des moteurs YADE est essentiel : le `GlobalStiffnessTimeStepper` doit précéder `FoamCoupling`, afin que le pas DEM transmis au couplage soit disponible. L’activation de la décomposition de domaine et du mode parallèle doit rester cohérente avec `parallelMode` du timestepper; une incohérence peut provoquer un `PMPI_Allreduce` invalide dans YADE.

## 3. Théorie DEM

Dans la méthode des éléments discrets, chaque particule est une entité lagrangienne possédant une position, une vitesse, une rotation, une masse et une inertie. Pour une particule `p`, les équations de translation et de rotation sont

\[
 m_p\frac{d\mathbf{U}_p}{dt}=\sum\mathbf{F}_{c,p}+\mathbf{F}_{hyd,p}+m_p\mathbf{g},
\]

\[
 \mathbf{I}_p\frac{d\boldsymbol{\omega}_p}{dt}+\boldsymbol{\omega}_p\times(\mathbf{I}_p\boldsymbol{\omega}_p)=\sum\mathbf{T}_{c,p}+\mathbf{T}_{hyd,p}.
\]

Les forces de contact sont produites par la géométrie d’interaction, le modèle de matériau et la loi de contact. Dans les cas fournis, les sphères utilisent `FrictMat`, une loi normale tangentielle de type Cundall–Strack et des murs fixes `Box`. Le pas DEM est contrôlé par la rigidité et la stabilité du contact; il ne doit pas être choisi uniquement à partir du pas CFD.

Pour une sphère, le volume et la masse valent

\[
 V_p=\frac{\pi d_p^3}{6},\qquad m_p=\rho_pV_p.
\]

Le couplage hydrodynamique agit dans l’équation de translation et respecte le principe d’action-réaction : une force reçue par la particule est réinjectée avec signe opposé dans le fluide.

## 4. Théorie CFD et couplage ponctuel

Pour `icoFoamYade`, les particules sont modélisées comme des points de force sur la grille fluide. La continuité incompressible s’écrit

\[
\nabla\cdot\mathbf{U}_f=0,
\]

et l’équation de quantité de mouvement sous forme normalisée peut s’écrire

\[
\frac{\partial\mathbf{U}_f}{\partial t}
+\nabla\cdot(\mathbf{U}_f\mathbf{U}_f)
=-\frac{\nabla p}{\rho_f}+\nabla\cdot\boldsymbol{\tau}
+\mathbf{f}_{h}.
\]

Dans le régime de Stokes, la force de traînée sur une sphère est

\[
\mathbf{F}_{drag}=3\pi\mu d_p(\mathbf{U}_f-\mathbf{U}_p).
\]

La validité de cette approximation doit être vérifiée par le nombre de Reynolds particulaire

\[
 Re_p=\frac{\rho_f\lvert\mathbf{U}_f-\mathbf{U}_p\rvert d_p}{\mu}.
\]

La documentation YADE recommande le modèle ponctuel lorsque les particules sont plus petites que les longueurs caractéristiques résolues et, pour la formulation de Stokes, dans un régime `Re_p < 1` [1]. La force opposée est convertie en terme volumique dans la cellule `c` contenant la particule, sous une forme du type

\[
\mathbf{f}_{h,c}=-\frac{\mathbf{F}_{h}}{V_c\rho_f}.
\]

Cette formulation exige que le diamètre particulaire soit inférieur à la taille caractéristique des cellules si l’on veut éviter une interprétation sous-résolue incohérente.

## 5. Théorie du couplage moyenné en volume

Pour `pimpleFoamYade`, la fraction volumique solide est prise en compte. Avec `\epsilon_f=1-\epsilon_s`, l’équation de continuité moyennée est

\[
\frac{\partial\epsilon_f}{\partial t}
+\nabla\cdot(\epsilon_f\mathbf{U}_f)=0,
\]

et l’équation de quantité de mouvement prend la forme

\[
\frac{\partial(\epsilon_f\mathbf{U}_f)}{\partial t}
+\nabla\cdot(\epsilon_f\mathbf{U}_f\mathbf{U}_f)
=-\frac{\nabla p}{\rho_f}
+\epsilon_f\nabla\cdot\boldsymbol{\tau}
-K(\mathbf{U}_f-\mathbf{U}_p)
+\mathbf{S}_u+\epsilon_f\mathbf{g}.
\]

Le coefficient de traînée utilisé par l’implémentation est basé sur Schiller–Naumann :

\[
K=\frac{3}{4}C_d\frac{\rho_f}{d_p}
\lvert\widetilde{\mathbf{U}}_f-\mathbf{U}_p\rvert\epsilon_f^{-h_{exp}},
\]

avec `h_exp=2.65` dans la documentation YADE, et

\[
C_d=\frac{24}{Re_p}\left(1+0.15Re_p^{0.687}\right).
\]

La force de traînée particulaire est donc `\mathbf{F}_{drag}=V_pK(\widetilde{\mathbf{U}}_f-\mathbf{U}_p)`. Le terme explicite `\mathbf{S}_u` regroupe notamment la force d’Archimède/ambiante, la masse ajoutée et les contributions de pression et de contrainte visqueuse [1]. Cette option doit être considérée avec prudence dans une étude de production, car la documentation YADE la signale comme en développement actif [1].

## 6. Interpolation et moyenne gaussienne

L’interpolation gaussienne utilise une enveloppe `G_*` autour de chaque particule. Pour un centre de cellule `x_c` et une position de particule `x_p`, la pondération est

\[
G_*(\mathbf{x}_c-\mathbf{x}_p)
=(2\pi\sigma^2)^{-3/2}
\exp\left[-\frac{\lvert\mathbf{x}_c-\mathbf{x}_p\rvert^2}{2\sigma^2}\right],
\]

avec une largeur déterminée par la distance de coupure `\delta`, généralement liée à quelques tailles de cellule. La fraction solide dans la cellule est calculée par

\[
\epsilon_{s,c}=\frac{\sum_iV_{p,i}G_{*,i,c}}{V_c},
\]

puis `\epsilon_{f,c}=1-\epsilon_{s,c}`. Une grandeur eulérienne `\phi` interpolée vers la particule s’écrit

\[
\widetilde{\phi}_p=\sum_c\phi_cG_{*,c,p}.
\]

L’option `isGaussianInterp=True` est utilisée avec `pimpleFoamYade` dans le cas correspondant. Pour `icoFoamYade`, le cas fourni utilise l’interpolation ponctuelle (`isGaussianInterp=False`).

## 7. Installation reproductible

Il faut d’abord charger l’environnement OpenFOAM Foundation 13 dans chaque shell :

```bash
source /opt/openfoam13/etc/bashrc
```

L’installation YADE retenue est le paquet officiel quotidien pour Ubuntu 24.04. Après installation, vérifier que l’exécutable et l’import Python sont disponibles :

```bash
yadedaily --version
yadedaily-batch --help
yadedaily -x -c 'from yade import mpy as mp; print(mp)'
```

Dans le dépôt FoamPilot, compiler les bibliothèques et les solveurs portés vers OpenFOAM 13 :

```bash
cd /home/ubuntu/work/foampilot/third_party/yade-openfoam-coupling
source /opt/openfoam13/etc/bashrc
./Allwmake
```

La compilation doit produire les bibliothèques de communication, d’arbre de maillage et de couplage, ainsi que `icoFoamYade` et `pimpleFoamYade`. Vérifier ensuite leur présence dans `$FOAM_USER_APPBIN`.

OpenFOAM 13 a modifié plusieurs interfaces par rapport aux versions historiques utilisées par le couplage, notamment les en-têtes, la recherche de maillage, les dimensions de viscosité cinématique et les chemins de modèles de transport. Ces adaptations sont incluses dans `third_party/yade-openfoam-coupling` [4].

## 8. Lancement des cas FoamPilot

Les cas sont situés dans `validation/yade-openfoam13/icoFoamYade` et `validation/yade-openfoam13/pimpleFoamYade`. Un lancement reproductible de smoke test est :

```bash
source /opt/openfoam13/etc/bashrc
export FOAM_USER_APPBIN=/home/ubuntu/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin
cd /home/ubuntu/work/foampilot/validation/yade-openfoam13/icoFoamYade
CFDEM_NSTEPS=20 OPENFOAM_PROCS=2 YADE_PARALLEL=false ./run.sh
```

Pour le second cas, remplacer le répertoire par `pimpleFoamYade`. Les variables disponibles sont les suivantes :

| Variable | Défaut | Fonction |
|---|---:|---|
| `CFDEM_NSTEPS` | 5000 ou 408 selon le cas | nombre de pas YADE/CFD |
| `OPENFOAM_PROCS` | 2 | nombre de processus OpenFOAM |
| `YADE_PARALLEL` | `true` | active les options parallèles cohérentes de YADE |
| `CFDEM_KILL_MPI` | `false` | réactive l’ancien arrêt forcé `MPI_Abort(-100)` |

Le lanceur exécute `blockMesh`, copie `0_org` vers `0`, lance `decomposePar`, crée le répertoire des sphères puis démarre `yadedaily-batch`. Le journal détaillé YADE est `scriptMPI.py.default.log`.

## 9. Validation effectuée

Les deux cas ont été lancés avec 20 pas, deux processus OpenFOAM et le mode YADE série cohérent avec le timestepper. Les critères observés sont : démarrage du solveur, progression de `Time = ...`, résidus OpenFOAM, erreurs de continuité, `Master: RUN FINISH`, statut YADE `0 (OK)`, code retour du lanceur nul et absence de processus orphelins.

| Cas | Pas | Processus OF | Statut YADE | `RUN FINISH` | `MPI_ABORT` | Résultat |
|---|---:|---:|---|---|---|---|
| `icoFoamYade` | 20 | 2 | `0 (OK)` | présent | absent | PASS |
| `pimpleFoamYade` | 20 | 2 | `0 (OK)` | présent | absent | PASS |

Pour `pimpleFoamYade`, les journaux montrent des erreurs globales de continuité de l’ordre de `10^{-26}` sur le smoke test, ainsi que des nombres de Courant très faibles dans cette configuration. Ces valeurs valident l’échange et la stabilité de ce cas précis; elles ne remplacent pas une étude de sensibilité au maillage, au pas de temps et aux paramètres physiques.

## 10. Diagnostic de l’erreur MPI

Avec l’ancien script, les deux simulations avançaient jusqu’à la fin, puis produisaient :

```text
MPI_ABORT was invoked on rank 0 in communicator <Unknown>
with errorcode -100.
```

L’analyse de la bibliothèque réellement chargée par `yadedaily` a établi que `FoamCoupling::killMPI()` n’effectue pas une finalisation MPI normale. Le désassemblage de `/usr/lib/x86_64-linux-gnu/yadedaily/libpkg_openfoam.so` montre :

```text
mov 0xd8(%rdi),%rdi
mov $0xffffff9c,%esi
jmp MPI_Abort@plt
```

`0xffffff9c` correspond à `-100`. L’erreur finale ne vient donc ni d’un résidu CFD divergent ni d’un défaut d’échange pendant les pas : elle vient de l’appel explicite à `fluidCoupling.killMPI()` dans les scripts.

Le correctif retenu est de rendre cet appel optionnel et désactivé par défaut. Après `RUN FINISH`, le processus YADE se termine normalement et libère ses ressources MPI au niveau du processus. Le comportement historique peut être reproduit avec `CFDEM_KILL_MPI=true`, mais il doit être interprété comme un arrêt forcé et non comme un échec physique.

Un premier essai avait aussi déclenché un segfault dans `GlobalStiffnessTimeStepper::computeTimeStep` sur `PMPI_Allreduce`. La cause était une configuration incohérente : `YADE_PARALLEL=false` mais `parallelMode=True` et `DOMAIN_DECOMPOSITION=True`. Les scripts utilisent maintenant la même valeur `parallelYade` pour ces trois choix, ce qui supprime cette incohérence.

## 11. Limites et recommandations de production

La validation actuelle démontre l’initialisation, l’échange, l’avancement et la fermeture propre des deux cas. Elle ne constitue pas encore une validation quantitative contre une solution analytique ou une expérience. Pour une campagne de production, il faut ajouter une étude de convergence en maillage et en pas de temps, vérifier `Re_p`, `St_k`, la résolution `d_p/\Delta x`, la fraction solide maximale, le bilan global de quantité de mouvement et la conservation de l’énergie mécanique des contacts.

Il faut également s’assurer que les frontières YADE et OpenFOAM sont compatibles et qu’aucune particule ne sort du domaine fluide. La documentation YADE indique qu’une particule hors du domaine local peut interrompre le couplage [1]. Le mode gaussien doit être utilisé avec prudence dans les régimes fortement denses ou lorsque la largeur du filtre est mal résolue.

## 12. Couplages externes CFD–DEM

Un couplage externe conserve le solveur CFD et le solveur DEM dans des processus séparés. Les données peuvent circuler par fichiers, sockets, MPI direct ou bibliothèque intermédiaire. Le choix doit préciser les variables transférées, les unités, le rythme de synchronisation, le propriétaire MPI de chaque particule et le protocole d’arrêt.

| Méthode | Points forts | Risques ou limites | Choix dans ce projet |
|---|---|---|---|
| Fichiers | inspection facile, mise au point simple | très lent, fichiers partiels, désynchronisation | non retenue |
| Sockets TCP | processus séparés, machines différentes possibles | latence et protocole applicatif à gérer | non retenue |
| MPI direct | débit élevé, adapté au HPC et à OpenFOAM | communicateurs, rangs et fermeture sensibles | **retenue** |
| preCICE | bibliothèque générique, mapping et convergence multi-physique [6] | adaptateur DEM YADE à fournir | alternative future |
| MUI | interface multi-physique asynchrone possible | dépendance et mapping supplémentaires | non retenue |
| CFDEM | interface OpenFOAM–LIGGGHTS documentée [7] | dépendance à LIGGGHTS et versions associées | abandonnée au profit de YADE |

Le projet utilise MPI direct via `FoamCoupling`, qui existe déjà dans YADE et diffuse les données particulaires aux processus OpenFOAM [1]. Un adaptateur preCICE OpenFOAM est une autre architecture possible, mais il faudrait définir un participant YADE, les champs d’interface et la projection des forces [6]. Le choix MPI évite ici une couche supplémentaire et conserve la logique de couplage fournie par YADE.

## 13. Références

[1]: https://yade-dem.org/doc/FoamCoupling.html "YADE — CFD-DEM coupled simulations with Yade and OpenFOAM"
[2]: https://yade-dem.org/doc/yade.wrapper.html "YADE — wrapper class reference"
[3]: https://gitlab.com/yade-dev/Yade-OpenFOAM-coupling "YADE — Yade OpenFOAM Coupling source repository"
[4]: https://openfoam.org/version/13/ "OpenFOAM Foundation — OpenFOAM 13"
[5]: https://doc.cfd.direct/openfoam/user-guide-v13/ "OpenFOAM Foundation — OpenFOAM v13 User Guide"
[6]: https://precice.org/adapter-openfoam-overview "preCICE — OpenFOAM adapter"
[7]: https://www.cfdem.com/media/CFDEM/docu/CFDEMcoupling_Manual.html "CFDEM coupling manual"
