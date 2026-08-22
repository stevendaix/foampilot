# Couplage YADE–OpenFOAM 13 mis en place dans FoamPilot

**Document d’implémentation et d’exploitation**
**Système validé : Ubuntu 24.04 — OpenFOAM Foundation 13 — YADE `yadedaily` — FoamPilot**

## 1. Résumé de l’implémentation

Le couplage intégré dans FoamPilot est un couplage **Euler–Lagrange CFD–DEM bidirectionnel**. YADE calcule les particules et leurs contacts; OpenFOAM calcule l’écoulement fluide. Les deux solveurs s’exécutent comme processus distincts et échangent les données par MPI. La méthode `FoamCoupling` de YADE diffuse les positions, vitesses, vitesses angulaires et rayons des particules vers OpenFOAM; les processus OpenFOAM localisent ensuite les particules dans leur maillage local, évaluent les forces hydrodynamiques et renvoient ces forces à YADE [1].

L’intégration ne remplace pas YADE par un modèle simplifié et ne contourne pas OpenFOAM. Elle vendore les sources du couplage dans `third_party/yade-openfoam-coupling`, les porte vers les API OpenFOAM 13 et ajoute deux cas exécutables dans `validation/yade-openfoam13`.

| Élément | Emplacement | Fonction |
|---|---|---|
| Sources portées | `third_party/yade-openfoam-coupling/` | bibliothèques et solveurs compatibles OF13 |
| Communication | `FoamYade/commYade/` | échanges MPI et données YADE–OF |
| Recherche de cellule | `FoamYade/meshtree/` | localisation des particules dans le maillage |
| Moteur de couplage | `FoamYade/` | force hydrodynamique, interpolation et retour d’action |
| Solveur ponctuel | `icoFoamYade/` | couplage de type point-force |
| Solveur volumique | `pimpleFoamYade/` | fraction fluide/solide et interpolation gaussienne |
| Exemple ponctuel | `validation/yade-openfoam13/icoFoamYade/` | 1000 sphères, cas MPI |
| Exemple volumique | `validation/yade-openfoam13/pimpleFoamYade/` | 2000 sphères, cas MPI |

## 2. Couplages externes : options et comparaison

Un couplage externe signifie que le solveur CFD et le solveur DEM restent deux programmes séparés. Ils peuvent communiquer par fichiers, sockets, MPI direct ou bibliothèque intermédiaire. Le choix détermine le coût de synchronisation, la robustesse du protocole et la quantité de code à maintenir.

| Stratégie | Communication | Avantages | Limites | Pertinence ici |
|---|---|---|---|---|
| Fichiers à chaque pas | fichiers texte ou binaires | simple à inspecter et à déboguer | très lent, risques de fichiers partiels et de désynchronisation | utile pour un prototype, non retenu |
| TCP/UDP ou sockets | réseau local ou distant | découplage des processus et des machines | protocole, latence, reprise et ordre des messages à gérer | possible, mais inutile sur un nœud MPI |
| MPI direct | communicateurs et messages MPI | rapide, adapté au HPC, synchronisation explicite | initialisation/arrêt très sensibles, compatibilité ABI obligatoire | **retenu pour YADE–OpenFOAM** |
| Bibliothèque de couplage | preCICE, MUI ou équivalent | interpolation, échanges et convergence parfois mutualisés | dépendance et adaptation d’un nouvel adaptateur | pertinent pour multi-physique générale |
| Couplage interne au solveur | code DEM dans le même exécutable CFD | latence faible et contrôle direct | compilation monolithique, maintenance et parallélisme plus complexes | non retenu |
| CFDEM | OpenFOAM + LIGGGHTS | solution CFD–DEM mature autour de LIGGGHTS [4] | dépendance à LIGGGHTS et compatibilités de versions | abandonné dans ce projet au profit de YADE |
| Adaptateur preCICE | OpenFOAM comme participant via un function object [3] | architecture générique, multi-solveurs et multi-physique | nécessite un adaptateur DEM et le mapping correspondant | alternative future, pas le backend actuel |

Le couplage installé utilise donc **MPI direct**, car YADE fournit déjà `FoamCoupling` et ses routines de communication. La documentation YADE indique que les données sont envoyées aux processus OpenFOAM et que la force opposée est réinjectée dans l’équation de quantité de mouvement [1]. À l’inverse, preCICE propose une couche générique de couplage et un adaptateur OpenFOAM sous forme de function object [3]; il faudrait cependant développer ou sélectionner un participant DEM YADE compatible avec les maillages, les champs et les conventions de transfert utilisés ici.

Un couplage externe doit toujours définir cinq contrats : les variables échangées, l’unité de chaque variable, le moment de synchronisation, le rang propriétaire d’une particule et la procédure de terminaison. Une implémentation qui ne documente que les forces mais pas la fermeture MPI est incomplète.

## 3. Architecture et séquence exacte

Le script `scriptMPI.py` construit la scène YADE et configure `FoamCoupling`. L’appel `fluidCoupling.SetOpenFoamSolver("icoFoamYade", numProcOF)` ou `SetOpenFoamSolver("pimpleFoamYade", numProcOF)` sélectionne le solveur et le nombre de processus OpenFOAM. L’appel `mp.mpirun(NSTEPS)` lance la boucle d’intégration YADE et la synchronisation avec OpenFOAM.

À chaque échange, le protocole suit cette logique :

1. YADE met à jour les contacts, les forces mécaniques et l’état cinématique des particules.
2. Les données particulaires sont diffusées vers le monde OpenFOAM.
3. Chaque rang OpenFOAM détermine les particules qui appartiennent à son sous-maillage.
4. La vitesse fluide et les grandeurs nécessaires sont interpolées à la position de chaque particule.
5. La force de traînée et, pour le modèle volumique, les forces complémentaires sont calculées.
6. Les forces hydrodynamiques sont retournées à YADE et ajoutées à la dynamique DEM.
7. L’action opposée est projetée sur les cellules OpenFOAM comme source de quantité de mouvement.
8. OpenFOAM résout pression et vitesse, puis les deux solveurs passent au temps suivant.

L’ordre des moteurs YADE est important. `GlobalStiffnessTimeStepper` précède `FoamCoupling`, puis `NewtonIntegrator` intègre la réponse mécanique. Les listes d’identifiants sont construites avec `sphereIDs = [b.id for b in O.bodies if type(b.shape) == Sphere]`.

## 4. Théorie DEM

Pour chaque particule `p`, la translation et la rotation suivent

\[
 m_p\frac{d\mathbf{U}_p}{dt}
 =\sum\mathbf{F}_{contact,p}+\mathbf{F}_{hyd,p}+m_p\mathbf{g},
\]

\[
 \mathbf{I}_p\frac{d\boldsymbol{\omega}_p}{dt}
 +\boldsymbol{\omega}_p\times(\mathbf{I}_p\boldsymbol{\omega}_p)
 =\sum\mathbf{T}_{contact,p}+\mathbf{T}_{hyd,p}.
\]

Pour une sphère de diamètre `d_p`,

\[
V_p=\frac{\pi d_p^3}{6},\qquad m_p=\rho_pV_p.
\]

Les cas fournis utilisent `FrictMat`, des interactions sphère–sphère et boîte–sphère, une loi de contact de type Cundall–Strack et des boîtes fixes. Le pas DEM est déterminé par la rigidité, la masse et la stabilité du contact. Il est dangereux de choisir `deltaT` uniquement à partir du pas CFD.

## 5. Théorie CFD–DEM ponctuelle : icoFoamYade

Le solveur ponctuel considère les particules comme des sources localisées dans le fluide. Pour un fluide incompressible,

\[
\nabla\cdot\mathbf{U}_f=0,
\]

\[
\frac{\partial\mathbf{U}_f}{\partial t}
+\nabla\cdot(\mathbf{U}_f\mathbf{U}_f)
=-\frac{\nabla p}{\rho_f}
+\nabla\cdot\boldsymbol{\tau}
+\mathbf{f}_{h}.
\]

Dans le régime de Stokes, la traînée d’une sphère est

\[
\mathbf{F}_{drag}=3\pi\mu d_p(\mathbf{U}_f-\mathbf{U}_p).
\]

Le nombre de Reynolds particulaire est

\[
Re_p=\frac{\rho_f\lvert\mathbf{U}_f-\mathbf{U}_p\rvert d_p}{\mu}.
\]

L’approximation ponctuelle est destinée aux particules plus petites que les longueurs résolues; la documentation YADE associe la formulation de Stokes au régime `Re_p<1` [1]. Dans la cellule `c`, la force opposée peut être écrite sous la forme

\[
\mathbf{f}_{h,c}=-\frac{\mathbf{F}_{h}}{V_c\rho_f}.
\]

Le rapport entre diamètre particulaire et taille de cellule doit donc être contrôlé avant toute interprétation quantitative.

## 6. Théorie moyennée en volume : pimpleFoamYade

Le modèle volumique introduit les fractions solide et fluide. Avec `\epsilon_f=1-\epsilon_s`,

\[
\frac{\partial\epsilon_f}{\partial t}
+\nabla\cdot(\epsilon_f\mathbf{U}_f)=0,
\]

\[
\frac{\partial(\epsilon_f\mathbf{U}_f)}{\partial t}
+\nabla\cdot(\epsilon_f\mathbf{U}_f\mathbf{U}_f)
=-\frac{\nabla p}{\rho_f}
+\epsilon_f\nabla\cdot\boldsymbol{\tau}
-K(\mathbf{U}_f-\mathbf{U}_p)+\mathbf{S}_u+\epsilon_f\mathbf{g}.
\]

La traînée est basée sur Schiller–Naumann :

\[
K=\frac{3}{4}C_d\frac{\rho_f}{d_p}
\lvert\widetilde{\mathbf{U}}_f-\mathbf{U}_p\rvert
\epsilon_f^{-h_{exp}},
\qquad
C_d=\frac{24}{Re_p}(1+0.15Re_p^{0.687}),
\]

avec `h_exp=2.65` dans la formulation documentée par YADE [1]. L’interpolation gaussienne calcule la fraction solide par

\[
\epsilon_{s,c}=\frac{\sum_iV_{p,i}G_{*,i,c}}{V_c},
\]

et une grandeur eulérienne au centre particulaire par

\[
\widetilde{\phi}_p=\sum_c\phi_cG_{*,c,p}.
\]

Dans le dépôt, `pimpleFoamYade` active `isGaussianInterp=True`. Cette option doit être considérée comme un modèle à vérifier pour chaque régime; la documentation YADE la signale comme étant en développement actif [1].

## 7. Installation complète

L’installation commence par l’environnement OpenFOAM :

```bash
source /opt/openfoam13/etc/bashrc
```

Vérifier ensuite YADE :

```bash
yadedaily --version
yadedaily-batch --help
yadedaily -x -c 'from yade import mpy as mp; print(mp)'
```

Compiler le couplage porté :

```bash
cd /home/ubuntu/work/foampilot/third_party/yade-openfoam-coupling
source /opt/openfoam13/etc/bashrc
./Allwmake
```

Puis vérifier les exécutables :

```bash
export FOAM_USER_APPBIN=/home/ubuntu/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin
ls -l "$FOAM_USER_APPBIN"/icoFoamYade "$FOAM_USER_APPBIN"/pimpleFoamYade
```

OpenFOAM 13 doit être utilisé avec ses propres chemins et bibliothèques. Il ne faut pas mélanger `OpenFOAM Foundation` et `openfoam.com` dans le même shell, ni charger successivement plusieurs `bashrc` incompatibles.

## 8. Exemples exécutables

### 8.1 Cas icoFoamYade

```bash
source /opt/openfoam13/etc/bashrc
export FOAM_USER_APPBIN=/home/ubuntu/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin
cd /home/ubuntu/work/foampilot/validation/yade-openfoam13/icoFoamYade
CFDEM_NSTEPS=20 OPENFOAM_PROCS=2 YADE_PARALLEL=false ./run.sh
```

Le cas contient 1000 sphères, une scène périodique, des murs `Box`, un couplage ponctuel et `isGaussianInterp=False`. Pour une exécution longue, omettre `CFDEM_NSTEPS=20` et utiliser la valeur par défaut du script.

### 8.2 Cas pimpleFoamYade

```bash
source /opt/openfoam13/etc/bashrc
export FOAM_USER_APPBIN=/home/ubuntu/OpenFOAM/root-13/platforms/$WM_OPTIONS/bin
cd /home/ubuntu/work/foampilot/validation/yade-openfoam13/pimpleFoamYade
CFDEM_NSTEPS=20 OPENFOAM_PROCS=2 YADE_PARALLEL=false ./run.sh
```

Le cas contient 2000 sphères, une boîte périodique avec parois `yminus` et `yplus`, le solveur PIMPLE et l’interpolation gaussienne. Le journal principal produit par YADE est `scriptMPI.py.default.log`.

Les scripts acceptent `CFDEM_NSTEPS`, `OPENFOAM_PROCS` et `YADE_PARALLEL`. Les lanceurs effectuent `blockMesh`, copient `0_org` vers `0`, exécutent `decomposePar` et contrôlent le statut réel du job YADE au lieu de se fier uniquement au code retour de `yadedaily-batch`.

## 9. Erreurs à ne pas faire

| Erreur | Symptôme | Correction |
|---|---|---|
| Lancer le script avec `mpirun` externe alors que `mp.mpirun()` est utilisé | double initialisation ou deadlock MPI | lancer seulement `./run.sh` |
| Mélanger `YADE_PARALLEL=false` avec `parallelMode=True` ou `DOMAIN_DECOMPOSITION=True` | segfault dans `PMPI_Allreduce` | utiliser le même booléen pour toutes les options parallèles |
| Oublier `source /opt/openfoam13/etc/bashrc` | solveur introuvable ou bibliothèques incompatibles | charger l’environnement avant `run.sh` |
| Oublier `blockMesh` ou `decomposePar` | maillage absent, échec de localisation | utiliser le lanceur fourni |
| Utiliser `isGaussianInterp=True` avec `icoFoamYade` | modèle d’interpolation incompatible | réserver l’option au cas `pimpleFoamYade` |
| Laisser une particule sortir du domaine fluide | erreur de recherche ou arrêt du couplage | synchroniser frontières YADE/OpenFOAM |
| Utiliser `d_p` très supérieur à `\Delta x` en modèle ponctuel | source localisée non résolue et résultats biaisés | raffiner ou changer de modèle |
| Ignorer les unités | forces et temps incohérents | vérifier SI, `rho`, `mu`, diamètre, rigidité et gravité |
| Juger la réussite sur le seul code de `yadedaily-batch` | un job interne peut être marqué FAILED malgré un code externe nul | lire `scriptMPI.py.default.log` |
| Réactiver `CFDEM_KILL_MPI=true` sans raison | `MPI_ABORT ... errorcode -100` en fin de calcul | conserver la fermeture normale par défaut |

## 10. Problème MPI de fermeture et solution

La version installée de `FoamCoupling::killMPI()` appelle directement `MPI_Abort(communicator,-100)`. C’est la raison du message final `MPI_ABORT was invoked on rank 0`. Il ne s’agit pas d’une divergence du solveur et les pas précédents peuvent être physiquement valides.

Le code actuel rend l’appel optionnel :

```python
mp.mpirun(NSTEPS)
mp.mprint("RUN FINISH")

if os.environ.get('CFDEM_KILL_MPI', 'false').lower() in ('1', 'true', 'yes'):
    fluidCoupling.killMPI()
```

La valeur par défaut `false` permet la terminaison normale du processus YADE après `RUN FINISH`. La valeur `true` est conservée uniquement pour reproduire l’ancien comportement et diagnostiquer une compatibilité de version.

## 11. Validation et critères d’acceptation

Une validation minimale doit prouver simultanément que le maillage est créé, que le solveur OpenFOAM démarre, que les pas avancent, que les données sont échangées, que la continuité reste contrôlée, que YADE écrit `RUN FINISH`, que le statut interne est `0 (OK)` et qu’aucun processus MPI ne reste orphelin.

Les smoke tests versionnés ont été exécutés avec 20 pas et deux processus OpenFOAM. `icoFoamYade` et `pimpleFoamYade` ont tous deux terminé avec `launcher_rc=0`, `status : 0 (OK)`, `RUN FINISH` et sans `MPI_ABORT`. Ces essais valident le fonctionnement logiciel du couplage; une validation scientifique de production doit en plus comparer une grandeur à une solution analytique ou expérimentale et réaliser des études de maillage, pas de temps et paramètres physiques.

## 12. Fichiers importants

| Fichier | Description |
|---|---|
| `third_party/yade-openfoam-coupling/FoamYade/FoamYade.C` | moteur de couplage OpenFOAM porté |
| `third_party/yade-openfoam-coupling/FoamYade/commYade/yadeComm.C` | communication et données MPI |
| `third_party/yade-openfoam-coupling/FoamYade/meshtree/meshTree.C` | recherche dans le maillage |
| `third_party/yade-openfoam-coupling/icoFoamYade/icoFoamYade.C` | solveur ponctuel |
| `third_party/yade-openfoam-coupling/pimpleFoamYade/pimpleFoamYade.C` | solveur volumique |
| `validation/yade-openfoam13/icoFoamYade/scriptMPI.py` | scène et lancement YADE ponctuel |
| `validation/yade-openfoam13/pimpleFoamYade/scriptMPI.py` | scène et lancement YADE volumique |
| `validation/yade-openfoam13/*/run.sh` | préparation et exécution du cas |
| `validation/yade-openfoam13/*/Allclean` | nettoyage compatible OF13 |
| `validation/yade-openfoam13/test-logs/MPI_DIAGNOSTIC.md` | diagnostic détaillé de l’arrêt MPI |

## 13. Références

[1]: https://yade-dem.org/doc/FoamCoupling.html "YADE — CFD-DEM coupled simulations with Yade and OpenFOAM"
[2]: https://yade-dem.org/doc/yade.wrapper.html "YADE — wrapper class reference"
[3]: https://precice.org/adapter-openfoam-overview "preCICE — OpenFOAM adapter"
[4]: https://www.cfdem.com/media/CFDEM/docu/CFDEMcoupling_Manual.html "CFDEM coupling manual"
[5]: https://openfoam.org/version/13/ "OpenFOAM Foundation — OpenFOAM 13"
[6]: https://doc.cfd.direct/openfoam/user-guide-v13/ "OpenFOAM Foundation — OpenFOAM v13 User Guide"
