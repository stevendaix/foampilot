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
| Bibliothèque de couplage | preCICE, MUI ou équivalent | interpolation, échanges et convergence parfois mutualisés [3] [5] | dépendance et adaptation d’un nouvel adaptateur | pertinent pour multi-physique générale |
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

## 13. Interpolation gaussienne : formulation détaillée

### 13.1 Pourquoi interpoler sur plusieurs cellules

Les particules YADE sont décrites par des variables lagrangiennes, alors que les inconnues OpenFOAM sont stockées sur les cellules ou les faces d’un maillage eulérien. La vitesse `U_f(x_p)` n’est donc pas connue directement au centre d’une particule. Réciproquement, une force calculée sur une particule ponctuelle ne peut pas être injectée dans une équation volumique sans définir une règle de distribution sur les cellules.

L’interpolation gaussienne joue ces deux rôles. Elle transforme un champ de cellule en une grandeur vue par la particule et transforme une propriété particulaire en contribution distribuée sur un voisinage compact de cellules. Cette régularisation évite une source strictement concentrée dans une seule cellule, mais introduit une longueur de filtrage : la largeur du filtre devient une partie du modèle physique et numérique.

### 13.2 Noyau continu et noyau discret

Le noyau continu utilisé par la documentation YADE est

\[
G_\star(\mathbf{x}_c-\mathbf{x}_p)
=\frac{1}{(2\pi\sigma^2)^{3/2}}
\exp\left[-\frac{\lVert\mathbf{x}_c-\mathbf{x}_p\rVert^2}{2\sigma^2}\right].
\]

La quantité `G_*` possède une dimension inverse d’un volume dans sa définition continue. En pratique, le code ne travaille pas directement avec une intégrale analytique sur tout l’espace. Il sélectionne un ensemble fini de cellules voisines autour de la particule, calcule un poids pour chacune, puis renormalise les poids sur cet ensemble :

\[
 w_{pc}^{raw}=C_\sigma
 \exp\left[-\frac{\lVert\mathbf{x}_c-\mathbf{x}_p\rVert^2}{2\,\texttt{interpRange}^2}\right],
\]

\[
 w_{pc}=\frac{w_{pc}^{raw}}{\sum_{j\in\mathcal{N}(p)}w_{pj}}.
\]

Ainsi,

\[
\sum_{c\in\mathcal{N}(p)}w_{pc}=1.
\]

Cette normalisation discrète est essentielle : elle empêche une particule de recevoir une vitesse artificiellement réduite ou amplifiée uniquement parce qu’elle se trouve près d’une frontière du support gaussien. Dans `FoamYade.C`, `locatePt()` demande à `meshTree` les cellules voisines dans `interpRange`, puis `calcInterpWeightGaussian()` calcule les distances aux centres de cellules et divise chaque poids par `allwt`.

La documentation YADE relie généralement la largeur `\sigma` à la distance de coupure `\delta` par

\[
\sigma=\frac{\delta}{2\sqrt{2\ln 2}},
\]

de sorte que le poids soit divisé par deux à une distance `\delta/2` [1]. Dans le port intégré, il faut distinguer ce paramètre théorique du paramètre effectif `interpRange` utilisé dans l’exponentielle. La relation exacte dépend de la valeur fournie par le moteur et doit être vérifiée dans le dictionnaire ou le code de version employé.

### 13.3 Interpolation du fluide vers une particule

Pour toute grandeur eulérienne `\phi_c`, l’interpolation à la position particulaire est

\[
\widetilde{\phi}_p
=\sum_{c\in\mathcal{N}(p)}w_{pc}\phi_c.
\]

En particulier,

\[
\widetilde{\mathbf{U}}_{f,p}
=\sum_cw_{pc}\mathbf{U}_{f,c},
\qquad
\widetilde{\epsilon}_{f,p}
=\sum_cw_{pc}\epsilon_{f,c}.
\]

La vitesse relative utilisée par la traînée est

\[
\mathbf{U}_{r,p}
=\widetilde{\mathbf{U}}_{f,p}-\mathbf{U}_{p}.
\]

La même opération est appliquée aux gradients nécessaires aux forces de pression, aux contraintes visqueuses et aux couples. Une interpolation cohérente exige que les grandeurs soient évaluées au même instant de couplage; mélanger `U` à l’instant `n+1` avec `U_p` à l’instant `n` introduit un retard numérique non documenté.

### 13.4 Projection d’une particule vers le maillage

Le volume solide pondéré apporté par une particule `p` à une cellule `c` est

\[
V_{p\to c}=V_p w_{pc}.
\]

Pour plusieurs particules,

\[
V_{s,c}=\sum_{p\in\mathcal{P}(c)}V_pw_{pc},
\qquad
\epsilon_{s,c}=\frac{V_{s,c}}{V_c},
\qquad
\epsilon_{f,c}=1-\epsilon_{s,c}.
\]

Le code accumule également la quantité de mouvement particulaire pondérée :

\[
\mathbf{M}_{p,c}=\sum_pV_pw_{pc}\mathbf{U}_p.
\]

La vitesse particulaire moyenne dans la cellule est alors reconstruite par

\[
\mathbf{U}_{p,c}=\frac{\mathbf{M}_{p,c}}{V_{s,c}},
\]

avec les protections numériques prévues par l’implémentation. Dans `setCellVolFraction()`, le code calcule la fraction fluide à partir du volume pondéré et impose une borne inférieure de `0.10` à la fraction fluide utilisée par le modèle. Cette borne évite une division par une fraction quasi nulle, mais elle signifie aussi que les situations très denses sont régularisées; les résultats proches de cette borne doivent donc être interprétés avec prudence.

### 13.5 Conservation discrète du filtre

Une propriété importante est la conservation du volume projeté. Pour une particule isolée loin des frontières, une normalisation adaptée doit donner approximativement

\[
\sum_cV_c\,\epsilon_{s,c}\approx V_p.
\]

Si le noyau est renormalisé uniquement par `\sum_cw_{pc}=1` mais que les volumes cellulaires varient fortement, cette égalité n’est pas automatiquement exacte. Il faut alors vérifier la convention de poids utilisée par le code et réaliser un test de conservation sur un maillage non uniforme. Un indicateur pratique est

\[
E_V=\frac{\left|\sum_cV_c\epsilon_{s,c}-\sum_pV_p\right|}{\sum_pV_p}.
\]

Un second indicateur concerne une grandeur constante : si `\phi_c=\phi_0` partout, l’interpolation doit restituer `\widetilde{\phi}_p=\phi_0` à l’erreur d’arrondi près. Ces deux tests détectent respectivement les erreurs de projection et les erreurs de normalisation.

## 14. Forces fluide–particule : dérivation et transfert

### 14.1 Principe d’action-réaction

La force hydrodynamique totale appliquée à une particule est notée `\mathbf{F}_{hyd,p}`. La force de réaction injectée dans le fluide doit être `-\mathbf{F}_{hyd,p}`. Pour une cellule, la source volumique de quantité de mouvement est donc construite à partir de la somme des forces projetées :

\[
\mathbf{S}_{f,c}
= -\frac{1}{\rho_fV_c}
\sum_{p}w_{pc}\mathbf{F}_{hyd,p}.
\]

Le facteur `1/V_c` transforme une force en force par volume et le facteur `1/\rho_f` transforme cette contribution en accélération lorsque l’équation OpenFOAM est écrite sous forme par unité de masse. La vérification globale doit comparer

\[
\sum_p\mathbf{F}_{hyd,p}
+\rho_f\sum_cV_c\mathbf{S}_{f,c}\approx\mathbf{0}.
\]

Cette identité est le bilan d’action-réaction discret. Elle doit être testée séparément pour la traînée et pour les forces complémentaires.

### 14.2 Traînée ponctuelle dans icoFoamYade

Dans le cas ponctuel, `stokesDragForce()` interpole `U` avec l’interpolateur de cellule OpenFOAM dans la cellule contenant la particule. La force est

\[
\mathbf{F}_{drag,p}
=3\pi\mu d_p
(\mathbf{U}_{f,p}-\mathbf{U}_p).
\]

Dans le code, `\mu=\rho_f\nu`, donc le coefficient est implémenté sous la forme `3*pi*d_p*nu*rhoF`. La force opposée est immédiatement ajoutée à la source de la cellule :

\[
\mathbf{S}_{f,c}
=-\frac{\mathbf{F}_{drag,p}}{\rho_fV_c}.
\]

Ce modèle suppose notamment une particule sphérique, un régime de faible Reynolds et une échelle particulaire non résolue par le maillage. Le nombre de Reynolds doit être surveillé :

\[
Re_p=\frac{\rho_f\lVert\mathbf{U}_f-\mathbf{U}_p\rVert d_p}{\mu}.
\]

Au-delà du domaine de validité de Stokes, il faut employer un modèle de traînée approprié plutôt que conserver silencieusement cette expression.

### 14.3 Traînée moyennée dans pimpleFoamYade

Dans le cas gaussien, le code interpole `\mathbf{U}_f` et `\alpha_f` sur le support de la particule. Il définit

\[
\alpha_p=1-\alpha_f,
\qquad
\mathbf{U}_r=\widetilde{\mathbf{U}}_f-\mathbf{U}_p,
\qquad
Re_p=\frac{\lVert\mathbf{U}_r\rVert d_p}{\nu}.
\]

Pour `Re_p<1000`, le coefficient de traînée Schiller–Naumann est

\[
C_d=\frac{24}{Re_p}\left(1+0.15Re_p^{0.687}\right),
\]

et le code prend `C_d=0.44` au-delà de cette limite. Dans une suspension suffisamment diluée, la fermeture est de la forme

\[
K=\frac{3}{4}C_d\frac{\rho_f}{d_p}
\lVert\mathbf{U}_r\rVert\alpha_f\alpha_p\alpha_f^{-2.65}.
\]

La force particulaire utilisée par l’implémentation est construite à partir du volume pondéré `V_p^{eff}` et d’une correction par la fraction solide :

\[
\mathbf{F}_{drag,p}
=V_p^{eff}K\mathbf{U}_r\,\alpha_p^{-1}.
\]

La dépendance en `\alpha_f` modélise l’effet de concentration et la réduction de l’espace disponible au fluide. Lorsque `\alpha_f\leq0.8`, le code bascule vers une fermeture dense comportant un terme visqueux et un terme inertiel de type Ergun–Wen–Yu :

\[
K_{dense}
=150\mu_f\frac{\alpha_p^2}{\alpha_f d_p^2}
+1.75\alpha_p\frac{\rho_f}{d_p\alpha_f}
\lVert\mathbf{U}_r\rVert.
\]

Cette bifurcation doit être mentionnée dans toute étude scientifique : le modèle réellement utilisé n’est pas une unique loi Schiller–Naumann dans tout le domaine de fraction solide.

### 14.4 Force de pression et contrainte visqueuse

La force dite d’Archimède ou force ambiante est évaluée à partir du gradient de pression et de la divergence de la contrainte visqueuse interpolés sur le support gaussien :

\[
\mathbf{F}_{by,p}
=V_p^{eff}\left(-\widetilde{\nabla p}
+\widetilde{\nabla\cdot\boldsymbol{\tau}}\right).
\]

Le code calcule cette contribution dans `archimedesForce()`, l’ajoute à `hydroForce` puis projette la réaction opposée avec les mêmes poids `w_{pc}`. La cohérence des signes doit être contrôlée avec un cas simple de gradient de pression uniforme : une particule immobile doit recevoir une force dans la direction de la diminution de pression, tandis que le fluide reçoit la réaction.

### 14.5 Masse ajoutée

Une force de masse ajoutée peut être écrite sous la forme

\[
\mathbf{F}_{am,p}
=C_mV_p\left(
\frac{D\widetilde{\mathbf{U}}_f}{Dt}
-\frac{d\mathbf{U}_p}{dt}
\right).
\]

La fonction `addedMassForce()` existe dans le port et projette elle aussi la réaction sur les cellules. Toutefois, dans la version actuelle de `calcHydroForce()`, elle n’est pas appelée : elle ne doit donc pas être annoncée comme active dans les résultats sans modification supplémentaire du code. Cette distinction entre fonction disponible et modèle activé est importante pour la traçabilité scientifique.

### 14.6 Couple hydrodynamique

Pour une sphère, le couple visqueux est calculé à partir de la rotation fluide interpolée, reconstruite depuis l’antisymétrie du gradient de vitesse :

\[
\boldsymbol{\omega}_f
\sim\frac{1}{2}\nabla\times\mathbf{U}_f,
\]

\[
\mathbf{T}_{hyd,p}
\propto \mu d_p^3
(\boldsymbol{\omega}_f-\boldsymbol{\omega}_p).
\]

Le couple est ajouté à `hydroTorque` et transmis à YADE avec la force dans le tampon MPI. La conservation du moment est plus délicate que celle de la force, car une projection au centre de cellule peut modifier le bras de levier; elle doit être vérifiée dans un cas de cisaillement uniforme.

## 15. Stabilité, unités et tests de cohérence

Le filtre gaussien ne remplace pas une analyse de stabilité. Le pas de couplage doit résoudre à la fois le temps de relaxation hydrodynamique `\tau_p`, le temps de collision DEM et la variation du champ fluide. Une estimation de Stokes pour une sphère est

\[
\tau_p=\frac{\rho_pd_p^2}{18\mu_f},
\]

et le nombre de Stokes peut être estimé par

\[
St_k=\frac{\tau_p\lVert\mathbf{U}_f\rVert}{d_p}.
\]

Les tests recommandés sont les suivants : une particule dans un champ uniforme pour vérifier l’interpolation de vitesse; une particule dans un gradient uniforme pour vérifier la force de pression; plusieurs particules dans une cellule pour vérifier l’additivité du volume solide; un maillage non uniforme pour vérifier la conservation du volume; et enfin un bilan global force fluide–particule pour vérifier l’action-réaction.

Les unités doivent rester cohérentes en SI : `\rho_f` et `\rho_p` en kg m⁻³, `\mu` en Pa s, `\nu` en m² s⁻¹, `d_p` en m, `V_p` en m³, les forces en N et les sources OpenFOAM en accélération ou force volumique selon la forme exacte de l’équation. Une erreur classique consiste à utiliser `\nu` comme une viscosité dynamique dans une formule qui attend `\mu`.

## 16. Références

[1]: https://yade-dem.org/doc/FoamCoupling.html "YADE — CFD-DEM coupled simulations with Yade and OpenFOAM"
[2]: https://yade-dem.org/doc/yade.wrapper.html "YADE — wrapper class reference"
[3]: https://precice.org/adapter-openfoam-overview "preCICE — OpenFOAM adapter"
[4]: https://www.cfdem.com/media/CFDEM/docu/CFDEMcoupling_Manual.html "CFDEM coupling manual"
[5]: https://mxui.github.io/ "MUI — Multiscale Universal Interface"
[6]: https://openfoam.org/version/13/ "OpenFOAM Foundation — OpenFOAM 13"
[7]: https://doc.cfd.direct/openfoam/user-guide-v13/ "OpenFOAM Foundation — OpenFOAM v13 User Guide"
