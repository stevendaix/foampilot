# Intégration OpenFOAM 13 : ressources Wolf Dynamics et Figshare

**Statut :** plan d’intégration et critères de portage. Cette page ne certifie pas qu’un ancien cas fonctionne sous OpenFOAM 13 tant qu’il n’a pas été exécuté et contrôlé dans un environnement OpenFOAM 13.

## Objectif et périmètre

Cette page transforme le catalogue de tutoriels et de supports de Wolf Dynamics en feuille de route FoamPilot. Elle distingue les ressources pédagogiques, les cas de validation et les supports explicitement publiés pour OpenFOAM 13. Les supports externes restent référencés par lien : la page Figshare signale que les archives de formation sont protégées et que leur distribution ou duplication non autorisée est interdite [1]. FoamPilot doit donc contenir des adaptations originales, des métadonnées, des générateurs de cas et des tests, et non une copie des diapositives ou des archives propriétaires.

OpenFOAM 13 a remplacé une grande partie des anciens solveurs applicatifs par `foamRun` et des modules sélectionnés dans la configuration du cas [2]. La migration d’un tutoriel ancien doit donc vérifier le solveur, les dictionnaires, les noms de champs, les modèles physiques et les utilitaires, au lieu de simplement renommer une commande.

## Matrice de sélection et de compatibilité

Le statut **existe** signifie que le module ou l’utilitaire est documenté dans OpenFOAM 13. Le statut **portage requis** signifie que la ressource Wolf Dynamics est annoncée pour une version antérieure ou ne donne pas de version ; il ne s’agit pas d’une validation du cas.

| Ressource | Version annoncée par la source | Cible OF13 à vérifier | Intérêt FoamPilot | Statut initial |
|---|---:|---|---|---|
| Driven cavity revisited | OF4.1–5.x | `foamRun -solver incompressibleFluid` | Cas minimal pour génération, BC, convergence et post-traitement | Module OF13 existe ; portage requis |
| Hagen–Poiseuille | OF4.1–5.x | `incompressibleFluid` | Vérification analytique débit/perte de charge | Module OF13 existe ; test analytique à écrire |
| Dam break VOF | OF4.1–5.x | `incompressibleVoF` | Test d’interface libre, MULES et pas de temps | Module OF13 existe ; dictionnaires à migrer |
| Vortex shedding autour d’un cylindre | OF4.1–5.x | `incompressibleFluid` | Cas instationnaire et nombre de Strouhal | Module OF13 existe ; validation quantitative à écrire |
| Cas de validation dam-break 3D | OF7 | `incompressibleVoF` | Régression multiphasique et géométrie 3D | Portage OF13 requis |
| NACA 0012 haut nombre de Mach | OF7 | `shockFluid` ou module compressible approprié | Cas compressible et maillage externe | Choix du module à confirmer par le cas |
| High-lift MDA 30P-30N | OF7 | `incompressibleFluid` ou `fluid` selon le modèle | Aérodynamique et forces intégrées | Portage et coût élevés |
| Wigley Hull avec surface libre | OF7 | `incompressibleVoF` | VOF, surface libre et forces de coque | Portage OF13 requis |
| Deux Ahmed bodies | OF7 | `incompressibleFluid` | Turbulence externe, traînée et symétrie | Portage OF13 requis |
| Formation turbulence | Version non indiquée | `incompressibleFluid` ; `fluid` si compressible | Guide de choix RANS/LES et contrôles de qualité | Support à synthétiser, cas à sélectionner |
| Formation multiphasique | DOI Figshare | `incompressibleVoF`, `incompressibleMultiphaseVoF`, `compressibleVoF`, `multiphaseEuler` | Priorité élevée pour un catalogue multiphasique moderne | Existence OF13 documentée ; cas à inventorier |
| Formation maillages dynamiques | DOI Figshare | `movingMesh` et modules avec mouvement/topologie | Mouvement, AMR, NCC et redistribution | Fonctionnalités OF13 à vérifier dans chaque cas |
| Formation FVM/discrétisation | Version non indiquée | `fvSchemes`, `fvSolution`, `foamRun` | Documentation de méthode et bonnes pratiques numériques | Applicable comme guide, pas comme cas |
| Processus chimiques et combustion | **OF13 explicitement annoncé** | `XiFluid`, `multicomponentFluid`, `fluid`, `multiphaseEuler` selon le cas | Priorité maximale : ressource la plus proche de la cible | Archives à inventorier ; exécution à faire |

La liste officielle des modules OpenFOAM 13 confirme notamment `incompressibleFluid`, `incompressibleVoF`, `incompressibleMultiphaseVoF`, `compressibleVoF`, `multiphaseEuler`, `fluid`, `XiFluid`, `solid` et `movingMesh` [2]. La disponibilité d’un module ne garantit pas que les dictionnaires d’un tutoriel ancien soient compatibles.

## Stratégie d’intégration : l’adaptateur FoamPilot

L’intégration des tutoriels Wolf Dynamics dans FoamPilot utilise une approche par adaptateur (voir `foampilot.tutorials.wolfdynamics_base`). Cette classe de base garantit que :

1. **Aucun script externe n'est exécuté** : Les scripts `Allrun` fournis dans les archives ne sont jamais appelés. L'exécution passe exclusivement par le validateur FoamPilot et le lanceur `foamRun`.
2. **Le cas est copié et isolé** : L'archive source reste intacte. Le cas est copié dans un répertoire jetable `.runs/`.
3. **Les paramètres de test sont maîtrisés** : L'adaptateur force des valeurs de `endTime` et `writeInterval` réduites pour valider numériquement le démarrage (smoke test) sans consommer les ressources d'une simulation complète.
4. **Le contrat de complétude est respecté** : La méthode `validate_generated_case` vérifie la présence des dictionnaires fondamentaux et la déclaration de la viscosité cinématique (`nu`) pour les cas incompressibles.

## Plan d’intégration proposé

### Phase A — catalogue et provenance

Créer un manifeste FoamPilot pour chaque cas retenu : URL source, auteur, version annoncée, domaine physique, solveur historique, module OF13 cible, géométrie externe, licence, commande d’exécution et statut de vérification. Les archives Figshare doivent être traitées comme des sources à consulter ; seules les informations nécessaires, les liens et les adaptations originales doivent être versionnés.

### Phase B — cas pédagogiques minimaux

Porter d’abord la cavité entraînée, Hagen–Poiseuille, le cylindre instationnaire et le dam-break. **Note :** Les archives originales de ces tutoriels débutants (OF4.1–5.x) ne sont plus hébergées sur le site de Wolf Dynamics (erreur 404 sur les liens publics). L'intégration FoamPilot se concentre donc sur la reconstruction des cas VOF (DamBreak) à partir des modèles `incompressibleVoF` existants.

### Phase C — cas de validation et modèles avancés

Ajouter ensuite un sous-ensemble représentatif : un cas VOF 3D, un cas aérodynamique turbulent, un cas compressible et un cas de maillage dynamique. La terminologie **validation** ne sera utilisée que lorsqu’une grandeur de référence et une procédure reproductible sont définies. Sinon, le cas sera étiqueté **exemple pédagogique** ou **test de régression structurel**.

### Phase D — chimie et combustion OF13

L’édition Figshare d’octobre 2025 annonce explicitement OpenFOAM 13 et fournit des exemples ainsi qu’un support consacré aux processus chimiques, aux écoulements compressibles, à la FVM et à la turbulence [1].

**Cas intégrés et validés avec l'adaptateur FoamPilot :**
- **CounterFlow Flame (LTS)** : Flamme à contre-courant utilisant le module `multicomponentFluid`. Le cas valide le couplage thermodynamique et la chimie. L'adaptateur `CounterFlowFlameTutorial` a permis de lancer le maillage (`checkMesh`) et les 20 premières itérations de résolution des espèces (O2, H2O, CH4, CO2) et de l'enthalpie.
- **SandiaD Flame (EDC)** : Modélisation turbulente de la flamme Sandia D avec le modèle EDC (Eddy Dissipation Concept) et `multicomponentFluid`. Le cas valide l'intégration des propriétés chimiques complexes (mécanisme GRI30 réduit) et la résolution de l'énergie cinétique turbulente ($k$) et de son taux de dissipation ($\omega$).

## Bonnes pratiques OpenFOAM 13 à appliquer dans FoamPilot

> Un tutoriel réussi n’est pas automatiquement une validation physique. La source Wolf Dynamics avertit que ses tutoriels débutants sont didactiques et ne doivent pas être utilisés comme standards, benchmarks ou validations [3].

Chaque cas généré doit être contrôlé avant lancement. Le contrat minimal comprend `system/controlDict`, `system/fvSchemes`, `system/fvSolution`, les répertoires `constant` et `0`, ainsi que les conditions aux limites attendues. Pour les cas incompressibles, `constant/transportProperties` doit déclarer explicitement `nu`; l’absence de cette propriété doit produire une erreur de validation explicite et non un cas partiellement écrit.

Les anciens appels tels que `simpleFoam`, `pimpleFoam`, `interFoam` ou `XiFoam` doivent être documentés comme références historiques. Sous OpenFOAM 13, le cas doit utiliser `foamRun` avec un module approprié et un `controlDict` cohérent. La documentation officielle donne par exemple le démarrage `foamRun` avec le module `incompressibleFluid` pour le cas `pitzDailySteady` [4].

Pour les cas VOF, les contrôles MULES doivent être vérifiés dans `fvSolution` et le pas de temps doit être piloté par le nombre de Courant. OpenFOAM 13 améliore la bornitude MULES et introduit des contrôles plus structurés pour les fractions de phase [5]. Ces améliorations ne dispensent pas de vérifier `alpha` dans ses bornes, la conservation de masse et l’indépendance au pas de temps.

Pour les maillages dynamiques, `dynamicMeshDict` doit séparer, lorsque le cas le requiert, le mouvement (`mover`), le changement topologique (`topoChanger`) et la redistribution (`distributor`). Cette architecture correspond au modèle OpenFOAM moderne ; elle remplace les anciennes hypothèses basées sur une unique classe `dynamicFvMesh` [6]. Toute simulation adaptative ou parallèle doit contrôler la qualité du maillage après changement et l’équilibrage des sous-domaines.

Les rapports doivent conserver la version exacte (`foamVersion`), le chemin d’environnement, le commit FoamPilot, la commande exécutée, le nombre de processeurs, les tolérances et les fichiers d’entrée. Les sorties générées ne doivent pas être considérées comme des preuves de validité sans comparaison à une référence analytique, expérimentale ou publiée.

## Critères d’acceptation de la PR

| Critère | Vérification attendue |
|---|---|
| Compatibilité de version | La source et la version OF13 ciblée sont indiquées ; aucune ancienne version n’est présentée comme validée OF13 par défaut |
| Complétude du cas | `controlDict`, schémas, solution, champs initiaux, propriétés et géométrie sont présents ou explicitement externalisés |
| Génération FoamPilot | Le cas est généré sans édition manuelle de dictionnaire et peut être inspecté avant lancement |
| Contrat `nu` | Toute configuration incompressible écrit et vérifie `constant/transportProperties: nu` |
| Exécution | Le cas court s’exécute avec le module OF13 documenté, ou est marqué non exécuté avec la raison |
| Qualité numérique | `checkMesh`, résidus, Courant, bornes des fractions et bilans sont consignés selon le domaine |
| Provenance | Les liens et citations sont présents ; aucun support protégé n’est redistribué |
| Reproductibilité | Une commande et un test déterministes permettent de reproduire le contrôle |

## Références

[1]: https://figshare.com/articles/presentation/Overview_of_Chemical_Processes_with_OpenFOAM_Theory_and_applications/27640866 "Wolf Dynamics/Tonkomo — Overview of Chemical Processes with OpenFOAM: Theory and applications"
[2]: https://doc.cfd.direct/openfoam/user-guide-v13/solvers-modules "OpenFOAM v13 User Guide — Solver modules"
[3]: https://www.wolfdynamics.com/tutorials.html?id=126 "Wolf Dynamics — Getting started with OpenFOAM: Beginner tutorials"
[4]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM 13 — Download for Ubuntu"
[5]: https://openfoam.org/release/13/ "OpenFOAM 13 Released"
[6]: https://cfd.direct/openfoam/free-software/dynamic-meshes/ "CFD Direct — Dynamic meshes in OpenFOAM"
