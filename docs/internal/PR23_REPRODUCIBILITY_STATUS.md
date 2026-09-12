# PR #23 — Statut de reproductibilité Foundation 13

**Commit évalué :** `383e205b0845ee396dbba064a091fef93b49ef10`
**Branche :** `feature/marine-pr`
**Base :** `main`
**Environnement :** OpenFOAM Foundation 13, `WM_PROJECT_VERSION=13`, compilation GCC native, exécution séquentielle dans le sandbox Linux.

## Verdict actuel

La couche logicielle et les tests analytiques sont reproductibles depuis un checkout propre. Les trois cas marins sont maintenant relocalisables et leurs runners ne dépendent plus d’un chemin d’installation codé en dur. En revanche, la PR ne doit pas annoncer une validation complète des trois cas CFD : **Turning35 passe le smoke test complet**, DTC a terminé le maillage et le solver lorsqu’ils sont exécutés séparément, tandis que le pipeline DTC complet dépasse la fenêtre de smoke test à cause du coût de `snappyHexMesh`; le propeller génère le maillage AMI mais conserve deux erreurs `checkMesh` et son solver n’a pas terminé dans la limite imposée.

## Validations reproductibles

| Domaine | Commande ou protocole | Résultat réel |
|---|---|---|
| Compilation | `source /opt/openfoam13/etc/bashrc; ./openfoam13/Allwmake` depuis un clone neuf | Succès ; bibliothèque, `marineFoam` et les quatre harnesses compilés |
| Tests Python | `PYTHONPATH=foampilot/src pytest -q foampilot/test/test_marine_*.py` | **39 passed**, 3 warnings non bloquants |
| Matrice overset | `marineOversetMatrixTest` dans la fixture à deux régions | Succès ; 2 cellules contraintes, valeur interpolée 1,5 |
| Lecture stencil | `marineInterMeshStencilTest` | Succès ; 1 acceptor, donor `background`, acceptor `hull` |
| Couplage inter-mailles | `marineInterMeshCouplingTest` | Succès ; cible 4, donor 4, 1 stencil |
| Interpolation analytique | `marineOversetInterpolationTest` | Succès ; scalaire 5, vecteur `(3 4 5)`, erreurs nulles |
| Turning35 | Copie vide, `FOAMPILOT_MARINE_ENV=<checkout>/openfoam13/marine_env.sh`, `./Allclean && ./Allrun`, `endTime=0.0002` | **Succès, code retour 0** ; `marineFoam` termine à `End`, alpha borné, forces/moments écrits |
| DTC maillage | Copie vide, génération FoamPilot, `blockMesh/refineMesh/snappyHexMesh`, couches désactivées uniquement dans la copie smoke | Maillage produit sans erreur ; 879 026 cellules ; `snappyHexMesh` en 197,574 s |
| DTC calcul | `checkMesh`, `setWaves`, puis `marineFoam -solver incompressibleVoF` sur le maillage obtenu | Succès séparé ; `checkMesh=0`, `setWaves=0`, `marineFoam=0`, alpha borné, forces/moments non nuls, `End` |
| Propeller maillage | Copie vide, `./Allclean && ./Allrun`, `endTime=0.0002` | Maillage AMI créé ; `createNonConformalCouples` termine, 45 836 couplages calculés en 4,826 s |
| Propeller qualité | `checkMesh` après AMI | Réserve bloquante pour validation physique : 4 faces mal orientées et 15 faces très skew, deux contrôles échouent |
| Propeller calcul | `marineFoam -solver incompressibleVoF` dans la limite de 180 s | Non terminé dans la limite ; le log montre une progression PIMPLE et des sorties de forces, mais aucun succès complet ne doit être déclaré |

## Corrections intégrées dans cette passe

Le helper `openfoam13/marine_env.sh` résout l’environnement Foundation 13 et les racines du dépôt. Les runners acceptent également `FOAMPILOT_MARINE_ENV` lorsqu’un cas est copié hors de l’arborescence du dépôt. Les chemins absolus d’installation et les dépendances aux clones de tutoriels ont été supprimés dans les scripts marins ciblés. Les fichiers `.pyc` introduits par la PR ont été retirés de l’index.

Le générateur DTC normalise maintenant `system/functions` à chaque génération : il retire le function object `rigidBodyForces` hérité du tutoriel et ajoute le function object Foundation 13 `forces`. Les runners DTC et Turning35 n’appellent plus `postProcess -func forces`, qui cherche un fichier `system/forces` inexistant ; les forces et moments sont écrits pendant `marineFoam`.

## Conditions avant merge

La PR est techniquement mergeable selon GitHub (`MERGEABLE`, `CLEAN` au dernier contrôle), mais la validation CFD complète reste conditionnelle. Le merge peut être accepté pour intégrer l’architecture et les tests Foundation 13 si les réserves ci-dessus sont explicitement conservées. Avant toute annonce de validation hydrodynamique, il faut corriger les 4 faces mal orientées et les 15 faces skew du propeller, terminer un run propeller, exécuter le pipeline DTC complet dans une fenêtre adaptée et conduire les études de convergence en maillage et en temps.

## Reproduction minimale

```bash
source /path/to/OpenFOAM-13/etc/bashrc
export FOAMPILOT_MARINE_ENV=/path/to/foampilot/openfoam13/marine_env.sh
cd /tmp/case-copy
./Allclean
./Allrun
```

Pour le build et les tests analytiques :

```bash
source /path/to/OpenFOAM-13/etc/bashrc
cd /path/to/foampilot
./openfoam13/Allwmake
PYTHONPATH=foampilot/src pytest -q foampilot/test/test_marine_*.py
cd openfoam13/marineOversetMatrixTest/case
./Allrun
```
