# Matrice de réécriture des runners OpenFOAM 13

## Objectif

Cette matrice suit la réécriture réelle des runners de tutoriels OpenFOAM 13. Un runner ne sera considéré comme **réécrit avec FoamPilot** que lorsque les fichiers de mise en données sont produits par les classes FoamPilot et non simplement copiés depuis le tutoriel de référence.

Le statut **Validé fonctionnellement** signifie uniquement que le cas produit peut être exécuté et comparé sous OpenFOAM 13. Il ne signifie pas nécessairement que le cas a été entièrement réécrit avec l’API FoamPilot.

## Règles de classement

| Statut | Définition |
|---|---|
| `À auditer` | Le runner n’a pas encore été examiné selon les règles de cette matrice. |
| `Fonctionnel — import de référence` | Le cas fonctionne, mais importe un ou plusieurs fichiers de mise en données du tutoriel. |
| `Partiellement réécrit` | Les dictionnaires ou champs principaux sont générés par FoamPilot, mais des imports de mise en données subsistent. |
| `Réécrit FoamPilot` | Les champs, conditions aux limites, dictionnaires et propriétés principales sont générés avec FoamPilot; seuls les assets externes légitimes peuvent être importés. |
| `Réécrit et validé OF13` | Le statut `Réécrit FoamPilot` est atteint et la comparaison OpenFOAM 13 est réussie. |

## Imports autorisés

L’import d’une géométrie externe STL/OBJ/STEP ou d’un fichier imposé par un couplage externe peut rester autorisé. L’import doit cependant passer par FoamPilot et être déclaré comme ressource externe. Les imports de `0/`, `system/`, `constant/`, `blockMeshDict`, `controlDict`, `fvSchemes`, `fvSolution`, propriétés physiques ou conditions aux limites ne sont pas considérés comme une réécriture.

## Contrôles obligatoires

Chaque runner réécrit doit satisfaire les contrôles suivants:

1. Aucun import d’arbre complet depuis `0/`, `system/` ou `constant/`.
2. Les champs principaux sont créés par une API FoamPilot déclarative.
3. Les conditions aux limites sont définies par FoamPilot et non conservées uniquement dans un champ copié.
4. Les dictionnaires système sont construits ou configurés par FoamPilot.
5. Les propriétés physiques et les modèles sont écrits par FoamPilot; notamment `nu` doit être présent pour les cas incompressibles.
6. Les utilitaires OpenFOAM restent exécutés par FoamPilot, mais leurs dictionnaires sont générés par l’API.
7. Le cas généré est comparé à la référence OF13 sur les fichiers clés, le maillage, les patches, les champs initiaux et le comportement du solveur.
8. Le runner ne contient pas d’appel direct à `subprocess`, `os.system`, `shutil`, d’écriture directe de cas ou de chemin machine codé en dur.

## État initial mesuré

L’audit de la branche `refactor/api-generalization-core` a recensé **261 runners**. Les motifs suivants ont été mesurés:

| Motif | Runners ou occurrences | Conséquence |
|---|---:|---|
| `import_reference_field` | 239 runners / 244 appels | Les champs initiaux et leurs conditions aux limites sont encore majoritairement copiés. |
| `import_reference_file` | 121 runners / 274 appels | Les dictionnaires système ou régionaux sont encore souvent importés. |
| `import_reference_asset` | 150 runners / 190 appels | À distinguer entre géométrie externe légitime et mise en données déguisée. |
| `import_reference_dict` | 81 runners / 83 appels | Les `blockMeshDict` sont fréquemment repris au lieu d’être construits. |
| Boucles `rglob()`/`glob()` sur les références | 193 runners / 401 occurrences | Certains cas recopient des arbres complets de mise en données. |
| `run_command` | environ 614 appels | Les workflows OpenFOAM restent souvent exprimés comme des commandes brutes. |

## Ordre de réécriture

La réécriture sera effectuée séquentiellement, par familles, avec un commit séparé après chaque tranche validée:

1. cas simples `cavity`, `pitzDaily`, `scalarTransport` pour établir l’API déclarative des champs et dictionnaires;
2. maillages `blockMesh`, `snappyHexMesh` et `extrudeMesh`;
3. cas incompressibles et VoF;
4. cas compressibles et multiphasiques;
5. cas CHT et multi-région;
6. cas solides, chimie, XiFluid et couplages externes;
7. cas parallèles, multi-cas et mouvement de maillage.

Un cas ne sera déclaré réécrit qu’après comparaison des artefacts générés et validation OpenFOAM 13. Les runners précédemment validés par import resteront identifiés comme tels jusqu’à leur migration réelle.
