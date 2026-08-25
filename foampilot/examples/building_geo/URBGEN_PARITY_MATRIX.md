# Matrice de parité UrbGEN / foampilot

Cette matrice définit le score de parité et le critère de validation de chaque groupe fonctionnel. Un groupe n’est considéré comme validé que si son test numérique et son test visuel passent.

| Groupe | Pondération | Critère 100 % | État initial |
|---|---:|---|---|
| Normalisation des paramètres | 10 | Toutes les bornes et valeurs par défaut du GHA sont reproduites | À auditer |
| Population des centroïdes | 8 | Même filtrage, ordre déterministe et comportement multi-site | Partiel |
| Typologies I/L/T/H/C/Plus | 15 | Même modules, aire et orientation à tolérance définie | Partiel |
| Rotation | 5 | Mêmes candidats et règles d’alignement | Partiel |
| Courtyard | 20 | Même zones, anneaux, coins, ruptures et couverture | Partiel |
| Placement et espacement | 10 | Même containment, distance et déplacement maximal | Partiel |
| Croissance/réduction BCR | 15 | Même plafond, seuil d’expansion, tolérance et retrait | Partiel |
| Podium | 7 | Podiums individuels, regroupement et offset optimal | Partiel |
| FAR et hauteurs | 7 | Même GFA résiduelle, niveaux, variation et règlements | Partiel |
| Sorties et diagnostics | 3 | Toutes les métriques et métadonnées accessibles dans `UrbGENResult` | Partiel |

## Règle de score

Le score publié doit être calculé comme la somme des groupes validés, et non comme une estimation subjective. Tant qu’aucun fichier de référence Grasshopper n’est disponible, le score géométrique peut être validé fonctionnellement mais pas déclaré numériquement équivalent.

## Fixtures nécessaires

Pour une parité numérique complète, il faut au minimum exporter depuis Grasshopper, pour trois sites (rectangle, concave, multi-zone), deux seeds et les modes typologiques 0–7 : les empreintes, centres, angles, niveaux, surfaces, BCR, FAR, podiums et diagnostics. Les fixtures doivent être stockées dans un format JSON indépendant de Rhino, avec une tolérance géométrique explicitement indiquée.

## Validation visuelle round 3

Le visualiseur `plot_urbgen_validation.py` produit cinq scénarios avec un PNG et un JSON de métriques pour chacun.

| Scénario | BCR observé | FAR observé | Tours | Podiums | Contrôle visuel |
|---|---:|---:|---:|---:|---|
| random_typologies | 0.1600 | 2.4215 | 4 | 4 | Valide |
| courtyard | 0.1555 | 2.0221 | 3 | 0 | Couronne et ruptures visibles |
| edge_aligned | 0.1401 | 3.0278 | 4 | 4 | Orientations et containment valides |
| setback_moved | 0.0931 | 2.5131 | 5 | 0 | Déplacement et containment valides |
| podium_edge | 0.1201 | 2.0124 | 5 | 5 | Podiums individuels et containment valides |

Les contrôles visuels ont été effectués sur les cas `setback_moved`, `podium_edge`, `edge_aligned` et `courtyard`. Les fichiers JSON associés conservent les diagnostics de convergence, le nombre d’itérations et les hauteurs pour une validation automatisée ultérieure.
