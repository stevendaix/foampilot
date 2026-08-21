# Validation intermédiaire MakeHuman meshio – JOS-3 – OpenFOAM 13

## Périmètre

Cette validation porte sur le cas `openfoam_runs/meshio_openfoam_case`. Le maillage MakeHuman est converti par `meshio`, maillé par `snappyHexMesh`, puis couplé au modèle JOS-3 refactoré via la condition OpenFOAM 13 `externalCoupledTemperature`.

## Résultats du maillage

| Indicateur | Résultat |
|---|---:|
| Cellules finales | 89 604 |
| Points | 114 861 |
| Faces | 291 913 |
| Faces de la peau `human` | 20 223 |
| Volume total | 3,2311102 m³ |
| Non-orthogonalité maximale | 64,26° |
| Non-orthogonalité moyenne | 11,05° |
| Max skewness | 3,6227 |
| Faces illégales | 0 |
| État `checkMesh` | `Mesh OK` |

La surface `human` reste topologiquement signalée comme « multiply connected (shared edge) ». Cela n’empêche pas le maillage polyédrique ni le calcul en cavité fermée, mais la géométrie ne doit pas encore être considérée comme une surface topologiquement parfaite.

## Mapping physiologique

Le mapping utilise les centroïdes des faces du patch OpenFOAM. Les axes détectés sont `y` vertical, `x` latéral et `z` profondeur. Une correction a été appliquée à la règle de classification : les faces des bras étaient auparavant écrasées par la règle des jambes. Le mapping courant contient bien les 17 zones JOS-3.

| Grandeur | Résultat |
|---|---:|
| Faces échangées | 20 223 |
| Surface CFD totale | 3,208377175561 m² |
| Somme des ratios de surface | 1,000000000000 |
| Zone de surface maximale | Pelvis, 20,63 % |
| Zone de surface minimale | LHand, 0,0254 % |

Les ratios sont des ratios de surface du maillage CFD. Ils ne sont pas une preuve que la géométrie MakeHuman reproduit les BSA anatomiques internes de JOS-3 ; ils garantissent seulement la conservation de la partition de la surface CFD dans l’échange distribué.

## Couplage bidirectionnel réussi

Le pilote `openfoam13_jos3_driver.py` a été exécuté avec le cas meshio corrigé. OpenFOAM et Python terminent avec le statut zéro. Quatre échanges sont réalisés aux temps `0,05`, `0,10`, `0,15` et `0,20 s`, avec 20 223 valeurs de température retournées à chaque échange.

| Grandeur | Intervalle ou valeur |
|---|---:|
| HTC fourni par OpenFOAM | 2,028 à 51,36 W m⁻² K⁻¹ |
| Température JOS-3 cible | 33,39 à 34,02 °C au dernier échange |
| Température retournée après relaxation | environ 33,81 à 34,01 °C au dernier échange |
| Erreur globale de continuité cumulée à 0,20 s | 8,00 × 10⁻¹⁸ |

La sous-relaxation actuelle est `alpha = 0,1`. La conversion interne du protocole conserve les températures OpenFOAM en kelvins et expose les températures physiologiques à JOS-3 en degrés Celsius, puis reconvertit la température de peau avant écriture dans `data.in`.

## Limite frontière ouverte

La variante ouverte avec `inlet` en vitesse imposée et `outlet` en `zeroGradient` diverge vers `0,15–0,20 s` lors de la correction de pression. Le même maillage et le même couplage convergent jusqu’à `0,20 s` en cavité fermée. Le diagnostic courant attribue donc la limitation à la combinaison frontière ouverte–modèle compressible parfait gaz–flottabilité et non à la conversion meshio, au protocole d’échange ou au nombre de faces.

La prochaine étape doit isoler la stabilisation de la frontière ouverte : réduction du pas de temps, traitement cohérent de `p_rgh`/`U` au plafond et aux ouvertures, puis vérification de la formulation thermophysique. Il faut conserver la cavité fermée comme test de non-régression du couplage.

## Fichiers de preuve

Les traces reproductibles sont disponibles dans :

- `openfoam_runs/meshio_openfoam_case/meshio_Allrun.log`
- `openfoam_runs/meshio_openfoam_case/meshio_fluid_coupled_corrected.log`
- `openfoam_runs/meshio_openfoam_case/jos3_driver_meshio_corrected.log`
- `openfoam_runs/meshio_coupling_zone_audit.txt`
- `openfoam_runs/meshio_openfoam_case/zone_mapping_openfoam.csv`
