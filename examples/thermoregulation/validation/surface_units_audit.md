# Audit des unités et des surfaces OpenFOAM–JOS-3

## Unités du protocole OpenFOAM 13

La condition `externalCoupledTemperature` écrit dans `data.out` quatre colonnes :

| Colonne | Unité | Signification |
|---|---|---|
| `area` | m² | Aire de la face OpenFOAM |
| `T` | K | Température de la face |
| `qDot` | W/m² | Flux conductif/thermique surfacique calculé par OpenFOAM |
| `htc` | W/m²/K | Coefficient calculé comme `qDot/(Tp-Tc)` |

Le fichier `data.in` contient trois colonnes : `value` en K, `gradient` en K/m et `valueFraction` sans unité. La conversion Kelvin → Celsius du provider est correcte pour les entrées JOS-3. Les coefficients `h` sont déjà en W/m²/K et ne doivent pas être multipliés par une aire avant le calcul local du flux.

## Surfaces mesurées

| Modèle | Surface totale |
|---|---:|
| STL MakeHuman utilisée par snappyHexMesh | 4,563 m² |
| Patch `human` OpenFOAM actuel | 3,208 m² |
| BSA JOS-3, 1,70 m / 60 kg | 1,695 m² |
| BSA JOS-3 par défaut, 1,72 m / 74,43 kg | 1,874 m² |

Le patch CFD représente donc environ **1,71 fois** la BSA JOS-3 par défaut, et **1,89 fois** la BSA JOS-3 pour 1,70 m / 60 kg. L’écart est suffisamment important pour provoquer un déséquilibre énergétique : à température et coefficient identiques, le flux total intégré par OpenFOAM est beaucoup plus grand que la puissance thermorégulatrice distribuée par les surfaces physiologiques JOS-3.

## Écart par zone

Pour le mapping actuel et la BSA JOS-3 par défaut, les ratios `aire_CFD / aire_JOS3` sont très hétérogènes : tête 2,55, cou 6,66, poitrine 0,61, dos 1,92, pelvis 2,99, jambes 4,50 environ, cuisses 0,18, tandis que les bras sont à zéro dans le mapping produit. Cela révèle un problème indépendant de l’unité : la classification par centroïde ne rattache pas correctement certaines parties de la géométrie aux 17 zones.

## Conséquence dans `DistributedSurfaceNetwork`

Le réseau distribué répartit la capacité cutanée JOS-3 avec `area_fraction = area_CFD / zone_area_CFD`, ce qui conserve la capacité totale JOS-3 par zone. En revanche, la puissance environnementale est calculée par

`environment_power_face = h * (Ts - Ta) * area_CFD`.

Elle est donc intégrée sur la surface CFD réelle. Si `area_CFD` est supérieure à la surface physiologique, OpenFOAM reçoit et renvoie une puissance totale qui n’est pas cohérente avec la capacité et les conductances JOS-3. La division finale par `area_CFD` pour renvoyer un flux en W/m² est dimensionnellement correcte, mais elle ne corrige pas l’écart de surface entre les modèles.

## Conclusion

Il existe bien un problème potentiel de surface, mais pas une conversion d’unité élémentaire dans le provider principal. Les deux corrections prioritaires sont : premièrement, réparer le mapping des 17 zones; deuxièmement, choisir explicitement une convention de surface. Pour une conservation énergétique stricte, il faut soit utiliser les aires CFD réelles et recalibrer les capacités/conductances JOS-3 sur ces aires, soit appliquer un facteur d’aire par zone `A_JOS3_zone/A_CFD_zone` dans la puissance échangée, tout en documentant que le flux CFD local reste en W/m².

Le problème de surface ne doit pas être corrigé en multipliant ou divisant arbitrairement `h` : `h` conserve son unité W/m²/K. La correction doit agir sur l’aire utilisée dans l’intégration de puissance, après validation du mapping géométrique.
