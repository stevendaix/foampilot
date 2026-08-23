# Couplage VoF–DPM sous OpenFOAM 13

Cette extension fournit deux `fvModel` natifs pour raccorder un cloud Lagrangien aux solveurs modulaires `incompressibleVoF` et `compressibleVoF` d’OpenFOAM 13. Le modèle est appelé pendant la boucle PIMPLE, fait évoluer le cloud à chaque pas de temps et ajoute son terme de quantité de mouvement à l’équation `U` via `fvModels().source(rho, U)`.

## Modèles disponibles

| Solveur | Type fvModel | Bibliothèque | Dictionnaire |
|---|---|---|---|
| `incompressibleVoF` | `incompressibleVoFClouds` | `libincompressibleVoFClouds.so` | `constant/fvModels` |
| `compressibleVoF` | `compressibleVoFClouds` | `libcompressibleVoFClouds.so` | `constant/fvModels` |

Les deux variantes utilisent `parcelCloudList` et `clouds_.SU(...)` pour transférer la réaction de quantité de mouvement du cloud vers le champ porteur. La variante compressible utilise la densité et la viscosité cinématique du mélange compressible pour construire la viscosité dynamique interpolée du cloud.

## Validation

Le cas incompressible est exécuté par `test/openfoam13/incompressibleVoFCloudsDamBreak/Allrun`. Le cas compressible est exécuté par `test/openfoam13/compressibleVoFCloudsDamBreak/Allrun`. Chaque script compile le modèle nécessaire, prépare un damBreak OpenFOAM 13, active explicitement les modèles et le prédicteur de quantité de mouvement, puis vérifie dans le journal : la sélection du solveur, la sélection du `fvModel`, la création de `collidingCloud` et la présence d’un parcel actif.

Les deux tests passent dans l’environnement OpenFOAM 13 utilisé pour cette branche. Les journaux montrent un parcel actif, une masse conservée par le cloud d’environ `2.06658e-05 kg` dans le cas de test et une quantité de mouvement Lagrangienne non nulle évoluant au cours du temps.

## Exécution

```sh
. /opt/openfoam13/etc/bashrc
cd test/openfoam13/incompressibleVoFCloudsDamBreak
./Allrun

cd ../compressibleVoFCloudsDamBreak
./Allrun
```

## Portée et limite importante

Cette étape valide le couplage natif `fvModel`–cloud, l’évolution Lagrangienne et le transfert de quantité de mouvement avec une injection manuelle contrôlée. Elle ne constitue pas encore une preuve de conversion automatique de fragments VOF en parcels : l’injection utilisée par le cas de validation est fournie par `manualInjection`. La détection de composantes connexes, la soustraction conservative de volume/mass fraction à `alpha`, ainsi que la création dynamique de parcels à partir des fragments doivent rester isolées dans une prochaine couche de transfert dédiée, afin de garantir la conservation masse–quantité de mouvement et d’éviter une double comptabilisation.
