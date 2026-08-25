# Références GitHub pour un exemple spray VOF-to-DPM

## Référence principale : atomizationFoam

Dépôt : [imfd-stroemungsmechanik/atomizationFoam](https://github.com/imfd-stroemungsmechanik/atomizationFoam)

Le dépôt décrit un couplage 3D entre un solveur Volume-of-Fluid et le suivi lagrangien pour la simulation d’atomisation de sprays. Son README indique explicitement la conversion de petits éléments VoF en parcels lagrangiens, ainsi que les interactions collision, coalescence, breakup secondaire, raffinement adaptatif et exécution parallèle. La branche actuelle annonce une adaptation à OpenFOAM v2212. Elle contient un solveur dans `applications/atomizationFoam`, une bibliothèque dans `src/libAtomization` et un cas d’exécution `run/crossFlow`.

## Référence de code et publication

Dépôt : [MiaoYangFluid/VOFCouplLPT](https://github.com/MiaoYangFluid/VOFCouplLPT)

Ce dépôt, dérivé de `ElsevierSoftwareX/SOFTX_2020_30`, fournit le même exemple atomizationFoam et indique une base OpenFOAM v1912. Son cas `run/crossFlow` est particulièrement pertinent pour construire un exemple de spray, mais son architecture est un solveur monolithique spécifique et ne suit pas directement l’API `fvModel`/`parcelCloudList` d’OpenFOAM 13 utilisée par foampilot.

La publication associée est Heinrich et Schwarze (2020), « 3D-coupling of Volume-of-Fluid and Lagrangian particle tracking for spray atomization simulation in OpenFOAM », SoftwareX 11, DOI 10.1016/j.softx.2020.100483.

## Conséquence pour foampilot

Le cas le plus proche n’est donc pas un simple `damBreak` : il faut reproduire un écoulement de jet liquide en cross-flow ou une sortie de buse, avec une zone de ligament/gouttelettes où les fragments VoF sont progressivement convertis en parcels. Pour OpenFOAM 13 et foampilot, le meilleur premier exemple est un cas de cross-flow minimal conservant le solveur `compressibleVoF` ou `incompressibleVoF`, mais remplaçant l’injection manuelle par `vofFragmentInjection` et vérifiant la masse transférée, le volume résiduel VoF et, pour le chemin thermo, l’enthalpie.

## Références

[1]: https://github.com/imfd-stroemungsmechanik/atomizationFoam "atomizationFoam, dépôt GitHub principal"
[2]: https://github.com/MiaoYangFluid/VOFCouplLPT "VOFCouplLPT, dépôt GitHub associé"
[3]: https://www.sciencedirect.com/science/article/pii/S2352711020300303 "Publication SoftwareX sur le couplage VOF–LPT"
