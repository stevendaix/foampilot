# Tutoriel 1 : Cavité entraînée laminaire (icoFoam)

## Objectif
Apprendre à configurer un cas laminaire incompressible simple avec foampilot.

## Cas de référence
OpenFOAM-14 : `tutorials/incompressible/icoFoam/cavity`

## Physique
- Écoulement incompressible laminaire
- Cavité carrée 2D avec paroi mobile supérieure (lid) à U = 1 m/s
- Autres parois : no-slip (wall)
- Re ≈ 100 (basé sur la longueur de la cavité et la vitesse du lid)

## Fichiers générés
- `system/controlDict` — contrôle temporel et fréquence d'écriture
- `system/fvSchemes` — schèmes discrétisation
- `system/fvSolution` — solveurs linéaires
- `0/U` — champ de vitesse initial et conditions aux limites
- `0/p_rgh` — champ de pression modifié par la gravité
- `constant/transportProperties` — viscosité cinématique
- `constant/turbulenceProperties` — mode laminaire

## Résultats attendus
- Profil de vitesse U en x au centre de la cavité
- Recirculation dans le coin inférieur gauche
- Champ de pression p_rgh avec un minimum au centre