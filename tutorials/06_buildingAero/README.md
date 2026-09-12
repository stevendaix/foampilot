# Tutoriel 6 : Aérodynamique des bâtiments (simpleFoam)

## Objectif
Simuler l'écoulement extérieur turbulent autour d'un quartier de bâtiments.

## Cas de référence
OpenFOAM-14 : `tutorials/incompressible/simpleFoam/buildingAirFlow`

## Physique
- Écoulement turbulent extérieur (RWIND, bâtiments comme obstacles)
- Vitesse d'entrée : 10 m/s
- Turbulence intensité : 15% (conditions réalistes en ville)
- Modèle k-omega SST (robuste pour les parois)
- topoSet + createPatch pour la gestion des patchs

## Résultats attendus
- Zones de recirculation derrière chaque bâtiment
- Rue canyon effects
- Concentration de poussière/pollutants (si scalar transport ajouté)
- Cartographie du champ de vitesse au niveau du sol