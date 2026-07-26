# Tutoriel 7 : Moto (motorBike, simpleFoam)

## Objectif
Simuler l'écoulement aérodynamique autour d'une moto à haute vitesse.

## Cas de référence
OpenFOAM-14 : `tutorials/incompressible/simpleFoam/motorBike`

## Physique
- Écoulement extérieur à haute vitesse (30 m/s ≈ 108 km/h)
- Géométrie complexe : moto + route + roues tournantes
- Les roues sont modélisées comme des parois mobiles (sliding mesh)
- Modèle k-omega SST

## Résultats attendus
- Traînée aérodynamique totale (Cd)
- Distribution de pression sur les carénages
- Zones de séparation derrière la moto
- Effet des roues sur l'écoulement (roues comme obstacles)