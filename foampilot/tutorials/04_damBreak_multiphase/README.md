# Tutoriel 4 : DamBreak — Écoulement bifluide VOF (interFoam)

## Objectif
Modéliser l'écoulement d'une colonne d'eau en chute libre dans un réservoir avec l'interface aire/eau.

## Cas de référence
OpenFOAM-14 : `tutorials/multiphase/interFoam/damBreak`

## Physique
- Écoulement incompressible à deux phases (eau + air)
- Modèle VOF (Volume of Fluid) pour追踪 l'interface
- Sans turbulence (laminaire par défaut pour ce cas de référence)
- Gravité active si le cas de référence l'exige

## Résultats attendus
- Chute libre de la colonne d'eau
- Rebond et éclaboussure contre les parois
- Évolution de l'interface alpha.water / alpha.air
- Conservation de la masse