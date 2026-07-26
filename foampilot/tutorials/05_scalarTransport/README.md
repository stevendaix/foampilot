# Tutoriel 5 : Transport de scalaire (scalarTransportFoam)

## Objectif
Simuler le transport d'un scalaire passif (température, concentration) dans un écoulement.

## Cas de référence
OpenFOAM-14 : `tutorials/scalarTransportFoam/scalarTransport`

## Physique
- Écoulement laminaire incompressible
- Équation de convection-diffusion pour un champ scalaire T
- Conditions aux limites Dirichlet (entrée) et Neumann (parois)

## Champs résolus
- `U` — champ de vitesse
- `p` — pression
- `T` — température (scalaire passif)