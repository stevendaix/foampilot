# Tutoriel 2 : Écoulement turbulent autour d'un véhicule (simpleFoam)

## Objectif
Configurer une simulation RANS stationnaire avec moustache k-omega SST pour un écoulement extérieur autour d'une géométrie simplifiée.

## Cas de référence
OpenFOAM-14 : `tutorials/incompressible/simpleFoam/simpleCar`

## Physique
- Écoulement incompressible turbulent extérieur
- Vitesse d'entrée : 30 m/s (vent de face)
- Modèle de turbulence : k-omega SST
- Régime stationnaire (simpleFoam)

## Fichiers générés
- `system/controlDict` — with `adjustableRunTime` pour un pas de temps adaptatif
- `system/fvSchemes` — schèmes limitedGrad pour la robustesse
- `system/fvSolution` — solveurs GAMG pour la pression, smoothSolver pour les vitesses
- `system/functions` — fieldAverage et runTimeControls pour le monitoring

## Conditions aux limites
- **inlet** : `velocityInlet` avec turbulence (I=5%)
- **outlet** : `pressureOutlet` (p = 0 Pa)
- **walls** : `wall` avec no-slip
- **farfield** : `freestream` (pression de référence)

## Résultats attendus
- Coefficient de traînée Cd et portance Cl
- Champ de pression statique (Cp) sur la carrosserie
- Lignes de courant et visualisation des zones de séparation