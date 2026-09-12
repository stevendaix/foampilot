# Tutoriel 3 : Marche descendante (pitzDaily, simpleFoam)

## Objectif
Simuler un écoulement turbulent autour d'une marche descendante et visualiser le recouvrement de la recirculation.

## Cas de référence
OpenFOAM-14 : `tutorials/incompressible/simpleFoam/pitzDaily`

## Physique
- Écoulement incompressible turbulent
- Mach 1 faible (incompressible)
- Modèle k-omega SST
- Géométrie : marche descendante 2D
- Vitesse d'entrée : 1 m/s

## Résultats attendus
- Zone de recirculation derrière la marche
- Profil de vitesse rétabli en aval
- Pressions récupérées au mur
- Réduction de l'édition quand le maillage converge