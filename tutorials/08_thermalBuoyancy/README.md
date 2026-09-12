# Tutoriel 8 : Convection thermique naturelle (buoyantSimpleFoam)

## Objectif
Simuler la convection naturelle dans une pièce avec parois isothermes.

## Cas de référence
OpenFOAM-14 : `tutorials/incompressible/buoyantSimpleFoam/room`

## Physique
- Écoulement naturel couplé thermo-fluide
- Gravité active (+Z vers le haut)
- Modèle de turbulence k-epsilon
- Approximation Boussinesq pour la dilatation thermique
- Parois : hotWall (350 K), coldWall (300 K), autres adiabatiques

## Champs résolus
- `U` — vitesse (m/s)
- `p_rgh` — pression hydrostatique modifiée
- `T` — température (K)
- `k`, `epsilon` — turbulence

## Résultats attendus
- Cellules de convection naturelle dans la pièce
- Distribution de température en profil de coin chaud/froid
- Vitesse de montée près des parois chaudes
- Champ de pression hydrostatique p_rgh