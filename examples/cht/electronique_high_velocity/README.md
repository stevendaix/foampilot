# Electronique CHT — Haute vitesse (2 m/s)

Cas de comparaison du cas baseline `electronique` avec une **vitesse d'entrée de 2 m/s** au lieu de 1 m/s.

## Objectif

Étudier l'impact de la vitesse d'air sur :
- Le coefficient de convection forcée
- La température de sortie de l'air
- La température maximale de la puce

## Différences avec le cas baseline

| Paramètre | Baseline (electronique) | Ce cas |
|-----------|------------------------|--------|
| Vitesse entrée | 1 m/s | 2 m/s |
| Nombre de Reynolds | Plus faible | Plus élevé |
| Convection | Naturelle/mixte | Forcée |

## Usage

```bash
cd examples/cht/electronique_high_velocity
python run.py
```

## Référence

Voir aussi : `electronique/run.py` pour le cas baseline.