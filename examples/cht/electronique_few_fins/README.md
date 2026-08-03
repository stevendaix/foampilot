# Electronique CHT — Peu d'ailettes (3 fins)

Cas de comparaison du cas baseline `electronique` avec **3 ailettes** au lieu de 5.

## Objectif

Étudier l'impact du nombre d'ailettes sur :
- La résistance thermique du système
- La température maximale de la puce
- Le débit massique d'air à la sortie

## Différences avec le cas baseline

| Paramètre | Baseline (electronique) | Ce cas |
|-----------|------------------------|--------|
| Nombre d'ailettes | 5 | 3 |
| Dissipateur | 20×20×2 mm base + 5 ailettes | 20×20×2 mm base + 3 ailettes |
| Surface d'échange | Plus grande | Plus petite |

## Usage

```bash
cd examples/cht/electronique_few_fins
python run.py
```

## Référence

Voir aussi : `electronique/run.py` pour le cas baseline.