# Electronique CHT — Dissipateur en cuivre

Cas de comparaison du cas baseline `electronique` avec un **dissipateur en cuivre** au lieu d'aluminium.

## Objectif

Étudier l'impact du matériau du dissipateur sur :
- La résistance thermique conductrice
- La température maximale de la puce
- La répartition de température dans les ailettes

## Différences avec le cas baseline

| Paramètre | Baseline (electronique) | Ce cas |
|-----------|------------------------|--------|
| Matériau ailettes | Aluminium (k=205 W/m·K) | Cuivre (k=401 W/m·K) |
| Conductivité thermique | 205 | 401 |
| Résistance thermique | Plus élevée | Plus faible |

## Usage

```bash
cd examples/cht/electronique_copper_heatsink
python run.py
```

## Référence

Voir aussi : `electronique/run.py` pour le cas baseline.