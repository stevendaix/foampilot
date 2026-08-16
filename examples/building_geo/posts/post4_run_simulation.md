# Post 4 : Lancer la simulation avec foampilot

## Avant
On appelait `foamRun` à la main dans le terminal.

## Maintenant
`voxcity_vector_example.py` utilise `Solver.run_simulation()` de foampilot.

```python
solver = setup_openfoam_case(case_dir, nb_proc=args.nb_proc)
solver.run_simulation(nb_proc=args.nb_proc)
```

## Avantages
- tout est dans un seul script,
- log automatique dans `log.incompressibleFluid`,
- gestion du mode série / parallèle transparente.

## Résultat
La simulation tourne, écrit les champs dans `2000/`, et se termine sans erreur.
