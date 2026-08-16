# Post-traitement 4 : Analyser les résidus et la convergence

## Objectif
Vérifier que le calcul a bien convergé et identifier d'éventuelles instabilités.

## Méthode
- Lire `log.incompressibleFluid`.
- Extraire les résidus initiaux/finaux pour chaque champ.
- Tracer l'évolution des résidus en fonction du temps.

## Code minimal
```python
import re
import numpy as np
import matplotlib.pyplot as plt

log_path = "/tmp/voxcity_vector_demo3/log.incompressibleFluid"
with open(log_path) as f:
    lines = f.readlines()

fields = ["Ux", "Uy", "Uz", "p", "k", "epsilon"]
residuals = {f: [] for f in fields}
times = []

for line in lines:
    m = re.search(r"Time = (\d+)", line)
    if m:
        times.append(int(m.group(1)))
    for field in fields:
        m = re.search(rf"Solving for {field}.*Initial residual = ([\d\.e+-]+)", line)
        if m:
            residuals[field].append(float(m.group(1)))

plt.semilogy(times[:len(residuals["p"])], residuals["p"], label="p")
plt.semilogy(times[:len(residuals["Ux"])], residuals["Ux"], label="Ux")
plt.xlabel("Time (s)")
plt.ylabel("Initial residual")
plt.legend()
plt.grid(True, which="both")
plt.title("Convergence des résidus")
plt.show()
```

## À surveiller
- Les résidus de `p` qui stagnent : augmenter `nNonOrthogonalCorrectors` ou `pRefCell`.
- Les pics de résidu : vérifier la continuité et les conditions aux limites.
