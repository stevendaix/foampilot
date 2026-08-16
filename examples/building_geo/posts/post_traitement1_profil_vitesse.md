# Post-traitement 1 : Extraire un profil de vitesse avec PyFoam / OpenFOAM

## Objectif
Vérifier le profil logarithmique de vitesse à l'entrée du domaine après calcul, et comparer à la théorie.

## Méthode
- Utiliser `postProcess` ou `sample` avec un `cuttingPlane` à x = x_inlet.
- Extraire `U_z` le long de z.
- Tracer avec matplotlib.

## Code minimal
```python
import os
import numpy as np
import matplotlib.pyplot as plt

case = "/tmp/voxcity_vector_demo3"
os.system(f"postProcess -func sampleDict -case {case} > /dev/null 2>&1")

data = np.loadtxt(f"{case}/postProcessing/sampleDict/0.5/U")  # adapte le temps
z = data[:, 2]
Umag = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)

plt.plot(Umag, z)
plt.xlabel("|U| (m/s)")
plt.ylabel("z (m)")
plt.title("Profil de vitesse à l'entrée")
plt.grid(True)
plt.show()
```

## À surveiller
- La discrétisation du profil : vérifier que le maillage est suffisamment fin près du sol.
- Les effets de bord : la première ligne de cellules doit être dans la sous-couche visqueuse.
