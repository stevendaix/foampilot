# Patch OpenFOAM 13 — Direct Commit

Ce répertoire contient le correctif minimal requis dans OpenFOAM 13 pour permettre au `fvModel` de créer localement un parcel sans passer par les callbacks collectifs de `InjectionModel`.

Le patch modifie `parcelCloud`, `ParcelCloud<CloudType>` et `parcelCloudList`. Il ajoute `parcelCloud::directParcelData`, la méthode virtuelle `commitDirect()` et le dispatch mono-cloud de `parcelCloudList::commitDirect()`. La construction utilise l’API `meshSearch::New()`, initialise les propriétés cinématiques, thermo lorsque disponibles, puis appelle `addParticle()` exclusivement sur le rang propriétaire. Aucune opération MPI n’est exécutée dans cette méthode.

Depuis la racine des sources OpenFOAM 13, appliquer le patch avec :

```bash
cd /opt/openfoam13/src/lagrangian/parcel
sudo patch -p0 < /chemin/vers/vof_to_dpm/patches/openfoam13/commitDirect.patch
sudo bash -lc 'source /opt/openfoam13/etc/bashrc && cd /opt/openfoam13/src/lagrangian/parcel && wmake libso'
```

Le correctif est volontairement limité au cas mono-cloud par défaut, qui correspond au contrat actuel de `compressibleVoFClouds` et `incompressibleVoFClouds`. Une configuration comportant plusieurs clouds doit introduire un registre de noms dans l’API avant d’utiliser le commit direct.

Le chemin applicatif appelle ensuite `parcelCloudList::commitDirect()` avant tout `evolve()` du cloud. Le résultat local est enregistré comme confirmation et réconcilié collectivement après les commits. Cette séparation garantit que le commit local ne peut pas introduire de collective MPI asymétrique.

## Validation

Le cas `tests/run_thermoDamBreak_parallel.sh` a été exécuté avec deux rangs MPI après reconstruction de `liblagrangianParcel`, `libcompressibleVoFClouds` et `libincompressibleVoFClouds`. Le solveur atteint `End`, produit un commit direct réussi et satisfait l’audit massique et enthalpique via `tests/analyze_thermo_conservation.py`.

Le patch doit être appliqué à l’installation OpenFOAM utilisée par le job CI ou par l’environnement foampilot avant la compilation des bibliothèques de l’exemple.
