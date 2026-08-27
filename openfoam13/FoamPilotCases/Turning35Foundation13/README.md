# Turning35Foundation13

Cas de manœuvre marine construit pour **OpenFOAM Foundation 13** et `marineFoam`.

Le cas regroupe une géométrie hull/rudder, un mouvement rigide 6-DoF du navire,
un modèle de propulsion Foundation 13 `actuationDisk`, une surface libre eau/air et les
sorties de forces et moments. Il s’agit d’abord d’un cas de validation courte et
reproductible ; les valeurs hydrodynamiques finales nécessitent une étude de
convergence en temps et en maillage.

## Reproduction

```sh
source ../../marine_env.sh
python3 ../../build_turning35_foampilot.py
./Allmesh.FoamPilot
./Allrun
```

`Allclean` supprime uniquement les maillages, temps calculés, processeurs et
sorties de post-traitement. Aucun maillage généré n’est versionné.

## Donor/receveur

Ce cas est actuellement une validation mono-région du mouvement et de la
propulsion. Le couplage overset/inter-mailles est validé séparément par les
harnesses `marineInterMesh*` et le cas DTC multi-région ; il ne doit pas être
présenté ici comme un overset natif complet tant que la conservation de flux et
la classification hole/fringe n’ont pas été validées sur Turning35.
