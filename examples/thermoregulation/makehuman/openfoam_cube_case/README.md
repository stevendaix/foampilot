# Cas OpenFOAM cube autour d’un humain MakeHuman

Ce cas crée un domaine cubique autour du maillage humain MakeHuman. Le maillage de démonstration est converti depuis `base.npz` installé par MakeHuman Community ; il peut être remplacé par `output/makehuman_body_only.stl` exporté par le socket MakeHuman.

Le cube est `[-6,6] x [-10,10] x [-3,6] m` et l’humain mesuré est approximativement `[-4.97,4.97] x [-8.45,8.50] x [-1.10,3.26] m` dans le repère MakeHuman. La surface humaine devient un patch OpenFOAM nommé `human`.

Commandes :

```bash
python3 ../create_openfoam_cube_case.py
source /opt/openfoam13/etc/bashrc
cd openfoam_cube_case
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
```

Après `snappyHexMesh`, installer le champ `T` avec la condition `externalCoupledTemperature` et lancer le pilote FoamPilot. Le mapping des faces OpenFOAM vers les 17 zones JOS-3 doit être calculé après génération du maillage à partir des centres de faces du patch `human`; le mapping STL triangle n’est pas encore l’ordre des faces OpenFOAM.
