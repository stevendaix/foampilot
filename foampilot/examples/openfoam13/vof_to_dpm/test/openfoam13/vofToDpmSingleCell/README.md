# OpenFOAM 13 — VOF to DPM single-cell test

Ce cas vérifie le premier prototype `vofToDpm` sous OpenFOAM 13. Le champ `alpha.liquid` vaut un dans une cellule cubique de volume unitaire. Le convertisseur doit donc produire un volume liquide égal à `1`, un centroïde égal à `(0.5 0.5 0.5)` et un diamètre sphérique équivalent égal à `(6/pi)^(1/3)`.

Le prototype écrit `constant/cloudPositions` au format `vectorField`, compatible avec le mécanisme `manualInjection` des clouds de parcels OpenFOAM 13, ainsi qu’un rapport `constant/vofToDpmReport`. Il agrège volontairement le volume sélectionné en un seul parcel positionné au centroïde ; la détection de fragments séparés et l’insertion directe dans un `parcelCloud` restent des étapes ultérieures.

## Exécution

Depuis un environnement OpenFOAM 13 configuré :

```sh
./Allrun
```

Le script construit le maillage avec `blockMesh`, exécute `vofToDpm`, puis vérifie automatiquement le volume, le centroïde, le diamètre équivalent et la position écrite.
