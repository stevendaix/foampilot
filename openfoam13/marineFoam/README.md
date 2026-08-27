# marineFoam — Foundation OpenFOAM 13

`marineFoam` est un driver Foundation OpenFOAM 13 pour les cas marins monophasés ou VoF. Il reprend le mécanisme officiel des modules de solveur : le module est sélectionné par l’entrée `solver` de `system/controlDict` ou par `-solver` en ligne de commande, puis la boucle PIMPLE native est exécutée.

Exemple minimal pour un cas de surface libre :

```text
application     marineFoam;
solver          incompressibleVoF;
```

Le driver permet donc d’utiliser `incompressibleVoF` avec `dynamicMeshDict`, `fvModels`, `fvConstraints`, `momentumTransport`, `phaseProperties` et `physicalProperties.<phase>` Foundation 13. Les forces, la turbulence, la gravité et le mouvement de maillage restent définis par les dictionnaires OpenFOAM du cas ; le driver ne duplique pas les équations Foundation.

## Compilation

Après avoir chargé l’environnement OpenFOAM Foundation 13 :

```sh
cd openfoam13/marineFoam
wmake
```

L’exécutable est installé dans `$FOAM_USER_APPBIN/marineFoam`.

## Utilisation

```sh
marineFoam -case path/to/case
marineFoam -case path/to/case -solver incompressibleVoF
```

Le cas doit fournir les fichiers standards du module choisi. Pour un cas VoF mobile, cela comprend typiquement `0/{U,p_rgh,alpha.water,k,omega,nut}`, `constant/{g,hRef,phaseProperties,physicalProperties.water,physicalProperties.air,momentumTransport,dynamicMeshDict}` et `system/{controlDict,fvSchemes,fvSolution}`.

## Position par rapport aux trois références

Le driver couvre la base Foundation 13 nécessaire au cas DTC moving sans overset, en utilisant `incompressibleVoF` et les mécanismes de mouvement natifs disponibles. Il ne prétend pas encore fournir `dynamicOversetFvMesh`, `overInterDyMFoam` ou les bibliothèques `maneuveringLib` de l’écosystème OpenCFD.

La reproduction de `maneuveringLib` nécessitera un module supplémentaire pour les corps parent-enfant, le gouvernail, les sources de propulsion et les contrôleurs PID. La reproduction de `propeller-OpenFOAM` nécessitera un pipeline de maillage rotor/stator, une `cellZone` rotor et la création des interfaces AMI. La reproduction de `DTCMoving_Overset` nécessitera soit un portage Foundation 13 de l’overset, soit une adaptation documentée vers un mouvement de maillage non-overset.

Cette séparation est volontaire : elle évite de présenter un driver Foundation 13 compatible comme une implémentation automatique de fonctionnalités runtime qui n’existent pas sous les mêmes noms dans Foundation 13.
