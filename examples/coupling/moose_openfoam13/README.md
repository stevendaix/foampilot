# Validation MOOSE–OpenFOAM 13 dans Foampilot

Ce cas reprend le tutoriel officiel `externalCoupledCavity` d’OpenFOAM 13. Le solveur OpenFOAM échange la température avec un participant externe par fichiers texte dans `comms/`. Le fichier `external_participant.py` utilise `foampilot.coupling.ExternalCoupledTemperature` et applique une augmentation de 1 K à chaque échange. Cette opération représente le point d’extension où un transfert MOOSE `MultiApp`/`Transfers` ou un solveur dérivé de `ExternalProblem` fournira la nouvelle température.

Le cas ne dépend pas de preCICE. Il valide séparément les points essentiels de l’intégration : la condition aux limites native OpenFOAM 13, le protocole `OpenFOAM.lock`/`.out`/`.in`, le parsing des patches et la restitution des valeurs mixtes.

## Exécution

```bash
source /opt/openfoam13/etc/bashrc
cd examples/coupling/moose_openfoam13
FOAMPILOT_COUPLING_STEPS=4 ./Allrun
```

Le script officiel `Allclean` supprime les fichiers de calcul et `comms/` :

```bash
./Allclean
```

Pour brancher ensuite MOOSE, remplacer le calcul `next_temperature` dans `external_participant.py` par le transfert des données reçues vers l’application MOOSE, l’exécution du pas MOOSE et la projection des températures sur les faces OpenFOAM. La synchronisation et le format de fichiers Foampilot restent inchangés.
