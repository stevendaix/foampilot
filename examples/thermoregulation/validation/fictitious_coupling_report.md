# Test de couplage fictif JOS-3/OpenFOAM 13

## Objectif

Le test vise à séparer le protocole d’échange `externalCoupledTemperature` des problèmes propres au maillage humain MakeHuman et à sa configuration CFD. Un cas stable basé sur le tutoriel officiel OpenFOAM 13 `fluid/buoyantCavity` a été utilisé. Le mur `hot` a été remplacé par une condition `externalCoupledTemperature`; les autres conditions et le solveur restent ceux de la référence.

## Résultats

| Test | Résultat | Mesure |
|---|---|---|
| Maillage analytique `buoyantCavity` | Réussi | `blockMesh` puis `createExternalCoupledPatchGeometry T` |
| Échange initial | Réussi | 2 250 faces, température initiale `307,75 K` |
| Échanges transitoires fictifs | Réussi | 5 échanges `data.out → data.in` |
| Coefficient reçu | Stable | `h = 24,0265 W m⁻² K⁻¹` sur le patch |
| Température renvoyée | Stable | `307,750–307,900 K` selon l’itération |
| Solveur OpenFOAM 13 | Réussi | Temps `1–5 s`, `End`, aucune erreur fatale |
| Continuité | Bornée | erreurs locales `1,26e-4` à `6,51e-2`, globales proches de `1e-16` |

Le pilote fictif a lu les données de température et de coefficient d’échange produites par OpenFOAM, puis a renvoyé une température de surface légèrement modulée. OpenFOAM a accepté les cinq retours et a terminé normalement. Le fichier de données contenait 2 250 lignes, soit une ligne par face du patch couplé.

## Conclusion

Le protocole d’échange `externalCoupledTemperature` n’est pas la cause générale du blocage observé sur le cas humain. Il fonctionne sur un cas OpenFOAM 13 stable, avec une géométrie analytique, un patch de taille connue et cinq itérations bidirectionnelles terminées.

Le problème restant est donc localisé au cas humain : maillage snappyHexMesh, conditions limites, échelle et qualité locale du maillage, ou interaction entre la convection naturelle et les valeurs de flux/température générées autour de la géométrie MakeHuman. Le test ne prouve pas encore que le réseau JOS-3 complet est stable sur OpenFOAM, car le pilote utilisé ici est volontairement fictif. L’étape suivante est de remplacer progressivement le pilote fictif par `DistributedSurfaceNetwork` sur ce même cas stable, puis d’exécuter le même protocole sur le maillage humain.

## Reproduction

```bash
cd /home/ubuntu/foampilot/openfoam_runs/fictitious_coupling_buoyantCavity
. /opt/openfoam13/etc/bashrc
blockMesh
createExternalCoupledPatchGeometry T
python3 fake_temperature_driver.py > fake_driver.log 2>&1 &
foamRun > openfoam.log 2>&1
wait
```

Les journaux de l’exécution sont conservés dans le répertoire du cas : `fake_driver.log`, `openfoam.log`, `blockMesh.log` et `createExternalCoupledPatchGeometry.log`.

## Références

[1]: https://doc.cfd.direct/openfoam/user-guide-v13/case-management "OpenFOAM User Guide v13"
[2]: https://github.com/TanabeLab/JOS-3 "JOS-3 repository"
