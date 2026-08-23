# Sphere Meshing with Layer Generation

## Objet

Ce tutoriel de Tobias Holzmann montre la génération d’un maillage autour d’une sphère placée dans un canal, avec extraction des features du canal et ajout de couches prismatiques sur la surface de la sphère. Il constitue un cas fondamental pour comprendre l’enchaînement `snappyHexMesh` : découpe du domaine, raffinement, snapping puis inflation de couches [1].

## Portage FoamPilot

Le runner écrit les dictionnaires du cas avec `OpenFOAMDictAddFile.write_raw`, copie le maillage UNV et les surfaces CAD, reconstruit `channel.stl` à partir des trois surfaces de canal (`inlet`, `outlet`, `wall`) et copie `sphere.stl`. Il exécute ensuite `ideasUnvToFoam` puis `snappyHexMesh -overwrite` via `Solver.run_command`.

Le cas source appelle l’ancien utilitaire `surfaceFeatureExtract`. Sous OpenFOAM 13, cet utilitaire est remplacé par `surfaceFeatures` et le dictionnaire historique `surfaceFeatureExtractDict` n’est pas directement accepté par cette nouvelle commande. L’archive officielle fournit déjà `channel.eMesh`, qui est précisément l’artefact consommé par `snappyHexMesh`; le runner le conserve donc comme actif du cas et n’appelle pas un utilitaire incompatible. L’audit FoamPilot n’a identifié aucune méthode manquante et aucune extension du cœur n’a été ajoutée.

## Workflow exécuté

```text
ideasUnvToFoam cad/backgroundMesh.unv
snappyHexMesh -overwrite
```

La commande `snappyHexMesh` utilise `channel.eMesh`, découpe la sphère dans le domaine fluide, effectue le snapping et ajoute les couches paramétrées sur le patch `sphere`.

## Résultats OpenFOAM 13

| Vérification | Résultat |
| --- | --- |
| Maillage de fond | 44 541 points, 40 000 cellules et 8 800 faces de frontière convertis avec succès. |
| Raffinement/castellation | Le maillage raffiné et sous-échantillonné est écrit sans erreur. |
| Snapping | Le maillage snappé atteint 48 940 cellules, 154 574 faces et 57 032 points. |
| Couches | La phase d’ajout de couches est exécutée ; 3 720 faces internes de couche sont écrites. |
| Maillage final | 53 404 cellules, 169 262 faces et 62 804 points. |
| Fin du workflow | `snappyHexMesh` termine normalement avec `End`. |

Le cas est **validé comme workflow de maillage**. Il ne comporte pas de solveur dans le workflow source ; la validation porte donc sur la conversion, le raffinement, le snapping et la génération des couches.

## Limites

La validation confirme l’exécution OpenFOAM 13 et la présence de couches dans le maillage. Elle ne constitue pas une étude de qualité indépendante ni une comparaison visuelle automatisée avec les figures du tutoriel. Les sorties de maillage sont régénérables en exécutant `python3 run.py` depuis le dossier du cas avec OpenFOAM 13 chargé.

## Référence

[1]: https://holzmann-cfd.de/community/training-cases/sphere-meshing — Tobias Holzmann, *Sphere Meshing*.
