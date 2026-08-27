# Correction du workflow VMTK-like Python

## Résultat du test dense

L’extracteur Python précédent utilisait `stride=12` et ne produisait que 38 sections pour 417 points centerline. Cette densité était insuffisante pour reconstruire fidèlement les extrémités et les variations de rayon.

Avec `stride=1`, l’extracteur produit 416 sections fermées sur les six branches, ce qui correspond à la densité attendue d’une procédure VMTK basée sur les points centerline.

Cependant, la reconstruction directe de ces 416 sections, branche par branche, suivie d’une union voxelisée à 0,75 mm, donne un volume de 16450,33 unités³ contre 13184,27 unités³ pour la référence VMTK, soit **+24,77 %**. Elle donne aussi une erreur d’aire de **+24,77 %**.

## Interprétation

Le problème n’est donc pas seulement le nombre de sections. Le stride 12 sous-échantillonne la géométrie, tandis que le stride 1 expose le problème inverse : chaque branche est reconstruite sur toute sa trajectoire et les zones communes autour des bifurcations sont comptées plusieurs fois avant l’union. La densité correcte doit être accompagnée d’une partition topologique des branches.

Le workflow corrigé doit suivre cette logique :

```text
Surface VMTK
→ surface triangulée et capée
→ centerlines et arrays
→ sections à chaque station utile
→ graphe des branches et bifurcations
→ découpage des portions communes
→ reconstruction des volumes disjoints
→ union globale
→ validation VTK/VMTK
```

Il ne faut donc pas remplacer aveuglément `stride=12` par `stride=1` dans le recontructeur actuel. Il faut d’abord implémenter la partition des zones de bifurcation, ou utiliser directement la surface VMTK comme référence pour localiser les limites de branches.

## Point concernant VMTK installé

Le filtre natif `vtkvmtkPolyDataCenterlineSections` segmente le processus Python dans l’environnement actuel, y compris avec les fichiers centerline officiels contenant `MaximumInscribedSphereRadius`, `EdgeArray`, `EdgePCoordArray`, `CenterlineIds`, `TractIds`, `Blanking` et `GroupIds`. VMTK reste utilisable pour la lecture, l’écriture STL et la validation des surfaces, mais cette étape native de sections doit être encapsulée ou remplacée par notre implémentation Python stable.

## Conclusion

La meilleure reproduction Python n’est pas un simple cutter dense. Elle doit conserver la densité VMTK, mais aussi la topologie partagée du réseau et une partition unique du volume. Le fallback voxelisé reste utile pour obtenir un STL CFD fermé, mais il ne doit pas être présenté comme une géométrie VMTK identique tant que les volumes par branche et les bifurcations n’ont pas été alignés.
