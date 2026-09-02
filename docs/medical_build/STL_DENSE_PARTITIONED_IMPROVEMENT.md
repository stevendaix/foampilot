# Amélioration STL par sections denses et correction morphologique

## Méthode testée

La reconstruction a été relancée avec une section à chaque station centerline utile au lieu du sous-échantillonnage `stride=12`. Les six branches ont été reconstruites avec 64 points par contour, voxelisées à 0,75 mm, réunies, puis soumises à une fermeture morphologique d’une itération et une érosion morphologique d’une itération.

L’érosion n’est pas présentée comme une opération anatomique exacte. Elle sert ici de correction contrôlée du sur-volume produit par les branches reconstruites indépendamment. Le résultat doit rester comparé à la surface VMTK officielle.

## Résultat par rapport à VMTK

| Métrique | VMTK officiel | Nouveau STL | Écart |
|---|---:|---:|---:|
| Volume | 13184,2667 | 13148,3848 | **−0,27 %** |
| Aire | 4517,7631 | 5128,4337 | **+13,52 %** |
| Composantes | 1 | 1 | Identique |
| Arêtes frontière | 0 | 0 | Identique |
| Arêtes non-manifold | 0 | 0 | Identique |
| Normales cohérentes | Oui | Oui | Identique |

Le résultat est nettement meilleur que l’ancien candidat 0,5 mm avec fermeture minimale, qui présentait −14,76 % d’erreur de volume et +24,92 % d’erreur d’aire.

## Fichier produit

`examples/medical_build/outputs/aorta_six_branch_union_dense_partitioned_0p75mm.stl`

Le volume est maintenant quasiment égal au volume VMTK, mais l’aire reste supérieure de 13,52 %. Le maillage est donc topologiquement propre et volumétriquement bien calibré, sans être encore une reproduction surface-par-surface exacte.

## Interprétation

L’amélioration confirme que le sous-échantillonnage des sections était une cause importante de l’écart. Toutefois, le bon volume obtenu ne prouve pas à lui seul une parité anatomique : l’érosion compense une partie du sur-volume des zones recouvertes, mais elle peut déplacer la surface localement. La prochaine étape de qualité est une comparaison de distances de surface et une comparaison par branche, pas un simple ajustement supplémentaire du volume.

La variante à deux érosions n’est pas retenue : elle atteint une aire proche de VMTK, mais détruit la connectivité et produit cinq composantes.
