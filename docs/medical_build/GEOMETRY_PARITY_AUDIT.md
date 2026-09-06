# Audit de l’affirmation de géométrie identique

## Correction nécessaire

L’affirmation selon laquelle la géométrie foampilot était identique à la géométrie VMTK était trop large et doit être corrigée. Les résultats démontrent uniquement une identité dans certains contextes précis : les fixtures copiées sont identiques aux fichiers source, et la lecture puis l’écriture du même `vtkPolyData` par VMTK conserve exactement la géométrie. En revanche, la reconstruction foampilot à partir des branches et sections n’est pas géométriquement identique à `aorta-surface.stl`.

## Comparaison directe

| Métrique | Surface officielle VMTK | STL global foampilot | Écart foampilot |
|---|---:|---:|---:|
| Points | 6468 | 30016 | Maillage foampilot beaucoup plus dense |
| Cellules | 12932 | 60032 | Maillage foampilot plus dense |
| Aire | 4517,7631 | 5643,6666 | **+24,92 %** |
| Volume | 13184,2667 | 11238,0990 | **−14,76 %** |
| Composantes | 1 | 1 | Identique topologiquement |
| Arêtes frontière | 0 | 0 | Identique topologiquement |
| Arêtes non-manifold | 0 | 0 | Identique topologiquement |
| Étendue X | 34,3040 | 34,0000 | Foampilot légèrement plus court |
| Étendue Y | 83,9568 | 77,0000 | Foampilot perd environ 6,96 unités |
| Étendue Z | 22,9052 | 21,5000 | Foampilot perd environ 1,41 unité |

## Ce qui est correct

Le STL foampilot est propre au sens topologique et CFD de base. Il est fermé, connecté en une composante, sans arêtes frontière, sans arêtes non-manifold et avec des normales cohérentes. Le format STL et l’écriture du fichier ne sont pas responsables de l’écart : VMTK réécrit sa propre surface sans modifier aucune métrique.

La structure générale des six branches est également représentée, et l’union voxelisée réussit à créer un volume unique au lieu de six surfaces indépendantes.

## Ce qui est faux ou incomplet

La géométrie n’est pas identique en volume. Le STL foampilot manque environ 1946,17 unités³ par rapport à la surface VMTK.

La géométrie n’est pas identique en aire. L’aire foampilot est supérieure d’environ 1125,90 unités², ce qui indique une surface plus irrégulière, plus dentelée ou artificiellement enrichie par la voxelisation et les raccordements.

Les bornes ne sont pas identiques. La perte principale est suivant Y : la reconstruction foampilot couvre environ 77 unités contre 83,96 pour VMTK. Les extrémités ou les branches ne s’étendent donc pas jusqu’aux mêmes positions.

Les sections ne sont pas encore validées comme les sections VMTK natives. Une section fermée n’implique pas qu’elle possède la même aire, le même centre, la même orientation, le même périmètre ou le même nombre de points que la section VMTK.

Les six branches n’ont pas encore été comparées individuellement à une segmentation VMTK correspondante. L’écart global ne permet donc pas de savoir quelle branche perd le plus de volume.

La fermeture morphologique améliore la connectivité mais ne rétablit pas automatiquement la matière perdue. Elle peut aussi déplacer la surface localement et augmenter l’aire.

## Origine probable des écarts

La première cause probable est la limitation de la reconstruction manuelle par sections : les sections exportées ne couvrent pas exactement toutes les stations et les extrémités utilisées par la surface VMTK.

La deuxième cause probable est la coupure des branches. Les branches générées sont fermées séparément, puis réunies. Cette procédure peut supprimer ou recouvrir des volumes autour des bifurcations au lieu de reproduire la surface anatomique continue.

La troisième cause est l’orientation et la position des plans de section. Une erreur de centre, de tangente ou de repère local modifie l’aire de la section et l’étendue longitudinale de la branche.

La quatrième cause est le traitement voxelisé. Il garantit la topologie, mais introduit une surface en escalier ou une sur-approximation de l’aire. Il ne doit pas être utilisé comme preuve de parité géométrique exacte.

## Statut corrigé

| Niveau | Statut |
|---|---|
| Identité des fichiers de référence copiés | Démontrée |
| Conservation par le writer STL VMTK | Démontrée à 0 % d’écart |
| Validité topologique du STL foampilot | Démontrée |
| Identité des volumes | Non démontrée ; réfutée par −14,76 % |
| Identité des surfaces | Non démontrée ; réfutée par +24,92 % d’aire |
| Identité des bornes | Réfutée |
| Parité branche par branche | Encore à réaliser |
| Parité des sections | Encore à réaliser |

La formulation correcte est donc : **foampilot possède une reconstruction topologiquement valide et structurellement comparable à VMTK, mais elle n’est pas encore géométriquement identique à VMTK**.
