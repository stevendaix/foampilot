# P14 — Distance géodésique (Section 14)
## Méthodes testées
- Distance géodésique par graphe (Dijkstra sur maillage surfacique)
- Heat method approché (diffusion + gradient)
- Extrêmes géodésiques (algorithme 14.2)

## Résultats
- Faces frontières totales : **24138**
- Arêtes du graphe : **36192**
- Faces inaccessibles : **10**

## Extrêmes géodésiques
- Extrémité 1 (face 21089) : [0.2161, 0.1891, 0.0437] m
- Extrémité 2 (face 600) : [0.2544, 0.2484, 0.0035] m
- Distance géodésique max : **0.312501 m**
- Distance euclidienne max : **0.081231 m**

## Heat method (face source = extrémité 1)
- Distance min (finie) : 0.000000 m
- Distance max (finie) : 1.000000 m

## Projection sur axe principal
- Axe PCA : [0.5652, 0.8211, -0.0799]
- Étendue axiale : 0.182046 m

## Performance
- Temps d'exécution : **5.12 s**

## Conclusion
Les deux extrémités géodésiques correspondent aux ouvertures du vaisseau.
La distance géodésique (0.3125 m) est cohérente avec la longueur du vaisseau.
