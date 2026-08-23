# Références OpenFOAM 13 exécutées

| Cas | Référence | Résultat |
|---|---|---|
| `buoyantCavity` | Convection naturelle avec profils expérimentaux dans `validation/exptData` | OK |
| `coolingSphere` | CHT transitoire air–cuivre, `Tinitial=296 K`, solide initial à `348 K` | OK |

Le cas humain MakeHuman utilise ces références à deux niveaux. `buoyantCavity` valide le solveur de convection naturelle et la chaîne de comparaison à des mesures. `coolingSphere` valide la chaîne transitoire CHT multi-région. Le corps humain reste une application géométrique et thermophysiologique ; il ne doit pas être présenté comme une validation expérimentale humaine tant qu’un jeu de mesures correspondant n’est pas intégré.
