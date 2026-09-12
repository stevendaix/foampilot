# Rapprochement du cas réel UrbGEN

## Résultat intermédiaire

Le scénario `plot_urbgen_realistic_district.py` ne génère plus un site rectangulaire avec une grille unique. Il utilise désormais **11 parcelles**, un noyau central plus dense, des îlots périphériques, une parcelle Courtyard, des typologies différenciées par îlot et une vue 3D colorée par hauteur.

La sortie actuelle contient **39 bâtiments** répartis sur **11 îlots**. Les hauteurs sont limitées à 70 m pour éviter les tours artificiellement disproportionnées dues à la combinaison de petites emprises et de FAR élevés.

## Écart avec les références originales

Cette version est plus proche de la structure visible dans `urbgen-alternatives.png` et `urbgen-generative.gif`, car elle représente plusieurs sous-quartiers et un noyau de densité. Elle ne constitue cependant pas encore une parité 1:1 : les parcelles sont encore synthétiques, les rues ne sont pas issues du fichier `.gh`, la distribution des centroïdes n’est pas encore le port complet de `UrbGENPopulateRegion`, et les formes contextuelles existantes autour du quartier ne sont pas importées depuis le modèle Rhino.

La prochaine correction de fond doit donc porter sur la population : introduire un composant Python autonome équivalent à `PopulateRegion` avec les modes Random, Regular Grid, Jittered Grid et Staggered Grid, puis transmettre ces points à `generate_urbgen` par parcelle. La fidélité visuelle dépend davantage de cette étape et de la géométrie réelle des îlots que d’une augmentation isolée du BCR.
