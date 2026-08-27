# Audit OF13 — potentialFoam/pitzDaily

La référence OpenFOAM 13 exécute `blockMesh -dict $FOAM_TUTORIALS/resources/blockMesh/pitzDaily`, `potentialFoam -writePhi -writep`, puis `foamPostProcess -func streamFunction`. Les champs initiaux sont fournis sous `0/U.orig` et `0/p.orig`, que FoamPilot importe comme `0/U` et `0/p`.

Le runner `237_potentialFoam_pitzDaily/run.py` reproduit la chaîne sérielle avec uniquement les commandes FoamPilot et un environnement OF13 explicite. Il importe la ressource de maillage partagée dans `system/pitzDaily`, convertit les champs `.orig` en champs actifs, puis exécute le solveur potentiel et le post-traitement de la fonction de courant.

La validation est complète. `blockMesh`, `potentialFoam` et `foamPostProcess` terminent avec succès. Le calcul rapporte une erreur de continuité d’environ `2,70e-3` et une erreur de vitesse interpolée d’environ `1,14e-4`. Le champ `phi` est écrit par `-writePhi` et `streamFunction` le lit puis produit le champ `streamFunction`. Aucun `FOAM FATAL` ni avertissement fonctionnel n’apparaît.

Statut : **validé OF13 — écoulement potentiel pitzDaily avec fonction de courant produite**.

Le runner utilise `BaseSolver.run_command(environment=...)` et `BaseSolver.import_reference_asset`; aucune nouvelle API publique n’a été ajoutée.
