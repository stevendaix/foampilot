# Audit OF13 — potentialFoam/cylinder

La référence OpenFOAM 13 exécute `blockMesh`, `potentialFoam -functionObjects -writePhi -writep`, puis `foamPostProcess -func streamFunction`. Les champs initiaux sont fournis sous `0/U.orig` et `0/p.orig`, que FoamPilot importe comme `0/U` et `0/p`.

Le runner `236_potentialFoam_cylinder/run.py` reproduit la chaîne sérielle avec uniquement les commandes FoamPilot et un environnement OF13 explicite. Il importe les champs `.orig` sans les suffixes, les dictionnaires `system` et `constant`, puis conserve `phi` produit par `potentialFoam` afin que le calcul de la fonction de courant puisse s’exécuter.

La validation est complète. `blockMesh`, `potentialFoam` et `foamPostProcess` terminent avec succès. Le calcul rapporte une erreur de continuité d’environ `1,35e-4`, une erreur de vitesse interpolée d’environ `1,19e-5` et un rayon de cylindre calculé proche de `0,4996 m`. Le champ `phi` est écrit par `-writePhi` et `streamFunction` le lit puis produit le champ `streamFunction`. Aucun `FOAM FATAL` ni avertissement fonctionnel ne subsiste.

Statut : **validé OF13 — écoulement potentiel autour d’un cylindre avec fonction de courant produite**.

Le runner utilise `BaseSolver.run_command(environment=...)`; aucune nouvelle API publique n’a été ajoutée.
