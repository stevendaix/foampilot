# Foundation 13 — travaux restant à réaliser

## État de référence

La branche `port/openfoam-foundation-13-minimal-fsi` contient le portage natif Foundation 13, la matrice bloc LDU, l’échange processor via `PstreamBuffers`, le modèle Newton autonome, le gradient de face co-discret, la tangente matériau `mat66` et l’instrumentation du résidu Piola. Les derniers commits sont `885e0dc`, `08858d8`, `6b9c743`, `21e0eb4`, `f60a6ba` et `9c6caee`.

La compilation des bibliothèques réussit avec OpenFOAM Foundation 13. Les tests abstraits de `Amul` réussissent en série et en MPI. Le test local du noyau Piola est validé. En revanche, le test physique global `blockPunch.foundation13` n’est pas validé : avec une direction J·v non affine, l’erreur relative observée est `0.999882`, la corrélation est `0.04086`, `normK=0.0189495` et `normFD=0.240696`.

## Priorité P0 — rendre le test J·v local décisif

Il faut ajouter un test par face interne qui compare, pour les six colonnes Voigt, `piolaIncrement()` à une différence finie de la même réponse constitutive. Le test doit imprimer la face, la colonne, `dGradD`, `dSigma`, `dP`, le flux analytique, le flux par différence finie et l’erreur relative. Les pas `epsilon` doivent être testés au minimum à `1e-4`, `1e-6` et `1e-8`.

Le critère d’acceptation est une erreur locale décroissante avec `epsilon`, puis un plateau de précision, sans permutation entre `XY`, `XZ` et `YZ`. Une colonne qui échoue seule indique un problème Voigt ; toutes les colonnes qui échouent indiquent un problème de transformation Piola ou de convention de gradient.

## Priorité P0 — finaliser la décomposition du résidu

L’instrumentation `PiolaResidualDiagnostics` doit être complétée par une décomposition de `J·v` et de `J_fd·v` pour les catégories suivantes : faces internes, patches de déplacement imposé, patches de traction prescrite et faces processor. Les normes doivent être calculées sur `K·v`, `J_fd·v` et leur différence, pas seulement sur le résidu de base.

Le champ temporel peut convertir les patches prescrits en `calculated`. La nature et la charge prescrite des frontières doivent donc être conservées explicitement pendant l’initialisation du modèle, et non déduites uniquement du type du champ temporel courant.

## Priorité P1 — vérifier les quatre blocs internes

Pour chaque face interne, vérifier numériquement :

```text
KPP = dRowner/dDowner
KPN = dRowner/dDneighbour
-KPP = dRneighbour/dDowner
-KPN = dRneighbour/dDneighbour
```

Les différences finies doivent être appliquées avec une perturbation locale d’un seul degré de liberté, puis comparées aux blocs avant leur insertion dans `diag`, `lower` et `upper`. Le test doit isoler le signe, la transposition, le poids géométrique et la convention owner-neighbour.

## Priorité P1 — vérifier les faces processor

Le test MPI doit utiliser des valeurs distantes marquées et vérifier séparément la contribution locale du propriétaire et la contribution distante de `coupleUpper`. Il faut confirmer la permutation des faces, l’ordre des valeurs reçues par `PstreamBuffers`, la contribution locale `-dPhi/dDowner` et la contribution distante `dPhi/dDremote`.

Le critère d’acceptation est l’égalité entre `Amul()` et une référence distribuée, puis l’égalité entre les blocs processor et les différences finies du résidu avec deux et au moins quatre rangs.

## Priorité P1 — vérifier les conditions limites

Pour `fixedDisplacement`, le gradient de face dépend de la valeur propriétaire et une tangente de flux peut être assemblée. Pour `solidTraction` héritant de `fixedGradient`, le gradient est prescrit ; la traction prescrite est une charge morte et ne doit pas être additionnée à un flux Piola extrapolé. Cette distinction doit être testée sur un mini-cas comportant une seule face de chaque type.

## Priorité P2 — réactiver le solveur linéaire

BiCGStab doit rester désactivé comme critère scientifique tant que `J·v` n’est pas validé. Après obtention d’une erreur globale inférieure à `1e-6`, il faudra réactiver progressivement Jacobi, ILU(0), puis le line-search Newton. Les critères sont une décroissance monotone du résidu linéaire, l’absence de `rho=0`, et une convergence Newton quadratique sur un cas suffisamment petit.

## Priorité P2 — validation FSI

Le cas `beamInCrossFlow` ne doit être utilisé qu’après validation de `blockPunch` et du test MPI physique. Il faudra ensuite distinguer une instabilité fluide du couplage structurel, vérifier la conservation des efforts à l’interface et comparer Aitken/IQN-ILS sur un cas de référence.

## Critères de clôture

La PR pourra être déclarée techniquement complète pour le portage Foundation 13 lorsque les conditions suivantes seront satisfaites :

| Critère | Seuil attendu |
|---|---:|
| Compilation propre Foundation 13 | bibliothèques et tests compilés |
| Test local Piola | erreur relative au plus `1e-6` |
| Test global J·v | erreur relative au plus `1e-6` |
| Corrélation globale | au moins `0.999999` |
| Test `Amul` MPI | résultats identiques à la référence |
| Blocs processor | concordance FD au plus `1e-6` |
| Newton blockPunch | convergence sans breakdown |
| BiCGStab/ILU(0) | convergence reproductible |
| beamInCrossFlow | exécution FSI sans SIGFPE |

Tant que les critères J·v et Newton ne sont pas atteints, la PR doit rester présentée comme une intégration Foundation 13 en validation active, et non comme une validation FSI finale.
