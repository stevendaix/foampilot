# Audit OF13 — isothermalFilm/rivuletPanel

Source locale OpenFOAM 13 : `/opt/openfoam13/tutorials/isothermalFilm/rivuletPanel`.

Le cas ne contient pas de script `Allrun`. Sa mise en données comprend les champs `T`, `U`, `delta` et `p`, `constant/g`, `constant/momentumTransport` et `constant/physicalProperties`, ainsi que `system/blockMeshDict`, `controlDict`, `fvSchemes` et `fvSolution`.

Les paramètres de contrôle relevés sont `solver isothermalFilm`, `endTime 5`, `deltaT 1e-04`, `writeInterval 0.02`, `adjustTimeStep yes`, `maxCo 0.2` et `maxDeltaT 5e-3`. Le maillage contient un patch `wall` de type `filmWall`. Le champ `delta` utilise `turbulentInlet` à l’entrée et `filmContactAngle` sur le mur, avec un paramètre `alpha 0.1` et une loi de contact angle. Le solveur est thermodynamique (`heRhoThermo`) et le schéma emploie les opérateurs `filmGauss`.

Faute d’Allrun, le runner reproduit la séquence minimale conventionnelle `blockMesh` puis `foamRun -solver isothermalFilm`, uniquement via FoamPilot, en important les champs et dictionnaires OF13 sans les réécrire. La validation construit le maillage sans erreur et conserve le comportement `filmWall`/`filmContactAngle`; le calcul reste stable jusqu’à `Time≈2,508/5 s` après environ 545 s, avec Courant maximal ≈`0,198` et aucun `FOAM FATAL`. Il est arrêté préventivement pour coût disproportionné avant `End=5 s`, statut accepté avec réserve. Aucune extension d’API n’est nécessaire.
