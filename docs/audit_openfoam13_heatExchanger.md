# Audit OF13 — multiRegion/CHT/heatExchanger

La référence OpenFOAM 13 `multiRegion/CHT/heatExchanger` utilise deux régions : `air` et `porous`. L’Allrun appelle `Allmesh`, puis `foamMultiRun`; la variante `Allrun-parallel` génère le même maillage, décompose séparément les régions, lance `foamMultiRun -parallel` et reconstruit le dernier temps dans chaque région.

Le pipeline de maillage reproduit par le runner FoamPilot est : `blockMesh -region air`, `blockMesh -region porous`, `createZones -region air -dict createZonesDict.1` pour les pales, `createBaffles -region air -dict createBafflesDict`, puis `createZones -region air -dict createZonesDict.2` pour la zone rotor MRF. Les dictionnaires régionaux, champs, propriétés physiques, modèles de turbulence, porosité, transfert inter-région et contrôle global sont importés depuis OF13 sans réécriture de la référence.

Le runner `174_multiRegion_CHT_heatExchanger/run.py` exécute ensuite `decomposePar -region air`, `decomposePar -region porous`, `foamMultiRun -parallel` à quatre domaines, puis `reconstructPar -latestTime` pour `air` et `porous`. Il n’utilise ni `Allmesh`, ni `Allrun`, ni commande shell directe dans la logique du runner; toutes les opérations OpenFOAM passent par les managers et commandes FoamPilot.

La validation OF13 est complète : les deux maillages sont créés, les baffles et les zones rotor sont générés, `foamMultiRun` démarre les solveurs couplés air/porous et atteint `Time=2000 s` suivi de `End`. La reconstruction de `air` et `porous` à `Time=2000 s` se termine avec `End`. Aucun `FOAM FATAL` n’est observé.

Statut : **validé OF13**.
