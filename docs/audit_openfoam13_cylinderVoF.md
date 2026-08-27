# Audit OF13 — multiRegion/film/cylinderVoF

La référence OpenFOAM 13 possède une Allrun sérielle et une Allrun-parallel. La chaîne parallèle exécute `blockMesh -region VoF`, `decomposePar -region VoF -noFields`, `extrudeToRegionMesh -region VoF` en parallèle, `decomposePar -fields -copyZero`, `foamMultiRun` parallèle, `reconstructPar -allRegions` et `paraFoam -touchAll`. Le contrôle couple `VoF compressibleVoF` et `film film`, avec `endTime=20`, `deltaT=1e-2`, `maxCo=0.3`, `maxAlphaCo=1` et ajustement automatique du pas. L’extrusion crée une couche film d’épaisseur `0.01`.

Le runner `185_multiRegion_film_cylinderVoF/run.py` reproduit cette procédure uniquement avec FoamPilot. Il importe `VoF`/`film`, les champs `alpha.liquid`, `T`, `U`, `p`, `p_rgh`, les propriétés air/liquide, `cloudProperties`, `parcelInjectionProperties` et les dictionnaires d’extrusion. Le helper parallèle transmet explicitement `-parallel`, afin que le maillage film soit écrit dans chacun des quatre processeurs.

La validation confirme le maillage VoF/film par processeur et le couplage `compressibleVoF`/film. MULES résout `alpha.liquid` avec des valeurs bornées. Les particules sont injectées puis absorbées par le film; l’extrait final indique plusieurs milliers de parcelles introduites et absorbées, sans erreur de transport. Les solveurs fluides et film convergent sans `FOAM FATAL`.

Le calcul atteint `Time=20 s` et `End`. `reconstructPar -allRegions` reconstruit les champs VoF et film, y compris `alpha.liquid`, le nuage lagrangien et les champs de film, jusqu’à `Time=20 s`; `paraFoam -touchAll` termine sans erreur.

Statut : **validé OF13**.
