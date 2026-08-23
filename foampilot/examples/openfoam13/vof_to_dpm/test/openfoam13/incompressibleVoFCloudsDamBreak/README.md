# incompressibleVoFClouds / damBreakLaminar

Ce cas valide le chargement du module VoF incompressible OpenFOAM 13 avec le modèle C++ `incompressibleVoFClouds` et un `collidingCloud` lagrangien natif.

Le lanceur copie le tutoriel officiel `incompressibleVoF/damBreakLaminar`, injecte le modèle dans `constant/fvModels`, ajoute la configuration `cloudProperties` d’un cloud incompressible, crée une position de parcel au centre du domaine, puis exécute :

```text
blockMesh
setFields
foamRun -solver incompressibleVoF
```

La validation vérifie que le solveur `incompressibleVoF`, le modèle `incompressibleVoFClouds`, le `collidingCloud` et l’évolution du cloud sont tous sélectionnés et exécutés.

Le modèle utilise la densité de mélange `rho`, la vitesse `U` et la viscosité cinématique `nu` produites par `incompressibleVoF`. Il construit la viscosité dynamique `mu = rho*nu`, fait évoluer le cloud une fois par index temporel et ajoute la source de quantité de mouvement `SU` du cloud à l’équation de `U`.

Cette étape valide le raccordement solver-side et le couplage cloud→écoulement. La conversion automatique de chaque composante connexe VOF vers un nouveau parcel et la soustraction de sa masse dans `alpha.water` nécessitent encore un injecteur spécifique ou une extension de `parcelCloudList`; elles ne doivent pas être prétendues réalisées par le simple modèle `fvModel` actuel.
