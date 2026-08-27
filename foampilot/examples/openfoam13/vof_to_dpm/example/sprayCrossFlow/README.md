# Exemple spray VOF-to-DPM : `sprayCrossFlow`

Cet exemple porte dans **foampilot** le cas `crossFlow` du dépôt [atomizationFoam](https://github.com/imfd-stroemungsmechanik/atomizationFoam), référence GitHub orientée vers l’atomisation de spray. La géométrie représente une buse de liquide débouchant dans un écoulement d’air transversal. Le jet est d’abord résolu par VOF ; les fragments liquides détachés peuvent ensuite être convertis en parcels par `vofFragmentInjection`.

Le cas original ciblait des versions anciennes ou commerciales d’OpenFOAM et utilisait un solveur atomizationFoam monolithique. Cette version conserve sa géométrie STL, son écoulement cross-flow et ses conditions limites liquide/air, mais utilise le solveur `foamRun -solver incompressibleVoF` d’OpenFOAM 13 et le modèle `incompressibleVoFClouds` de foampilot. Tous les fichiers nécessaires au calcul sont versionnés dans ce répertoire : le cas ne fusionne aucun tutoriel et ne nécessite aucun clone GitHub au moment de l’exécution. Les dictionnaires `phaseProperties` et `physicalProperties.*` sont au format OpenFOAM 13 ; le vieux `transportProperties` de la référence n’est volontairement pas utilisé.

## Exécution

Depuis la racine du dépôt :

```sh
cd examples/openfoam13/vof_to_dpm/example/sprayCrossFlow
./Allrun
```

Le script construit une copie temporaire du cas autonome, exécute `blockMesh`, `snappyHexMesh` et le solveur jusqu’à environ `0.01 s`, puis vérifie la sélection du solveur, du modèle `incompressibleVoFClouds`, du cloud et la fin normale du calcul. Le pas de temps est adaptatif ; la valeur finale peut donc être légèrement inférieure à `0.01 s`.

Pour conserver le cas calculé, les journaux et les résultats de post-traitement :

```sh
KEEP_CASE=1 ./Allrun
```

Le chemin du cas temporaire est alors imprimé sous la forme `CASE_DIR=...`.

## Comment vérifier que le portage fonctionne réellement

Le script exécute automatiquement `postprocess.py` après le solveur. Ce post-traitement lit les sorties réelles d’OpenFOAM et ne se contente pas de rechercher un code retour : il lit l’intégrale volumique `∫ alpha.water dV` produite par `volFieldValue`, extrait du journal le premier volume de fragment et la masse du parcel, puis vérifie `m_parcel = rho_liquid × V_fragment`.

Le répertoire `postProcessing/` contient alors `spray_balance.json`, `spray_balance.csv` et `spray_liquid_volume.png`. Le champ `conversion_mass_balance_pass` doit être `true`, avec une erreur relative inférieure à `1e-10`, et `solver_end_pass` doit également être `true`. La figure permet de contrôler visuellement l’entrée progressive du liquide dans le domaine ; le CSV permet une analyse indépendante avec Python, pandas ou Excel.

## Conversion VOF-to-DPM

La conversion est activée dans `constant/fvModels` par `consumeAlpha true` et dans `constant/cloudProperties` par le modèle `vofFragmentInjection`. Le modèle détecte les composantes liquides détachées du champ `alpha.water`, construit un parcel équivalent et retire le volume converti du champ VOF. Le seuil `minVolume 1e-8` est volontairement bas pour rendre l’exemple utile à l’exploration de la fragmentation ; pour une étude de production, il doit être calibré avec la résolution de la maille et la taille minimale de goutte acceptable.

Le test spray a également conduit à corriger le rafraîchissement dynamique de `vofFragmentInjection` : une détection vide au premier pas ne doit pas empêcher l’injection des fragments apparaissant après l’entrée du jet dans le domaine.

Le cas utilise une surface STL et un raffinement local autour de la buse. Le dictionnaire `snappyHexMeshDict` est adapté à la syntaxe OpenFOAM 13 (`type triSurface` avec une clé `file`) et la précision d’écriture est cohérente avec la tolérance de fusion de la maille.

## Validation

La validation minimale attend les marqueurs suivants dans `log.foamRun` :

| Vérification | Attendu |
|---|---|
| Solveur | `Selecting solver incompressibleVoF` |
| Modèle fvModel | `Selecting finite volume model type incompressibleVoFClouds` |
| Cloud | `Selecting parcelCloud collidingCloud` |
| Fin du calcul | présence de `End` après environ `0.01 s` |
| Détection | au moins une ligne `VOF fragments detected: 1` |
| Conversion | un état final avec `Current number of parcels = 1` |

La validation réalisée sous OpenFOAM 13 passe sans erreur fatale, atteint `End`, détecte un fragment dont le volume convertible est d’environ `2.42e-05 m3` et crée un parcel d’environ `0.00679141 kg`.

La référence GitHub est [imfd-stroemungsmechanik/atomizationFoam](https://github.com/imfd-stroemungsmechanik/atomizationFoam). Sa publication associée est Heinrich et Schwarze, « 3D-coupling of Volume-of-Fluid and Lagrangian particle tracking for spray atomization simulation in OpenFOAM », SoftwareX 11 (2020), [DOI 10.1016/j.softx.2020.100483](https://doi.org/10.1016/j.softx.2020.100483).
