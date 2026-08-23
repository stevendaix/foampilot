# Comparaison longue OpenFOAM–JOS-3 et JOS-3 seul

## Protocole

Le cas CFD utilise la géométrie MakeHuman `body-only`, le modèle Boussinesq, la gravité active, le plafond comme unique ouverture et le patch humain `externalCoupledTemperature`. Le pas OpenFOAM vaut `0,05 s`. Le calcul a été lancé jusqu’à 60 s, mais le benchmark a été arrêté après environ 29,5 s physiques pour limiter le temps de calcul ; les 584 échanges JOS-3 complets et comparables correspondent à `29,2 s`.

La référence JOS-3 seule utilise le même mapping de 9 418 faces, les mêmes surfaces nodales et le champ `h` du dernier état CFD disponible, avec `Ta = Tr = 20 °C`, `hr = 4,5 W m⁻² K⁻¹` et `dtime = 0,05 s`. Cette référence n’est donc pas une reproduction exacte de la trajectoire CFD, mais un contrôle physiologique à environnement imposé.

## Résultats CFD couplé

Le couplage effectue 584 échanges avec `dt=0,05 s`, sans exception du pilote. La simulation OpenFOAM progresse jusqu’à environ `29,5 s` avant l’arrêt contrôlé du benchmark. La continuité reste de l’ordre de `10⁻⁴` par pas vers la fin, sans explosion thermique ni erreur de conversion enthalpie-température.

| Échange | Temps équivalent | T retour min | T retour max |
|---:|---:|---:|---:|
| 1 | 0,05 s | 34,00 °C | 34,00 °C |
| 100 | 5,00 s | 33,92 °C | 34,01 °C |
| 200 | 10,00 s | 33,84 °C | 34,03 °C |
| 300 | 15,00 s | 33,76 °C | 34,04 °C |
| 400 | 20,00 s | 33,69 °C | 34,05 °C |
| 500 | 25,00 s | 33,61 °C | 34,06 °C |
| 584 | 29,20 s | 33,55 °C | 34,07 °C |

## Résultats JOS-3 seul

La simulation physiologique seule termine les 584 pas. La température cutanée globale moyenne passe de `34,38 °C` à `34,31 °C`. La plage entre zones reste plus large que dans le retour CFD couplé, de `33,76–35,41 °C` à la fin, car le modèle seul conserve les contrastes de ses 17 nœuds cutanés au lieu de les observer après relaxation vers chaque face CFD.

| Temps | Tsk moyenne | Tsk min | Tsk max | Puissance corps totale | Puissance environnement totale |
|---:|---:|---:|---:|---:|---:|
| 0,05 s | 34,382 °C | 33,868 °C | 35,849 °C | −171,17 W | 179,48 W |
| 5 s | 34,364 °C | 33,936 °C | 35,712 °C | −156,41 W | 178,53 W |
| 10 s | 34,351 °C | 33,926 °C | 35,610 °C | −147,82 W | 177,60 W |
| 15 s | 34,340 °C | 33,887 °C | 35,534 °C | −141,74 W | 176,71 W |
| 20 s | 34,330 °C | 33,845 °C | 35,478 °C | −137,23 W | 175,85 W |
| 25 s | 34,320 °C | 33,801 °C | 35,437 °C | −133,84 W | 175,02 W |
| 29,2 s | 34,312 °C | 33,765 °C | 35,412 °C | −131,64 W | 174,34 W |

## Interprétation

Le calcul couplé long est stable sur près de 30 secondes physiques. La correction `dtime = deltaT` est importante : le modèle physiologique avance désormais au même rythme que le CFD et non plus 20 fois plus vite. Le couplage ne montre ni oscillation divergente ni valeur non finie.

La comparaison brute des températures min/max n’est pas suffisante pour conclure à une erreur JOS-3 : le champ CFD retourné est une température de surface par face, sous-relaxée avec `alpha=0,1`, tandis que JOS-3 seul rapporte ses 17 températures cutanées physiologiques. La bonne comparaison finale doit agréger le champ CFD par zone JOS-3, puis comparer les 17 moyennes pondérées par aire à `model.Tsk` de la référence seule.

Le résultat actuel valide la stabilité opérationnelle du couplage sur `29,2 s`, mais le calcul n’est pas encore un benchmark physiologique strict, car le champ `h` de la référence seule est figé à partir d’un état CFD. Le prochain contrôle scientifique doit enregistrer `h`, `Ta`, `Tr`, la température CFD et les puissances par zone à chaque échange, puis rejouer JOS-3 seul avec les mêmes séries temporelles d’entrée.

## Fichiers

| Fichier | Description |
|---|---|
| `fluid_60s.log` | Trace OpenFOAM du calcul long, arrêt contrôlé vers 29,5 s |
| `jos3_60s.log` | Trace des 584 échanges couplés |
| `jos3_only.csv` | Référence JOS-3 seule sur 29,2 s |
| `jos3_only_29s.log` | Journal de génération de la référence physiologique |
