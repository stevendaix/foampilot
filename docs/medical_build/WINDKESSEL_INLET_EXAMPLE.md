# Exemple 3 — Condition Windkessel à l’entrée

## Principe

Le module `foampilot.model_addon.windkessel.Windkessel` résout un modèle lumped cardiovasculaire à partir d’un débit périodique `Q(t)`. Dans cet exemple, le modèle à cinq éléments calcule une pression aortique d’entrée `p1(t)` qui est exportée sous forme de table OpenFOAM sur le patch `aorta_surface_inlet`.

Le système résolu est :

`C dp2/dt + p2/Rp = Q(t)`

et, avec une compliance proximale :

`dp_prox/dt = [Q - (p_prox-p2)/Rc] / Cprox`.

La pression imposée à l’entrée est reconstruite par :

`p1(t) = p_prox(t) + L dQ/dt`.

Toutes les grandeurs sont en unités SI : débit en m³/s, pression en Pa, résistance en Pa·s/m³, compliance en m³/Pa et inertance en Pa·s²/m³.

## Exécution

```bash
PYTHONPATH=foampilot/src python3 examples/medical_build/example_windkessel_inlet.py
```

La commande utilise `examples/coa/data_typec_q.csv`, convertit le débit de ml/s vers m³/s, simule cinq cycles et conserve le dernier cycle pour supprimer le transitoire initial.

| Fichier | Contenu |
|---|---|
| `windkessel_inlet_waveform.csv` | temps, débit et pression d’entrée du dernier cycle |
| `p.windkessel` | champ `p` OpenFOAM avec table `uniformFixedValue` à l’entrée |
| `windkessel_inlet_report.json` | paramètres, amplitudes et statut du solveur |

## Paramètres de l’exemple

| Paramètre | Valeur |
|---|---:|
| `Rc` | `1.0e6 Pa·s/m³` |
| `Rp` | `2.0e9 Pa·s/m³` |
| `C` | `2.0e-7 m³/Pa` |
| `L` | `5.0e3 Pa·s²/m³` |
| `Cprox` | `1.0e-8 m³/Pa` |
| Période du débit | `0.76496 s` |
| Patch | `aorta_surface_inlet` |

Le résultat obtenu est une pression d’entrée comprise entre environ `269.5 kPa` et `270.3 kPa` pour les paramètres conservés du cas coa. Ces valeurs sont un exemple numérique et ne constituent pas une calibration clinique.

## Utilisation dans OpenFOAM

Le fichier `p.windkessel` est un champ de pression complet. Pour l’utiliser dans un cas issu de `openfoam_case`, il faut le copier vers `0/p` après le maillage :

```bash
cp outputs/windkessel_inlet_example/p.windkessel \
   openfoam_case/0/p
```

Le patch d’entrée reçoit alors :

```text
aorta_surface_inlet
{
    type uniformFixedValue;
    uniformValue table ((t0 p0) (t1 p1) ...);
}
```

Le champ `U` conserve une condition de débit `flowRateInletVelocity`. Il faut donc vérifier que le débit imposé par `U` et la pression imposée par `p` représentent bien le même protocole numérique. Les sorties restent à pression de référence nulle et le mur conserve `noSlip` pour `U`.

`Allrun` reste dédié à `surfaceCheck`, `blockMesh` et `snappyHexMesh`. La simulation s’exécute ensuite avec `Allrun_solver`, après remplacement contrôlé de `0/p`.

## Limites et vérifications

Une condition Windkessel est un modèle 0D couplé à une frontière 3D ; elle ne remplace pas une calibration patient-spécifique ni un couplage fort avec le solveur. Avant une étude quantitative, il faut vérifier la conservation du débit, l’absence de réflexion numérique excessive, la stabilité avec le pas de temps et la compatibilité de `p` et `U` avec la convention de pression du solveur.

Le script ne modifie pas le cas asymétrique ni le cas d’entrée standard. Il produit un artefact séparé, ce qui permet de comparer les trois configurations : entrée débit standard, entrée avec déformation géométrique et entrée Windkessel.
