# Post-traitement CFD orienté ingénieur

Le module `foampilot.postprocess.monitoring` fournit des sondes et des suivis temporels indépendants du solveur. Il s’appuie sur `FoamPostProcessing` et les fichiers VTK déjà produits par OpenFOAM.

```python
from foampilot.postprocess import CFDMonitor, FoamPostProcessing

post = FoamPostProcessing("cases/pitzDaily")
monitor = CFDMonitor(post)

# Statistiques par temps pour la pression dans le domaine.
pressure = monitor.track_region("p", region="cell")
monitor.export_csv(pressure, "postProcessing/pressure_statistics.csv")

# Suivi de la norme de vitesse au point le plus proche.
velocity_probe = monitor.track_point(
    point=(0.10, 0.02, 0.0),
    field="U",
    magnitude=True,
)
monitor.export_csv(velocity_probe, "postProcessing/velocity_probe.csv")

# Résumé JSON du dernier temps disponible.
summary = monitor.summary(
    fields=["p", "U"],
    region="cell",
    magnitudes=["U"],
)
monitor.export_json(summary, "postProcessing/case_summary.json")
```

`track_region` retourne, pour chaque temps, la moyenne, l’écart-type, les valeurs minimale et maximale, les percentiles 5/50/95 et la RMS. Pour un champ vectoriel comme `U`, le paramètre `magnitude=True` calcule la norme avant les statistiques.

`track_point` utilise la sonde de maillage la plus proche et conserve les coordonnées demandées dans la table de sortie. Cette méthode est adaptée au suivi de `p`, de `T`, de `U` ou de toute grandeur disponible en `point_data`.

Les sorties restent volontairement descriptives : elles ne remplacent pas un bilan de masse ou d’énergie intégré sur une frontière. Pour une validation physique, il est recommandé de compléter les séries par les débits massiques, les flux thermiques et les résidus issus de `functionObjects` OpenFOAM, puis de comparer les bilans avec des tolérances explicites.

## Recommandations d’intégration

Les suivis devraient être exécutés après chaque campagne de calcul et archivés avec le nom du cas, le solveur, la version OpenFOAM, le pas de temps et les paramètres physiques. Les CSV sont adaptés à l’analyse exploratoire et les JSON aux rapports automatisés. En production, il est préférable de figer la liste des champs, l’association point/cellule et les unités attendues dans la configuration du cas.

## Calcul automatique de y+

Pour un calcul fiable, le maillage doit fournir `wallDistance` et `wallShearStress` en `point_data`, ou bien une surface de paroi doit être passée à `calc_y_plus`. La formule utilisée est `y+ = y*u_tau/nu`, avec `u_tau = sqrt(tau_w/rho)`. Les grandeurs attendues sont la distance en mètres, `tau_w` en Pa, `rho` en kg/m³ et `nu` en m²/s.

```python
mesh = post.load_time_step(10)["cell"]
mesh = post.calc_y_plus(
    mesh,
    density=1.225,
    viscosity=1.5e-5,
    wall_distance_field="wallDistance",
    wall_shear_field="wallShearStress",
)
```

## Coefficients de force

`calc_force_coefficients` intègre la traction de pression `-p n` et la traction visqueuse sur les cellules d’une surface. Les normales sont orientées vers l’extérieur par PyVista. Les coefficients sont calculés avec `q_ref = 0.5*rho*U_ref²` et `C = F/(q_ref*A_ref)`.

```python
wall = post.load_time_step(10)["boundaries"]["airfoil"]
forces = post.calc_force_coefficients(
    wall,
    density=1.225,
    reference_velocity=30.0,
    reference_area=1.0,
    drag_direction=(1.0, 0.0, 0.0),
    lift_direction=(0.0, 1.0, 0.0),
    pressure_reference=101325.0,
)
print(forces["Cd"], forces["Cl"])
```

Le champ de cisaillement doit être une traction vectorielle en Pa. En son absence, seule la contribution de pression est intégrée. Pour une surface ouverte, l’orientation des normales doit être contrôlée avant d’interpréter le signe de `Cl` et `Cd`.


## Résultats CFD structurés et bilan de masse

Les résultats destinés aux rapports peuvent être encapsulés dans `ResultMetadata` et `EngineeringResult`. Les métadonnées précisent le cas, le temps, la région ou le patch, le champ, l’association point/cellule, les unités, la méthode et la source.

```python
from foampilot.postprocess import EngineeringResult, ResultMetadata

result = EngineeringResult(
    metadata=ResultMetadata(
        case="motorBike", time=100.0, region="cell", field="U",
        association="cell", units="m/s", method="volume_mean",
        source="OpenFOAMDirectReader",
    ),
    values={"mean": 12.4, "p95": 18.1},
)
result.to_dict()
```

Le bilan de masse suit la convention de normales sortantes : le flux signé est `rho U·n dA`, positif en sortie et négatif en entrée. Les contributions sont regroupées par patch.

```python
from foampilot.postprocess import mass_balance

balance = mass_balance({
    "inlet": {"normals": n_in, "areas": a_in, "velocity": U_in},
    "outlet": {"normals": n_out, "areas": a_out, "velocity": U_out},
}, density=1.225)
print(balance["net_mass_flux"])
```

Un bilan proche de zéro est nécessaire mais ne suffit pas à démontrer la convergence. Il doit être suivi avec les résidus, les grandeurs intégrées (`Cd`, `Cl`, `Cm`) et une fenêtre temporelle statistiquement stationnaire.
