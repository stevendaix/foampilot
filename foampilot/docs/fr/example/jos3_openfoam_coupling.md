# Couplage JOS-3–OpenFOAM

Cette page décrit le couplage thermique entre [JOS-3](https://github.com/TanabeLab/JOS-3) et OpenFOAM dans FoamPilot. Le modèle JOS-3 fournit la physiologie thermique ; OpenFOAM fournit les champs CFD locaux sur la surface humaine. FoamPilot relie les deux modèles avec une couche de surface distribuée.

> **Point important.** Le JOS-3 original possède 85 états physiologiques internes, mais seulement 17 températures cutanées de segment. Les températures indépendantes de chaque face ou point CFD appartiennent à `DistributedSurfaceNetwork`, une extension distribuée explicitement séparée du modèle JOS-3 original.

## Architecture générale

Le couplage comporte trois niveaux :

| Niveau | Nombre d’états | Rôle |
|---|---:|---|
| JOS-3 interne | 85 | Sang central, artères, veines, cœur, muscle, graisse et peau physiologiques |
| Surface JOS-3 | 17 | Une température cutanée par segment : tête, cou, thorax, dos, etc. |
| Surface CFD distribuée | N | Une température dynamique indépendante par face ou point du maillage humain |

La classe principale est `DistributedSurfaceNetwork` :

```python
from foampilot.physiology import (
    DistributedSurfaceNetwork,
    JOS3,
    SurfaceMapping,
)

model = JOS3(ex_output="all")
mapping = SurfaceMapping(
    zone_ids=zone_ids,       # un identifiant 0–16 par face CFD
    areas=node_areas,        # aire duale de chaque face [m²]
    points=points,            # coordonnées optionnelles [m]
)
network = DistributedSurfaceNetwork(model, mapping)
```

Chaque point CFD possède une température `surface_temperature[i]`, une capacité thermique `capacity[i]` et une conductance `anchor_conductance[i]` vers la température cutanée du segment JOS-3 associé.

## Équations de la surface distribuée

Pour un point CFD `i`, le bilan local est :

$$
C_i \frac{dT_{s,i}}{dt} = -Q_{body,i} - Q_{env,i}
$$

avec :

$$
Q_{body,i} = G_i(T_{s,i}-T_{sk,z(i)})
$$

$$
Q_{env,i} = h_i A_i(T_{s,i}-T_{air,i})
$$

où `z(i)` est la zone JOS-3 de la face CFD. Les capacités et conductances sont réparties proportionnellement aux aires :

$$
\sum_{i \in z} C_i = C_{skin,z}
$$

$$
\sum_{i \in z} G_i = G_{skin,z}
$$

Cette répartition conserve les propriétés de chaque segment JOS-3, sans prétendre que le JOS-3 original contient une résolution spatiale qu’il n’a pas.

## Les 17 zones JOS-3

JOS-3 ne reconnaît pas automatiquement la bonne zone à partir de la géométrie OpenFOAM. Le mapping doit être défini explicitement par l’utilisateur ou par une étape de prétraitement géométrique.

| `zone_id` | Nom JOS-3 | `zone_id` | Nom JOS-3 |
|---:|---|---:|---|
| 0 | Head | 9 | RArm |
| 1 | Neck | 10 | RHand |
| 2 | Chest | 11 | LThigh |
| 3 | Back | 12 | LLeg |
| 4 | Pelvis | 13 | LFoot |
| 5 | LShoulder | 14 | RThigh |
| 6 | LArm | 15 | RLeg |
| 7 | LHand | 16 | RFoot |
| 8 | RShoulder | | |

Il n’est **pas nécessaire** de créer 17 patches OpenFOAM. Un seul patch `humanPatch` peut contenir toute la surface humaine, à condition que chaque face possède une ligne de mapping stable :

```text
face CFD → zone_id JOS-3 → aire de face
```

Une autre organisation possible consiste à créer 17 patches nommés `Head`, `Neck`, `Chest`, etc. Cette solution est plus visuelle, mais elle augmente le nombre de conditions limites et n’est pas indispensable.

## Fichier de mapping CSV

L’exemple fourni se trouve dans :

```text
examples/thermoregulation/openfoam_jos3_coupling/openfoam_case/zone_mapping.csv
```

Il contient notamment :

```csv
face_index,zone_id,zone_name,area_m2,temperature_unit,h_unit,flux_unit
0,0,Head,0.010000,K,W/m2/K,W/m2
1,1,Neck,0.010000,K,W/m2/K,W/m2
```

Le chargement côté FoamPilot est contrôlé :

```python
mapping = SurfaceMapping.from_csv(
    "openfoam_case/zone_mapping.csv",
    points=human_patch_points,
)
```

La classe vérifie les colonnes `zone_id` et `area_m2`, la positivité des aires et les unités déclarées pour la température et le coefficient `h`.

## Échange OpenFOAM transitoire

La méthode native OpenFOAM utilisée est `externalCoupled`. Elle synchronise les applications au moyen de fichiers et d’un verrou :

| Fichier | Sens | Unité |
|---|---|---:|
| `h.out` | OpenFOAM → FoamPilot | W/m²/K |
| `air_temperature.out` | OpenFOAM → FoamPilot | K, convertis en °C pour JOS-3 |
| `T.out` | OpenFOAM → FoamPilot, optionnel | K |
| `qJOS3.in` | FoamPilot → OpenFOAM | W/m² |
| `OpenFOAM.lock` | Synchronisation | — |

Le suffixe `.out/.in` est imposé par le protocole natif `externalCoupled`. Il ne s’agit pas d’un choix arbitraire de FoamPilot. Un CSV avec en-tête est possible avec un `functionObject` C++ personnalisé, mais il ne serait pas directement compatible avec le mécanisme natif OpenFOAM.

La configuration générale se trouve dans :

```text
examples/thermoregulation/openfoam_jos3_coupling/openfoam_case/system/controlDict
```

Extrait :

```text
functions
{
    jos3Coupling
    {
        type            externalCoupled;
        libs            (fieldFunctionObjects);
        commsDir        "${FOAM_CASE}/comms";
        initByExternal  true;
        waitInterval    1;
        timeOut         300;
        calcFrequency   1;
        executeControl  timeStep;
        executeInterval 1;

        regions
        {
            region0
            {
                humanPatch
                {
                    writeFields (h air_temperature);
                    readFields  (qJOS3);
                }
            }
        }
    }
}
```

La boucle Python est :

```python
from foampilot.postprocess import OpenFOAMExternalCoupledProvider

provider = OpenFOAMExternalCoupledProvider(
    "case/comms/region0_humanPatch",
    fields=("h", "air_temperature"),
    output_field="qJOS3",
    temperature_unit="K",
)

for _ in range(n_steps):
    fields = provider.read_nodal_fields()
    exchange = network.step(
        fields["h"],
        fields["air_temperature"],
        dtime=delta_t,
    )
    provider.write_nodal_flux(exchange.environment_power)
```

Le schéma temporel est :

```text
OpenFOAM résout le pas n
    → écrit h.out et air_temperature.out
FoamPilot lit les champs
    → applique le mapping face-vers-zone
DistributedSurfaceNetwork avance Tsurface[i]
    → calcule qJOS3[i]
FoamPilot écrit qJOS3.in et recrée OpenFOAM.lock
OpenFOAM reprend au pas n+1
```

## Initialisation

Les valeurs initiales peuvent être fournies directement dans `0/`, comme dans l’exemple :

```text
openfoam_case/0/T
openfoam_case/0/h
openfoam_case/0/air_temperature
```

Les températures OpenFOAM sont en kelvins :

```text
dimensions [0 0 0 1 0 0 0];
```

Le coefficient `h` est en W/m²/K :

```text
dimensions [1 0 -3 -1 0 0 0];
```

Pour une initialisation par l’application externe, `initByExternal true` permet de réaliser le premier échange avant la progression du calcul.

## Mode stationnaire

Pour un calcul stationnaire, on peut d’abord converger OpenFOAM avec une condition thermique initiale, lire les champs de surface, avancer JOS-3 ou le réseau distribué, puis relancer OpenFOAM avec le flux retourné. Une autre possibilité consiste à utiliser `externalCoupled` pour effectuer des itérations explicites jusqu’à convergence.

Un critère de convergence couplé peut contrôler :

$$
\max_i |q_i^{n+1}-q_i^n| < \varepsilon_q
$$

et :

$$
\max_i |T_{s,i}^{n+1}-T_{s,i}^n| < \varepsilon_T
$$

Une relaxation est recommandée lorsque l’échange est fortement couplé :

```text
q_applied = alpha*q_new + (1-alpha)*q_old
```

avec `0 < alpha <= 1`.

## Mode transitoire

En transitoire, `calcFrequency` doit être cohérent avec le pas d’échange. À chaque pas, OpenFOAM fournit les champs locaux, puis FoamPilot avance `DistributedSurfaceNetwork` de `delta_t`. Le flux local est retourné à OpenFOAM pour le pas suivant.

Il est recommandé de conserver les mêmes pas de temps pour OpenFOAM et pour le réseau distribué dans une première validation. Une sous-intégration du modèle physiologique pourra ensuite être introduite si le pas CFD est plus grand que le pas thermique de surface.

## Validation

Les tests associés sont disponibles dans :

```text
examples/thermoregulation/openfoam_jos3_coupling/
```

Les commandes principales sont :

```bash
python3 test_distributed_surface.py
python3 test_external_coupled_provider.py
python3 compare_official_example.py
```

Le premier test vérifie que deux points d’une même zone possèdent des températures différentes et que les capacités sont conservées. Le deuxième vérifie le protocole `.out/.in` et la synchronisation par `OpenFOAM.lock`. Le troisième compare le JOS-3 embarqué avec l’exemple officiel du dépôt JOS-3.

## Limites actuelles

Le provider `OpenFOAMExternalCoupledProvider` utilise le protocole natif par fichiers. Un échange strictement en mémoire dans le même processus OpenFOAM nécessiterait un `functionObject` C++ personnalisé et un bridge C++/Python. La solution actuelle est donc un couplage Python externe synchronisé, robuste et compatible avec les installations OpenFOAM standard.

Le cas minimal fourni documente la structure et le protocole, mais son répertoire `constant/polyMesh` doit être remplacé par le maillage réel. Le mapping des faces doit être construit à partir de ce maillage et ne doit pas être déduit uniquement de l’ordre arbitraire d’un fichier exporté.

## Références

[1]: https://github.com/TanabeLab/JOS-3 "Dépôt JOS-3"
[2]: https://github.com/stevendaix/foampilot "Dépôt FoamPilot"
[3]: https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/externalCoupled/ "OpenFOAM externalCoupled"
[4]: https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/heatTransferCoeff/ "OpenFOAM heatTransferCoeff"
[5]: https://doc.openfoam.com/2306/tools/processing/boundary-conditions/rtm/derived/thermal/externalWallHeatFluxTemperature/ "OpenFOAM externalWallHeatFluxTemperature"
