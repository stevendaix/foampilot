# Couplage OpenFOAM–JOS-3 en mémoire

La version recommandée est désormais le couplage Python en mémoire avec le JOS-3 embarqué dans FoamPilot. Le lecteur OpenFOAM par fichiers reste disponible comme adaptateur de compatibilité, mais il n’est plus nécessaire pour le couplage principal.

Le modèle embarqué se trouve dans `foampilot/src/foampilot/physiology/jos3/`. Il reprend le modèle JOS-3 validé et expose en plus `set_external_heat_flux`, `external_heat_flux` et `clear_external_heat_flux`.


Ce cas montre l’intégration de la classe `OpenFOAMJOS3Coupler` dans FoamPilot. L’échange thermique est réalisé à partir de trois champs scalaires OpenFOAM associés aux nœuds de la surface humaine : `h` en W/m²/K, `Ta` en °C et `T` en °C.

La loi utilisée est

```text
q_out = h * (T - Ta)       [W/m²]
q_body = -q_out            [W/m²]
```

Le flux `q_out` est positif du corps vers le fluide. Le flux `q_body` est positif du fluide vers le corps et est intégré sur l’aire duale de chaque nœud avant d’être regroupé sur les 17 segments JOS-3. La puissance segmentée est injectée dans `model.ex_q` sur le tissu cutané.

## Exécution de la validation

Le test mémoire autonome se lance ainsi :

```bash
cd /home/ubuntu/foampilot
python3 examples/thermoregulation/openfoam_jos3_coupling/test_memory_coupling.py
```

Il vérifie un échange steady par moyenne pondérée sur les zones et un échange transitoire à chaque pas avec un provider Python simulant OpenFOAM.


Depuis le dépôt FoamPilot, avec le dépôt JOS-3 présent à côté :

```bash
PYTHONPATH=/home/ubuntu/JOS-3/src \
python3 examples/thermoregulation/openfoam_jos3_coupling/validate_coupling.py
```

Le test reproduit d’abord un cas JOS-3 de référence avec `q=0`, puis compare les températures cutanées et centrales du modèle natif et du modèle couplé. Il vérifie ensuite l’intégration analytique d’un flux de 40 W/m². Le résultat attendu est une puissance de `-0.8 W` par segment dans le maillage synthétique fourni.

## Utilisation avec un cas OpenFOAM réel

Le branchement recommandé avec OpenFOAM Python est basé sur deux callbacks, sans fichier intermédiaire :

```python
from foampilot.physiology import (
    CallbackFieldProvider, JOS3, JOS3NodeCoupler, SurfaceMapping,
)

model = JOS3(ex_output="all")
coupler = JOS3NodeCoupler(model, SurfaceMapping(zone_ids, areas, points))
provider = CallbackFieldProvider(
    reader=lambda: {
        "h": h_nodes,
        "surface_temperature": T_nodes,
        "air_temperature": Ta_nodes,
    },
    writer=lambda q_nodes: openfoam_boundary.set_heat_flux(q_nodes),
)
coupler.run_transient(provider, dtime=1.0, steps=100)
```

Pour un calcul stationnaire, l’échange est effectué ponctuellement avec `step_steady(...)`. Pour un calcul transitoire, `run_transient(...)` relit les champs et écrit le flux sur tous les points à chaque pas. JOS-3 reçoit ensuite la puissance intégrée par zone, car sa matrice physiologique possède 17 nœuds cutanés locaux.


Le mapping doit être construit à partir du maillage humain. `segment_ids[i]` est l’indice JOS-3 du nœud `i`, selon l’ordre `Head`, `Neck`, `Chest`, `Back`, `Pelvis`, `LShoulder`, `LArm`, `LHand`, `RShoulder`, `RArm`, `RHand`, `LThigh`, `LLeg`, `LFoot`, `RThigh`, `RLeg`, `RFoot`. `node_areas[i]` est l’aire duale du nœud en m².

```python
import jos3
from foampilot.postprocess import OpenFOAMJOS3Coupler

model = jos3.JOS3(ex_output="all")
coupler = OpenFOAMJOS3Coupler.from_openfoam(
    model,
    case_path="case_human",
    segment_ids=segment_ids,
    node_areas=node_areas,
    region=None,
    mode="raw_extra_heat",
)

exchange = coupler.exchange_from_openfoam(
    time_step="latest",
    h_field="h",
    air_temperature_field="Ta",
    surface_temperature_field="T",
)
model.simulate(times=1, dtime=60)

# Retour vers OpenFOAM : champ nodal pointScalarField en W/m².
coupler.write_point_scalar_field(
    exchange.body_flux,
    "case_human/1/qJOS3",
    field_name="qJOS3",
    time_name="1",
)
```

Le lecteur direct de FoamPilot accepte les champs `volScalarField` et `pointScalarField`. Pour un champ volumique OpenFOAM, le lecteur récupère les valeurs de cellules ; le mapping `segment_ids` doit alors être construit pour ces entités, ou bien le champ doit être exporté en champ point depuis OpenFOAM. Pour un échange strictement nodal, il est recommandé de produire `h`, `Ta` et `T` comme `pointScalarField`.

## Échange séquentiel et échange itératif

Le mode séquentiel consiste à lancer OpenFOAM, lire les trois champs, avancer JOS-3 d’un pas, puis écrire `qJOS3` pour le pas suivant. Ce protocole est reproductible et ne nécessite pas de modification du solveur OpenFOAM.

Le mode itératif peut réutiliser la même classe dans une boucle externe : OpenFOAM est lancé ou avancé, `exchange_from_openfoam` lit les champs, `model.simulate` avance la physiologie et `write_point_scalar_field` écrit le retour. La convergence peut être contrôlée sur la norme du flux nodal ou sur la température cutanée.

Le mode `sensible_correction` est disponible lorsque le flux sensible OpenFOAM remplace celui calculé par JOS-3. Il injecte la différence entre la puissance CFD et la perte sensible interne à JOS-3, afin de limiter le double comptage. Le mode `raw_extra_heat` est le choix par défaut et injecte directement la puissance CFD dans `ex_q`.
