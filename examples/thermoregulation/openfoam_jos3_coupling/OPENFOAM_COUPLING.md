# Mise en place du couplage OpenFOAM–JOS-3

## 1. Choix du mécanisme

OpenFOAM fournit un mécanisme natif `externalCoupled` pour transférer des champs vers une application externe et en relire d’autres pendant la simulation. Ce mécanisme est synchronisé par fichiers et verrous. Il est différent d’un simple post-traitement : OpenFOAM suspend la progression au point de couplage, écrit les champs sortants, attend le retour du programme externe, puis reprend avec les valeurs entrantes.

Pour un couplage Python sans modification C++ du solveur OpenFOAM, c’est le mécanisme recommandé. Un échange strictement en mémoire dans le même processus OpenFOAM demanderait un `functionObject` C++ personnalisé et un bridge Python/C++, car les `functionObjects` et conditions limites standards n’exécutent pas directement le code Python de FoamPilot.

## 2. Initialisation d’un calcul

Les températures initiales peuvent être préparées par FoamPilot dans le répertoire `0/`. Les champs OpenFOAM doivent utiliser les unités OpenFOAM, généralement les kelvins pour la température. Le provider FoamPilot convertit ensuite les températures Kelvin en Celsius pour JOS-3.

Pour un calcul transitoire avec valeurs initiales fournies par l’application externe, la configuration `externalCoupled` utilise `initByExternal true`. Le premier échange fournit alors les valeurs de la frontière avant la progression du calcul.

La géométrie échangée doit être connue et stable : le patch humain doit être identifié, et l’ordre des points/faces doit être conservé dans `zone_ids`, `areas` et dans les tableaux écrits par OpenFOAM. Le mapping ne doit pas être recalculé silencieusement à chaque pas.

## 3. Configuration indicative de `controlDict`

```text
functions
{
    jos3Coupling
    {
        type            externalCoupled;
        libs            (fieldFunctionObjects);
        commsDir        "${FOAM_CASE}/comms";

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

        initByExternal  true;
        waitInterval    1;
        timeOut         300;
        calcFrequency   1;
        executeControl  timeStep;
        executeInterval 1;
    }
}
```

Les noms réels des champs et du groupe de patch doivent correspondre à la condition limite utilisée par le cas. Pour une condition limite `externalCoupledMixed`, le format de ligne peut contenir plusieurs colonnes, par exemple valeur, gradient normal et paramètres mixtes. Le provider actuel lit la première colonne scalaire ; il faudra utiliser un parseur spécialisé si le cas échange le format complet de la condition mixte.

## 4. Boucle transitoire côté Python

La boucle de couplage est conceptuellement :

```python
from foampilot.postprocess import OpenFOAMExternalCoupledProvider
from foampilot.physiology import DistributedSurfaceNetwork

provider = OpenFOAMExternalCoupledProvider(
    "case/comms/region0_humanPatch",
    fields=("h", "air_temperature"),
    output_field="qJOS3",
    temperature_unit="K",
)

network = DistributedSurfaceNetwork(model, mapping)

for _ in range(n_steps):
    fields = provider.read_nodal_fields()
    exchange = network.step(
        fields["h"],
        fields["air_temperature"],
        dtime=delta_t,
    )
    provider.write_nodal_flux(exchange.environment_power)
```

Le provider attend l’apparition des fichiers `.out`, lit les valeurs, convertit les températures en Celsius, puis écrit `qJOS3.in` et recrée `OpenFOAM.lock`. La recréation du verrou signale à OpenFOAM que l’application externe a terminé son pas.

## 5. Données à échanger

| Direction | Champ | Unité | Utilisation |
|---|---|---:|---|
| OpenFOAM → FoamPilot | `h` | W/m²/K | Coefficient local de convection ou coefficient global choisi |
| OpenFOAM → FoamPilot | `air_temperature` | K dans OpenFOAM, °C dans JOS-3 | Température d’air locale |
| OpenFOAM → FoamPilot | `surface_temperature` si disponible | K | Initialisation ou contrôle de la température de surface |
| FoamPilot → OpenFOAM | `qJOS3` | W/m² | Flux local appliqué à la frontière humaine |

Le champ `h` peut être produit par le function object OpenFOAM `heatTransferCoeff`, mais il faut vérifier que la définition de `h` correspond à la loi utilisée dans FoamPilot. Si OpenFOAM fournit déjà un flux total, il ne faut pas le multiplier une seconde fois par `h`.

## 6. Steady et transitoire

En steady, deux stratégies sont possibles. La première consiste à lancer OpenFOAM jusqu’à convergence, lire les champs de frontière, exécuter JOS-3 ou le réseau distribué une fois, puis relancer OpenFOAM avec le flux obtenu. La seconde consiste à utiliser `externalCoupled` pour une itération explicite jusqu’à convergence conjointe ; dans ce cas, il faut définir un critère sur la variation du flux et de la température.

En transitoire, le `calcFrequency` doit correspondre au pas de couplage. FoamPilot doit recevoir les champs après la résolution OpenFOAM du pas `n`, avancer la physiologie sur `delta_t`, écrire le flux pour le pas `n+1`, puis laisser OpenFOAM reprendre. Une relaxation du flux peut être nécessaire :

```text
q_applied = alpha * q_new + (1 - alpha) * q_old
```

avec `0 < alpha <= 1`, surtout lorsque le pas CFD est grand ou lorsque la conductance d’ancrage de la surface est élevée.

## 7. Limite du couplage sans fichier

Le couplage `externalCoupled` est natif mais repose sur des fichiers. Le couplage Python véritablement en mémoire nécessite soit un pilote qui contrôle OpenFOAM par une interface interprocessus, soit un développement C++ dans OpenFOAM exposant les champs et le point d’exécution à Python. FoamPilot fournit déjà le contrat mémoire `CallbackFieldProvider`; le provider `OpenFOAMExternalCoupledProvider` fournit l’adaptateur natif sans modifier le solveur.

## Références

[1]: https://doc.openfoam.com/2312/tools/post-processing/function-objects/ "OpenFOAM Function objects"
[2]: https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/externalCoupled/ "OpenFOAM externalCoupled"
[3]: https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/heatTransferCoeff/ "OpenFOAM heatTransferCoeff"
[4]: https://doc.openfoam.com/2306/tools/processing/boundary-conditions/rtm/derived/thermal/externalWallHeatFluxTemperature/ "OpenFOAM externalWallHeatFluxTemperature"
