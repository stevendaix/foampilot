# JOS-3–OpenFOAM coupling

This page documents the coupling between the JOS-3 thermoregulation model and OpenFOAM through FoamPilot. The French version contains the complete implementation guide, including the distributed surface model, the 17-zone mapping, units, transient protocol and validation procedures.

See the [French documentation](../../fr/example/jos3_openfoam_coupling.md) for the detailed guide.

## Main components

```python
from foampilot.physiology import DistributedSurfaceNetwork, JOS3, SurfaceMapping
from foampilot.postprocess import OpenFOAMExternalCoupledProvider
```

`DistributedSurfaceNetwork` provides an independent surface temperature for every CFD face or point. The 17 JOS-3 zones are assigned explicitly through `zone_mapping.csv`. The standard OpenFOAM runtime protocol is `externalCoupled`, using `h.out`, `air_temperature.out`, `qJOS3.in` and `OpenFOAM.lock`.

The example case is located at:

```text
examples/thermoregulation/openfoam_jos3_coupling/openfoam_case/
```

## References

- [JOS-3 repository](https://github.com/TanabeLab/JOS-3)
- [FoamPilot repository](https://github.com/stevendaix/foampilot)
- [OpenFOAM externalCoupled](https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/externalCoupled/)
