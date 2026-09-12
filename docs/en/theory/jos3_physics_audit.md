# JOS-3 physical audit

The complete physical audit is available in the [French version](../../fr/theory/jos3_physics_audit.md). It checks energy conservation, signs, units, conduction, blood perfusion, convection, radiation, evaporation and the distributed FoamPilot surface extension.

The audit confirms that the original JOS-3 internal operator conserves pairwise exchanges and that the 85-state implementation uses positive thermal capacities. FoamPilot's distributed surface network conserves the JOS-3 skin capacity per body zone and produces independent surface temperatures at CFD points.

Two important coupling corrections are included: radiative temperature is now distinct from air temperature, and the flux written back to OpenFOAM is converted from nodal power [W] to surface flux [W/m²].

The main remaining physical limitation is latent heat: the distributed surface network currently handles local sensible and radiative exchange, while evaporation remains represented at the 17-zone JOS-3 level. A complete heat–moisture coupling must add local humidity and latent heat, while avoiding double counting of sweating.

## References

- [Takahashi et al., Thermoregulation model JOS-3 with new open source code](https://doi.org/10.1016/j.enbuild.2020.110575)
- [Official JOS-3 repository](https://github.com/TanabeLab/JOS-3)
- [OpenFOAM externalWallHeatFluxTemperature](https://doc.openfoam.com/2306/tools/processing/boundary-conditions/rtm/derived/thermal/externalWallHeatFluxTemperature/)
