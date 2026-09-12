# JOS-3 theory and equation audit

This page documents the JOS-3 thermophysiological model and the equation audit performed against the scientific article, the official Python implementation and the FoamPilot embedded copy.

The complete rewritten theoretical documentation is available in the [French version](../../fr/theory/jos3_theory.md). It covers the 85-state implementation, heat-balance equations, boundary conditions, body-surface scaling, blood-flow control, shivering, non-shivering thermogenesis, numerical resolution and the distributed CFD surface extension.

The audit report is stored in the repository at `audit_jos3/equation_audit.md`.

## Main conclusions

The article abstract mentions 83 nodes, while the construction section and the Python implementation resolve 85 states. FoamPilot preserves the 85-state topology. The original JOS-3 model has 17 physiological skin temperatures; independent temperatures at CFD faces are added only by FoamPilot's `DistributedSurfaceNetwork`.

The embedded copy reproduces the official example in native mode and with zero external flux. The distributed CFD exchange is an extension and must be validated separately from the published JOS-3 model.

## References

- [Takahashi et al., Thermoregulation model JOS-3 with new open source code](https://doi.org/10.1016/j.enbuild.2020.110575)
- [Official JOS-3 repository](https://github.com/TanabeLab/JOS-3)
- [FoamPilot repository](https://github.com/stevendaix/foampilot)
