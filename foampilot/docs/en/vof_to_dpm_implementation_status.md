# VOF-to-DPM implementation status — OpenFOAM 13

**Repository:** foampilot  
**Reference branch:** `feat/vof-to-dpm-conservative-transition`  
**Pull Request:** [#24](https://github.com/stevendaix/foampilot/pull/24)  
**Validated environment:** OpenFOAM Foundation 13, C++14, Ubuntu 24.04  
**Last update:** 25 August 2026

## Executive status

The VOF-to-DPM port is **functional and validated on the documented serial OpenFOAM 13 nominal cases**. It covers offline fragment extraction, native connected-component detection, dynamic parcel creation, incompressible and compressible mechanical coupling, and the thermodynamic `thermoCloud` path with confirmed enthalpy transfer.

The runtime conversion is transactional. A fragment is prepared first, parcel creation is confirmed by `postInject()`, and only then are VOF consumption and Eulerian source terms committed.

> **Overall status:** ready for review and merge for the documented OpenFOAM 13 nominal scope; MPI, multi-component thermodynamics and pathological geometries remain outside the regression coverage.

## Feature matrix

| Feature | Status | Evidence |
|---|---:|---|
| ASCII OpenFOAM field reader | **Complete** | Python tests |
| Connected VOF component extraction | **Complete** | Python and OpenFOAM cases |
| `sum(alpha × V)` volume | **Complete** | Spray balance, relative error `0.0` |
| Centroid, velocity and equivalent diameter | **Complete** | Python tests |
| Native runtime fragment detection | **Complete** | `vofFragmentInjection` |
| Deterministic FNV-1a IDs | **Complete** | Multi-step cases |
| Duplicate prevention by IDs and cells | **Complete** | Multi-step spray |
| Effective parcel creation confirmation | **Complete** | `postInject()`, fixed `nParticle` mode |
| Post-confirmation VOF consumption | **Complete** | Incompressible/compressible cases |
| Incompressible coupling | **Complete** | Dam-break regression |
| Compressible alpha-rho coupling | **Complete** | Compressible dam-break |
| Thermodynamic `thermoCloud` parcels | **Complete** | Dedicated thermoCloud case |
| Confirmed enthalpy transfer | **Complete** | One `e.water` source application per batch |
| MPI reconciliation | **Not covered** | Dedicated regression still required |
| Multi-component thermodynamics | **Not covered** | Current case uses liquid H2O |
| Strongly non-orthogonal or changing topology meshes | **Not covered** | Geometry hardening remains possible |

## Reproducible validation

```bash
cd foampilot
PYTHONPATH=src/foampilot/utilities python -m pytest -q test/test_vof_to_dpm.py
. /opt/openfoam13/etc/bashrc
cd examples/openfoam13/vof_to_dpm
wmake applications/vofToDpm
wmake applications/incompressibleVoFClouds
wmake applications/compressibleVoFClouds
```

The OpenFOAM cases are executed through their `Allrun` scripts. The dedicated thermoCloud case is:

```bash
cd test/openfoam13/compressibleVoFCloudsThermoDamBreak
./Allrun
```

It checks thermoCloud initialisation, H2O liquid-component registration, one confirmed parcel batch, two alpha-rho source applications, one enthalpy-source application to `e.water`, normal solver completion and the absence of fatal or floating-point errors.

## Reference results

| Test | Result |
|---|---:|
| Python converter tests | `8 passed` |
| OpenFOAM 13 C++ libraries | Successful compilation |
| Incompressible and compressible nominal cases | Normal completion |
| Spray example | `5` final parcels, mass-volume error `0.0` |
| Dedicated thermoCloud case | One confirmed batch of `0.646099 kg` |
| ThermoCloud enthalpy source | One confirmed application |
| Git working tree after cleanup | Clean |

## Known limits

The current transaction protects the nominal serial path but does not implement global MPI component reconciliation. A parallel extension must reconcile components crossing decomposition boundaries and guarantee one global commit.

The centroid lookup through `findCellAtPosition` assumes a valid local cell. Boundary-adjacent fragments, highly non-orthogonal meshes and topology changes need explicit rejection and diagnostics before production use in those configurations.

The thermoCloud regression uses a single H2O liquid composition. A multi-component case with an independent enthalpy balance is required before extending the qualification to reactive or complex liquid mixtures.

## References

- [OpenFOAM Foundation 13](https://openfoam.org/version/13/)
- [OpenFOAM Lagrangian documentation](https://doc.cfd.direct/openfoam/lagrangian/)
- [OpenFOAM 13 VOF-to-DPM guide](vof_to_dpm_openfoam13.md)
- [Technical audit](../fr/vof_to_dpm_code_audit_openfoam13.md)
- [Pull Request #24](https://github.com/stevendaix/foampilot/pull/24)
