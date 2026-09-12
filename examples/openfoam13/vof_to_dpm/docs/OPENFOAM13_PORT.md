# OpenFOAM 13 native C++ VOF-to-DPM port

This branch now contains a native OpenFOAM 13 C++ application, `applications/vofToDpm`. It reads a VOF phase-fraction field, detects connected eligible cells using `fvMesh::cellCells()`, computes each fragment volume as `alpha[cell] * mesh.V()[cell]`, and writes one parcel position per connected fragment.

The alpha threshold is a selector only. It is not used to renormalise the liquid fraction, which preserves the physical volume represented by the VOF field. Fragment centroids and optional velocities are weighted by the same liquid volume. The application also writes an equivalent diameter and liquid mass using the configurable `rhoLiquid` value.

## Build under OpenFOAM 13

```sh
. /opt/openfoam13/etc/bashrc
cd applications/vofToDpm
wmake
```

The executable is installed in `$FOAM_USER_APPBIN/vofToDpm`. The application links against `finiteVolume` and `meshTools` and uses the OpenFOAM 13 `fvMesh` and field APIs.

## Usage

```sh
vofToDpm \
    -alpha alpha.liquid \
    -U U \
    -threshold 0.5 \
    -minCells 1 \
    -minVolume 0 \
    -rhoLiquid 1000
```

The application writes `constant/cloudPositions`, which follows the OpenFOAM 13 `vectorField` format used by `manualInjection`, `constant/vofToDpmFragments`, which stores per-fragment volume, mass, diameter, centroid and velocity, and `constant/vofToDpmReport`, which records selected, converted and discarded volumes.

## Validation

The reproducible case is:

```sh
cd test/openfoam13/vofToDpmSingleCell
./Allrun
```

It builds a unit-cube mesh with `blockMesh`, runs the compiled C++ application and checks the exact volume, centroid, mass, velocity and parcel position. The expected result is:

```text
PASS: native C++ OpenFOAM 13 VOF fragments, volume, centroid, mass and velocity
```

## parcelInBox integration

The case `test/openfoam13/vofToDpmParcelInBox` uses the OpenFOAM 13 `multicomponentFluid/parcelInBox` tutorial as the solver-side reference. Its `Allrun` script copies the tutorial to a temporary directory, adds a VOF field, runs `vofToDpm`, extracts the generated fragment mass and diameter, updates `cloudProperties`, and runs `foamRun -solver multicomponentFluid`.

This validates the complete one-way path: native C++ VOF reading, connected-fragment reduction, OpenFOAM `cloudPositions` output, `manualInjection` parsing, parcel creation and solver execution. The expected result is:

```text
PASS: VOF-derived cloudPositions injected into OpenFOAM 13 parcelInBox
```

## Current scope

The current implementation is intentionally serial and stops with a clear error when launched under MPI. It performs offline VOF-to-parcel description and does not yet remove liquid from the VOF field or insert parcels into a live solver cloud. Those operations require solver-specific source terms and a parallel fragment-reconciliation algorithm; they must be added before claiming fully coupled two-way VOF–DPM physics.
