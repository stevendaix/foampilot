# VOF to parcelInBox integration test

This case uses the native OpenFOAM 13 `multicomponentFluid/parcelInBox` tutorial as its solver-side reference. The test copies the tutorial into a temporary run directory, adds a full-liquid `alpha.liquid` field, runs the compiled `vofToDpm` application, reads the generated fragment volume and equivalent diameter, patches the tutorial `cloudProperties`, and executes `foamRun -solver multicomponentFluid`.

The validation therefore covers more than the output file format: OpenFOAM 13 must construct the reacting parcel cloud, read the VOF-derived `cloudPositions`, create one parcel, and report the same mass introduced by the C++ converter.

Run it with:

```sh
cd test/openfoam13/vofToDpmParcelInBox
./Allrun
```

Expected result:

```text
PASS: VOF-derived cloudPositions injected into OpenFOAM 13 parcelInBox
```

The tutorial is used from `$FOAM_TUTORIALS` at runtime, so generated fields, processor directories and solver logs remain outside the source tree.
