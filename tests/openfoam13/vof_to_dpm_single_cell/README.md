# OpenFOAM 13 VOF-to-DPM integration case

This case validates the complete ASCII workflow rather than only the NumPy extraction algorithm. It creates a one-cell unit-cube mesh with OpenFOAM 13, writes the cell-centre field `C` and cell-volume field `Vc` using `foamPostProcess`, reads `alpha.liquid`, `U`, `C`, `Vc`, `owner` and `neighbour` with `OpenFoamCaseReader`, extracts one conservative VOF fragment, and writes DPM-oriented positions, fragment properties and a JSON audit report.

Run it from the repository root with OpenFOAM 13 available:

```sh
cd test/openfoam13/vof_to_dpm_single_cell
./Allrun
```

Expected result:

```text
PASS: OpenFOAM 13 mesh/fields -> VOF fragments -> DPM outputs
```

The case intentionally uses one full-liquid cell. Its expected liquid volume is `1`, centroid is `(0.5 0.5 0.5)`, velocity is `(2 0 0)`, and equivalent spherical diameter is `(6/pi)^(1/3)`. Generated mesh, fields and logs are ignored by the repository cleanup rules and are not part of the source fixture.
