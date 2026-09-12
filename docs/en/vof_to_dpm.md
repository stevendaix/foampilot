# Conservative VOF-to-DPM extraction

`VofToDpmConverter` provides a deterministic bridge between a cell-centred VOF result and a DPM parcel description. It is intentionally implemented as a Python data transformation so that it can be used after an OpenFOAM run, from a generated case workflow, or in a validation pipeline.

## Physical definition

A cell is eligible when `alpha >= alpha_threshold`. The threshold is only a selection criterion; the liquid volume is never renormalised:

```text
V_fragment = sum(alpha[cell] * cellVolume[cell])
```

The centroid and mean velocity use the same liquid-volume weights. Therefore, if no `min_volume` or `min_cells` filter removes a fragment, the sum of fragment volumes equals the selected VOF liquid volume. The equivalent diameter is calculated from the volume of a sphere:

```text
d = (6 V_fragment / pi)^(1/3)
```

## Usage

```python
from foampilot.utilities.vof_to_dpm import VofToDpmConverter

converter = VofToDpmConverter(alpha_threshold=0.5)
fragments = converter.extract(
    alpha=alpha,
    cell_centres=cell_centres,
    cell_volumes=cell_volumes,
    neighbours=cell_neighbours,
    velocity=U,
)
outputs = converter.write_openfoam_outputs(fragments, "constant")
```

For an ASCII OpenFOAM 13 case, the complete workflow can be run without manually assembling NumPy arrays:

```python
converter = VofToDpmConverter(alpha_threshold=0.5)
fragments = converter.extract_case(
    case_directory="case",
    time_directory="0",
    alpha_name="alpha.liquid",
    velocity_name="U",
)
outputs = converter.write_openfoam_outputs(fragments, "case/constant")
```

`OpenFoamCaseReader` reads `C` from the selected time directory (or from `constant/polyMesh`), `Vc` from the selected time directory (or `V` from `constant/polyMesh`), and the internal connectivity from the first `nInternalFaces` entries of `owner` paired with `neighbour`. Binary fields are rejected explicitly.


The converter writes three artifacts. The `vofToDpmCloudPositions` file is an OpenFOAM `vectorField` containing fragment centroids. The `vofToDpmCloudFragments` dictionary records volume, equivalent diameter and velocity for every fragment. The JSON report contains thresholds, counts, cell IDs and all fragment properties for automated auditing.

## Important limitations

The converter expects the mesh connectivity and fields to have already been read. It does not itself read OpenFOAM binary fields, alter the VOF field, or insert parcels into a running cloud. A production real-time coupling still needs a solver-side adapter that removes the converted liquid volume, transfers momentum consistently, prevents repeated conversion, and resolves fragments spanning processor boundaries. Filters must be used deliberately because rejected fragments represent discarded liquid.

## Validation

The test suite covers separated cells, connected fragments, liquid-volume weighting, velocity weighting, OpenFOAM output generation, invalid input rejection and explicit fragment filters:

```sh
PYTHONPATH=src/foampilot/utilities python -m pytest -q test/test_vof_to_dpm.py
```

With OpenFOAM 13 installed, the end-to-end case is:

```sh
cd test/openfoam13/vof_to_dpm_single_cell
./Allrun
```

It builds a real OpenFOAM mesh, writes `C` and `Vc` with `foamPostProcess`, reads the generated files, and checks the final DPM outputs.
