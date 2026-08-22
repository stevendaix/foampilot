# VOF-to-DPM with OpenFOAM 13

This page documents the complete VOF-to-DPM project bundled with FoamPilot. It covers the Python converter, the native OpenFOAM 13 C++ sources, the incompressible and compressible `fvModel` bridges, the validation cases, and the PDF technical-note generator.

> The current implementation has two distinct capabilities: offline fragment extraction and solver–cloud coupling. It does not yet claim a fully automatic real-time transition unless the VOF volume is removed and the parcel is inserted transactionally.

## 1. Requirements

Use Ubuntu with a working OpenFOAM 13 installation. The commands below assume that the OpenFOAM environment is available at `/opt/openfoam13`.

```bash
sudo apt update
sudo apt install -y git build-essential python3 python3-pip
. /opt/openfoam13/etc/bashrc
foamVersion
```

Install the Python dependencies from the FoamPilot repository:

```bash
cd foampilot
sudo pip3 install -r requirements.txt
sudo pip3 install pytest
```

For a minimal converter-only environment, NumPy and pytest are sufficient for the targeted module tests. The full FoamPilot package imports additional optional geometry and post-processing modules.

## 2. Project locations

The complete VOF-to-DPM implementation is stored under `foampilot/examples/openfoam13/vof_to_dpm/`.

| Path | Contents |
|---|---|
| `src/foampilot/utilities/vof_to_dpm.py` | Python reader, connected-component extractor and OpenFOAM output writer |
| `test/test_vof_to_dpm.py` | Python unit tests |
| `examples/course_vof_to_dpm.py` | Small pedagogical Python exercise |
| `examples/generate_vof_to_dpm_technical_note.py` | PDF report generator |
| `src/foampilot/report/typst_pdf.py` | Typst report engine used by the technical note |
| `examples/openfoam13/vof_to_dpm/applications/vofToDpm` | Native offline C++ extractor |
| `examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds` | Incompressible `fvModel` bridge |
| `examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds` | Compressible `fvModel` bridge |
| `examples/openfoam13/vof_to_dpm/test/openfoam13` | OpenFOAM 13 validation cases and `Allrun` scripts |
| `docs/fr/vof_to_dpm_technical_note.pdf` | Generated technical note |

## 3. Run the Python tests

From the `foampilot` directory, run the converter tests with the import path expected by the current test module:

```bash
PYTHONPATH=src/foampilot/utilities python -m pytest -q test/test_vof_to_dpm.py
```

The tests cover disconnected and connected fragments, `alpha V` weighting, invalid indices, explicit filters, ASCII OpenFOAM field reading and output generation.

Run the synthetic lesson as follows:

```bash
PYTHONPATH=src python examples/course_vof_to_dpm.py
```

The lesson prints the number of fragments, source and converted volume, the volume residual, and the weighted momentum before and after conversion.

## 4. Compile the native OpenFOAM components

Source the OpenFOAM environment in every shell that builds or runs a case:

```bash
. /opt/openfoam13/etc/bashrc
cd foampilot/examples/openfoam13/vof_to_dpm
```

Compile the three native components separately:

```bash
wmake applications/vofToDpm
wmake applications/incompressibleVoFClouds
wmake applications/compressibleVoFClouds
```

The `Make/files` and `Make/options` files are intentionally kept with each component. The generated `Make/linux64*` objects are not versioned and are recreated by `wmake`.

The original `statisticalDPMFoam` sources are also bundled under:

```text
examples/openfoam13/vof_to_dpm/statisticalDPMFoam/
```

Build that solver family with:

```bash
cd examples/openfoam13/vof_to_dpm/statisticalDPMFoam
./Allwmake
```

## 5. Run the offline C++ extractor

The native extractor operates on a serial OpenFOAM case and reads an `alpha` field, an optional `U` field and the mesh connectivity. A typical command is:

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/vofToDpmSingleCell
. /opt/openfoam13/etc/bashrc
../../../../applications/vofToDpm/Make/linux64GccDPInt32Opt/vofToDpm \
    -alpha alpha.liquid -U U -threshold 0.5 -rhoLiquid 1000
```

The exact executable path may vary with the OpenFOAM compiler and precision options. After compilation, use `which` or list the corresponding `Make/linux64*` directory.

The extractor writes positions, fragment properties and a volume report. It uses
`V_fragment = sum(alpha_i * V_i)` without renormalising interface cells.

## 6. Run the incompressible validation

Execute the complete dam-break smoke test:

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/incompressibleVoFCloudsDamBreak
./Allrun
```

The script prepares the OpenFOAM case, enables the `fvModels` and momentum-predictor hooks, loads `incompressibleVoFClouds`, and checks the resulting cloud activity. The current validation uses a controlled manual injection to validate the solver–cloud path.

## 7. Run the compressible validation

Execute the compressible counterpart:

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/compressibleVoFCloudsDamBreak
./Allrun
```

This case validates the `compressibleVoFClouds` runtime selection, the mechanical cloud coupling and the momentum source. It is not yet an energy-conservative VOF-to-DPM transition: mass, enthalpy and pressure consistency still require a dedicated thermodynamic transfer implementation.

## 8. Generate the technical PDF

The report generator uses FoamPilot’s `ScientificDocument` and `TypstRenderer` classes:

```bash
cd foampilot
python examples/generate_vof_to_dpm_technical_note.py
```

The output is written to the repository-level `report/` directory when run from the repository root. The generated artifacts are:

```text
report/vof_to_dpm_technical_note.pdf
report/vof_to_dpm_technical_note.typ
report/vof_to_dpm.bib
```

The generator documents the theoretical transition criteria, conservation equations, implementation audit and recommended production architecture.

## 9. Current scientific scope

The Python and native offline extractors correctly compute component volume, centroid, volume-weighted velocity and equivalent spherical diameter. The C++ `fvModel` bridges evolve a `parcelCloudList` and return its mechanical source to the carrier momentum equation.

A fully automatic production transition still needs bounded `alpha` consumption, dynamic parcel insertion, stable fragment IDs, MPI component reconciliation, duplicate-conversion prevention and, for compressible cases, consistent mass and energy transfer. These limitations are documented in the French and Chinese technical materials as well.

## 10. Related documentation

| Language | Main guide | Technical materials |
|---|---|---|
| English | `docs/en/vof_to_dpm_openfoam13.md` | `docs/en/vof_to_dpm.md` |
| Français | `docs/fr/vof_to_dpm_openfoam13.md` | `docs/fr/cours_vof_to_dpm.md`, `docs/fr/audit_implementation_vof_to_dpm.md` |
| 中文 | `docs/zh/vof_to_dpm_openfoam13.md` | `docs/zh/vof_to_dpm.md` |
