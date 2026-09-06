# VOF-to-DPM with OpenFOAM 13

This page documents the complete VOF-to-DPM project bundled with FoamPilot. It covers the Python converter, the native OpenFOAM 13 C++ sources, the incompressible and compressible `fvModel` bridges, the validation cases, and the PDF technical-note generator.

> The implementation provides both offline fragment extraction and runtime solver–cloud coupling. The runtime transition is now transactional: VOF volume and energy sources are committed only after the cloud confirms effective parcel creation in `postInject()`. The qualification remains limited to the documented serial OpenFOAM 13 nominal cases.

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
| `examples/openfoam13/vof_to_dpm/example/sprayCrossFlow` | Self-contained VOF-to-DPM spray example with mass-volume post-processing |
| `examples/openfoam13/vof_to_dpm/test/openfoam13/compressibleVoFCloudsThermoDamBreak` | Compressible `thermoCloud` regression with enthalpy-source checks |
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

The script prepares the OpenFOAM case, enables the `fvModels` and momentum-predictor hooks, loads `incompressibleVoFClouds`, and checks the fragment-to-parcel conversion path. Confirmation occurs after effective parcel creation, and confirmed fragments are not injected again.

## 7. Run the compressible validation

Execute the compressible counterpart:

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/compressibleVoFCloudsDamBreak
./Allrun
```

This case validates `compressibleVoFClouds` runtime selection, mechanical coupling, alpha-rho transfer and normal solver completion. Use the dedicated thermoCloud case below to validate the thermodynamic path.

## 8. Validate the compressible thermoCloud path

The dedicated case enables a `thermoCloud`, declares the H2O liquid components required by `parcelThermo`, and checks enthalpy-source application after parcel confirmation:

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/compressibleVoFCloudsThermoDamBreak
./Allrun
```

The post-processing checks one confirmed batch, one enthalpy-source application to `e.water`, two alpha-rho source applications, normal solver completion and absence of floating-point or fatal errors. See [`vof_to_dpm_implementation_status.md`](../fr/vof_to_dpm_implementation_status.md) for the detailed implementation matrix and current limitations.

## 9. Generate the technical PDF

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

## 10. Current scientific scope

The Python and native offline extractors correctly compute component volume, centroid, volume-weighted velocity and equivalent spherical diameter. The C++ `fvModel` bridges evolve a `parcelCloudList` and return its mechanical source to the carrier momentum equation.

The nominal automatic transition now provides bounded `alpha` consumption, dynamic parcel insertion, deterministic fragment IDs, duplicate-conversion prevention and confirmed compressible mass and energy transfer. MPI reconciliation, multi-component thermodynamics and pathological geometries remain outside the current regression coverage.

## 11. Related documentation

| Language | Main guide | Technical materials |
|---|---|---|
| English | `docs/en/vof_to_dpm_openfoam13.md` | `docs/en/vof_to_dpm.md` |
| Français | `docs/fr/vof_to_dpm_openfoam13.md` | `docs/fr/vof_to_dpm_implementation_status.md`, `docs/fr/cours_vof_to_dpm.md`, `docs/fr/audit_implementation_vof_to_dpm.md` |
| 中文 | `docs/zh/vof_to_dpm_openfoam13.md` | `docs/zh/vof_to_dpm.md` |
