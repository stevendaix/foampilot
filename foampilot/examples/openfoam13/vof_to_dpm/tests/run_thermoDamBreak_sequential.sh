#!/usr/bin/env bash
# Load OpenFOAM before enabling strict error handling; its bashrc uses
# shell contexts that are incompatible with errtrace during trap cleanup.
. /opt/openfoam13/etc/bashrc || true
set -e -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CASE_TEMPLATE="$ROOT_DIR/test/openfoam13/compressibleVoFCloudsThermoDamBreak"
ANALYZER="$ROOT_DIR/tests/analyze_thermo_conservation.py"
WORK_DIR=${THERMO_DAMBREAK_WORK_DIR:-"${TMPDIR:-/tmp}/foampilot-thermoDamBreak-seq-$$"}
KEEP_CASE=${KEEP_CASE:-0}
END_TIME=${END_TIME:-0.003}

cleanup() {
    status=$?
    if [[ "$KEEP_CASE" == 1 || "$status" != 0 ]]; then
        printf 'CASE_DIR=%s\n' "$WORK_DIR" >&2
    else
        rm -rf "$WORK_DIR"
    fi
    exit "$status"
}
trap cleanup EXIT

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

foamMergeCase "$FOAM_TUTORIALS/compressibleVoF/damBreak"

cat >> constant/physicalProperties.water <<'EOF'
liquids
{
    H2O;
}
solids
{}
EOF

sed -i \
    's/solver[[:space:]]*compressibleVoF;/solver          compressibleVoF;\nlibs            ("libcompressibleVoFClouds.so" "liblagrangianParcel.so");/' \
    system/controlDict
sed -i "s/endTime[[:space:]]*1;/endTime         ${END_TIME};/" system/controlDict
sed -i 's/writeInterval[[:space:]]*0.05;/writeInterval   0.001;/' system/controlDict
sed -i 's/momentumPredictor[[:space:]]*no;/momentumPredictor   yes;\n    models              yes;/' system/fvSolution

cat > constant/fvModels <<'EOF'
/* OpenFOAM dictionary */
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      fvModels;
}
compressibleVoFClouds
{
    type            compressibleVoFClouds;
    clouds          (cloud);
    liquidPhase     water;
    rhoLiquid       1000;
    thermoCloud     true;
    consumeAlpha    true;
}
EOF

cat > constant/cloudProperties <<'EOF'
/* OpenFOAM dictionary */
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      cloudProperties;
}
type            thermoCloud;
solution
{
    coupled         yes;
    transient       yes;
    cellValueSourceCorrection no;
    maxCo           0.2;
    sourceTerms
    {
        schemes
        {
            rho             explicit 1;
            U               explicit 1;
            h               explicit 1;
        }
    }
    integrationSchemes
    {
        U               Euler;
        T               analytical;
    }
    interpolationSchemes
    {
        rho             cell;
        rho.water       cell;
        U               cellPoint;
        mu.water        cell;
        T.water         cell;
        Cp.water        cell;
        kappa.water     cell;
        p               cell;
    }
}
constantProperties
{
    rho0            1000;
    T0              350;
    Cp0             4187;
    constantVolume  false;
}
subModels
{
    particleForces
    {
        sphereDrag;
        gravity;
    }
    injectionModels
    {
        model1
        {
            type            vofFragmentInjection;
            alpha           alpha.water;
            rhoLiquid       1000;
            uniformParcelSize volume;
            SOI             0;
            duration        1;
            parcelsPerSecond 1;
            massTotal       1;
            sizeDistribution
            {
                type            fixedValue;
                fixedValueDistribution
                {
                    value 1e-3;
                }
            }
        }
    }
    dispersionModel none;
    patchInteractionModel standardWallInteraction;
    heatTransferModel none;
    compositionModel singlePhaseMixture;
    phaseChangeModel none;
    stochasticCollisionModel none;
    surfaceFilmModel none;
    radiation off;
    standardWallInteractionCoeffs
    {
        type rebound;
    }
    singlePhaseMixtureCoeffs
    {
        phases
        (
            liquid
            {
                H2O 1;
            }
        );
    }
}
cloudFunctions {}
EOF

cat > constant/cloudPositions <<'EOF'
/* OpenFOAM vectorField */
FoamFile
{
    format ascii;
    class vectorField;
    location "constant";
    object cloudPositions;
}
(
    (0.05 0.05 0.005)
)
EOF

blockMesh > log.blockMesh 2>&1
setFields > log.setFields 2>&1
foamRun -solver compressibleVoF > log.foamRun 2>&1
python3 "$ANALYZER" --log log.foamRun --case "$WORK_DIR" --json > conservation.json
cat conservation.json
