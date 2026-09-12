#!/bin/sh
. /opt/openfoam13/etc/bashrc
set -e

rootDir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workDir=${TMPDIR:-/tmp}/foampilot-compressibleVoFCloudsThermoDamBreak-mpi-$$
np=${NP:-2}
trap 'status=$?; if [ "$status" -eq 0 ] && [ "${KEEP_CASE:-0}" -ne 1 ]; then rm -rf "$workDir"; else echo "CASE_DIR=$workDir" >&2; fi; exit "$status"' EXIT

mkdir -p "$workDir"
cd "$workDir"
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
    's/solver          compressibleVoF;/solver          compressibleVoF;\nlibs            ("libcompressibleVoFClouds.so" "liblagrangianParcel.so");/' \
    system/controlDict
sed -i 's/endTime         1;/endTime         0.003;/' system/controlDict
sed -i 's/writeInterval   0.05;/writeInterval   0.001;/' system/controlDict
sed -i 's/momentumPredictor[[:space:]]*no;/momentumPredictor   yes;\n    models              yes;/' system/fvSolution

cat > constant/fvModels <<'EOF'
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
cloudFunctions
{}
EOF

cat > constant/cloudPositions <<'EOF'
FoamFile
{
    format      ascii;
    class       vectorField;
    location    "constant";
    object      cloudPositions;
}
(
(0.05 0.05 0.005)
)
EOF

cat > system/decomposeParDict <<EOF
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}
numberOfSubdomains $np;
method scotch;
EOF

blockMesh > log.blockMesh 2>&1
setFields > log.setFields 2>&1
decomposePar -force > log.decomposePar 2>&1
mpirun --oversubscribe -np "$np" foamRun -parallel -solver compressibleVoF > log.foamRun.mpi 2>&1
python3 "$rootDir/tests/analyze_thermo_conservation.py" \
    --log log.foamRun.mpi \
    --case "$workDir" \
    --json > conservation.json
cat conservation.json

grep -q '^End$' log.foamRun.mpi
! grep -q -E 'FOAM FATAL|Floating point exception|MPI_ERR|deadlock' log.foamRun.mpi

echo "PASS: compressibleVoF + thermoCloud + MPI + conservation audit"

echo "CASE_DIR=$workDir"
