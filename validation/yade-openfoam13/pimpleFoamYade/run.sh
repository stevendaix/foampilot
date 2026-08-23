#! /bin/bash
set -euo pipefail
blockMesh
cp -r 0_org 0
decomposePar
mkdir -p spheres
# 6. Create a symbolic link to Yade Install
#ln -s ${YADE_EXEC:-yade} yadeimport.py

#In yade serial:
#python3 scriptMPI.py

# YADE mpy crée le communicateur MPI du couplage ; ne pas l’imbriquer dans mpirun.
${YADE_BATCH_EXEC:-yadedaily-batch} scriptMPI.py

# yadedaily-batch peut retourner zéro malgré l’échec du job ; vérifier le journal réel.
log_file="scriptMPI.py.default.log"
if grep -qE '^status[[:space:]]*:.*FAILED|^#.*FAILED' "$log_file"; then
    echo "YADE job failed; see $log_file" >&2
    exit 1
fi
