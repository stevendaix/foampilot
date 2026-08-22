#! /bin/bash
blockMesh
cp -r 0_org 0
decomposePar
mkdir spheres
# 6. Create a symbolic link to Yade Install
#ln -s ${YADE_EXEC:-yade} yadeimport.py

#In yade serial:
#python3 scriptMPI.py

# YADE mpy crée le communicateur MPI du couplage ; ne pas l’imbriquer dans mpirun.
${YADE_BATCH_EXEC:-yadedaily-batch} scriptMPI.py
