#! /bin/bash
blockMesh
cp -r 0_org 0
decomposePar
mkdir spheres
# 6. Create a symbolic link to Yade Install
#ln -s ${YADE_EXEC:-yade} yadeimport.py

#In yade serial:
#python3 scriptMPI.py

#In yade parallel
mpirun --allow-run-as-root -n 4 ${YADE_EXEC:-yade} scriptMPI.py
