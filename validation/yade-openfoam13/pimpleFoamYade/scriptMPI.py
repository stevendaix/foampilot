import os
# from yadeimport import *
from yade import mpy as mp

parallelYade = os.environ.get('YADE_PARALLEL', 'true').lower() in ('1', 'true', 'yes')
numProcOF = int(os.environ.get('OPENFOAM_PROCS', '2'))

O.periodic = True
O.cell.setBox(1.005, 1.005, 1.005)

numspheres = 2000
young = 5e6
density = 1000
NSTEPS = int(os.environ.get('CFDEM_NSTEPS', '408'))

O.materials.append(FrictMat(young=young, poisson=0.5, frictionAngle=radians(15), density=density, label='spheremat'))
O.materials.append(FrictMat(young=young * 100, poisson=0.5, frictionAngle=0, density=0, label='wallmat'))

minval = 1e-08
maxval = 1.0005

v0 = Vector3(minval, minval, minval)
v1 = Vector3(maxval, minval, minval)
v2 = Vector3(maxval, maxval, minval)
v3 = Vector3(minval, maxval, minval)

v4 = Vector3(minval, minval, maxval)
v5 = Vector3(maxval, minval, maxval)
v6 = Vector3(maxval, maxval, maxval)
v7 = Vector3(minval, maxval, maxval)

mn, mx = Vector3(2e-08, 2e-08, 2e-08), Vector3(1.0 - 2e-08, 1.0 - 2e-08, 1.0 - 2e-08)
sp = pack.SpherePack()
sp.makeCloud(mn, mx, rMean=0.0075, rRelFuzz=0.0, num=numspheres)
O.bodies.append([sphere(center, rad, material='spheremat') for center, rad in sp])

# box
#xminus = 0.25*(v4+v0+v3+v7)
#O.bodies.append(box(center=xminus,extents=(minval, maxval, maxval), fixed=True))

#xplus = 0.25*(v5+v6+v2+v1)
#O.bodies.append(box(center=xplus,extents=(minval, maxval, maxval), fixed=True))

yminus = 0.25 * (v4 + v0 + v1 + v5)
O.bodies.append(box(center=yminus, extents=(maxval, minval, maxval), fixed=True))

yplus = 0.25 * (v6 + v3 + v2 + v7)
O.bodies.append(box(center=yplus, extents=(maxval, minval, maxval), fixed=True))

#zminus = 0.25*(v0+v1+v2+v3)
#O.bodies.append(box(center=zminus,extents=(maxval, maxval, minval), fixed=True))
#
#zplus = 0.25*(v4+v7+v6+v5)
#O.bodies.append(box(center=zplus,extents=(maxval, maxval, minval), fixed=True))
#

fluidCoupling = FoamCoupling()
fluidCoupling.couplingModeParallel = parallelYade
fluidCoupling.isGaussianInterp = True
#use pimpleFoamYade for gaussianInterp (only in serial mode)
sphereIDs = [b.id for b in O.bodies if type(b.shape) == Sphere]

# Integrator
# add small damping in case of stability issues.. ~ 0.1 max, also note : If gravity is needed, set it in constant/g dir.
'''The yade specific (icoFoamYade, pimpleFoamYade) OpenFOAM solver can be found in $FOAM_USER_APPBIN, (
# full path here, the scond argument, 2 is the number of FoamProcs. '''
# fluidCoupling.SetOpenFoamSolver(os.environ.get('FOAM_USER_APPBIN')+'/icoFoamYade', 2)
# it also work without path after sourcing OFoam's bashrc
fluidCoupling.SetOpenFoamSolver("pimpleFoamYade", numProcOF)


def printStep():
	print("step = ", O.iter)


O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()], label='collider', allowBiggerThanPeriod=True),
        InteractionLoop(
                [Ig2_Sphere_Sphere_ScGeom(), Ig2_Box_Sphere_ScGeom()], [Ip2_FrictMat_FrictMat_FrictPhys()], [Law2_ScGeom_FrictPhys_CundallStrack()],
                label='InteractionLoop'
        ),
        GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.7, timeStepUpdateInterval=100, parallelMode=parallelYade, label="ts"),
        fluidCoupling,  #to be called after timestepper
        NewtonIntegrator(damping=0.0, label='newton', gravity=(0, 0.0, 0)),
        # VTKRecorder(fileName='spheres/3d-vtk-', recorders=['all'], parallelMode=True, iterPeriod=1000)
]
collider.verletDist = 0.0075
mp.YADE_TIMING = False
mp.FLUID_COUPLING = True
mp.VERBOSE_OUTPUT = False
mp.USE_CPP_INTERS = True
mp.ERASE_REMOTE_MASTER = True
mp.REALLOC_FREQUENCY = 0
mp.fluidBodies = sphereIDs
#mp.commSplit = True
mp.DOMAIN_DECOMPOSITION = parallelYade
mp.mpirun(NSTEPS)
mp.mprint("RUN FINISH")
# killMPI() appelle MPI_Abort(-100) dans yadedaily ; la sortie du processus
# libère le communicateur après RUN FINISH. L’ancien comportement reste opt-in.
if os.environ.get('CFDEM_KILL_MPI', 'false').lower() in ('1', 'true', 'yes'):
    fluidCoupling.killMPI()
exit()
