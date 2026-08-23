# -*- coding: utf-8 -*-

###
### This file is generated automatically by SALOME v7.7.1 with dump python functionality
###

import sys
import salome

salome.salome_init()
theStudy = salome.myStudy

import salome_notebook
notebook = salome_notebook.NoteBook(theStudy)
sys.path.insert( 0, r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New(theStudy)

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
block1 = geompy.MakeBoxDXDYDZ(0.02, 0.1, 0.001)
Box_2 = geompy.MakeBoxDXDYDZ(0.01, 0.03, 0.001)
block2 = geompy.MakeTranslation(Box_2, -0.01, 0, 0)
Domain = geompy.MakeFuseList([block1, block2], True, True)
Domain_vertex_25 = geompy.GetSubShape(Domain, [25])
Plane_1 = geompy.MakePlane(Domain_vertex_25, OX, 0.15)
Plane_2 = geompy.MakePlane(Domain_vertex_25, OY, 0.15)
meshingStuff = geompy.MakePartition([Domain], [Plane_1, Plane_2], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["VERTEX"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["VERTEX"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
listSubShapeIDs = geompy.SubShapeAllIDs(meshingStuff, geompy.ShapeType["FACE"])
hotWall = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(hotWall, [74])
adiabaticWalls = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(adiabaticWalls, [50, 69, 79, 31])
constWall = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(constWall, [4, 38])
front = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(front, [62, 14, 45])
back = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(back, [82, 26, 55])
interface = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(interface, [34])
interfaceFixed = geompy.CreateGroup(meshingStuff, geompy.ShapeType["FACE"])
geompy.UnionIDs(interfaceFixed, [58])
BB = geompy.MakeBoundingBox(meshingStuff, True)
geomObj_1 = geompy.MakeCDG(BB)
Point_1 = geompy.MakeCDG(BB)
backgroundMesh = geompy.MakeScaleAlongAxes(BB, Point_1, 1.1, 1.1, 2)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( block1, 'block1' )
geompy.addToStudy( Box_2, 'Box_2' )
geompy.addToStudy( block2, 'block2' )
geompy.addToStudy( Domain, 'Domain' )
geompy.addToStudyInFather( Domain, Domain_vertex_25, 'Domain:vertex_25' )
geompy.addToStudy( Plane_1, 'Plane_1' )
geompy.addToStudy( Plane_2, 'Plane_2' )
geompy.addToStudy( meshingStuff, 'meshingStuff' )
geompy.addToStudyInFather( meshingStuff, hotWall, 'hotWall' )
geompy.addToStudyInFather( meshingStuff, adiabaticWalls, 'adiabaticWalls' )
geompy.addToStudyInFather( meshingStuff, constWall, 'constWall' )
geompy.addToStudyInFather( meshingStuff, front, 'front' )
geompy.addToStudyInFather( meshingStuff, back, 'back' )
geompy.addToStudyInFather( meshingStuff, interface, 'interface' )
geompy.addToStudyInFather( meshingStuff, interfaceFixed, 'interfaceFixed' )
geompy.addToStudy( BB, 'BB' )
geompy.addToStudy( Point_1, 'Point_1' )
geompy.addToStudy( backgroundMesh, 'backgroundMesh' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New(theStudy)
backgroundMesh_1 = smesh.Mesh(backgroundMesh)
Regular_1D = backgroundMesh_1.Segment()
Local_Length_1 = Regular_1D.LocalLength(0.001,None,1e-07)
Quadrangle_2D = backgroundMesh_1.Quadrangle(algo=smeshBuilder.QUADRANGLE)
Hexa_3D = backgroundMesh_1.Hexahedron(algo=smeshBuilder.Hexa)
isDone = backgroundMesh_1.Compute()
Local_Length_2 = smesh.CreateHypothesis('LocalLength')
Local_Length_2.SetLength( 0.0005 )
Local_Length_2.SetPrecision( 1e-07 )
hotWall_1 = smesh.Mesh(hotWall)
status = hotWall_1.AddHypothesis(Local_Length_1)
Regular_1D_1 = hotWall_1.Segment()
MEFISTO_2D = hotWall_1.Triangle(algo=smeshBuilder.MEFISTO)
isDone = hotWall_1.Compute()
adiabaticWalls_1 = smesh.Mesh(adiabaticWalls)
status = adiabaticWalls_1.AddHypothesis(Local_Length_1)
Regular_1D_2 = adiabaticWalls_1.Segment()
MEFISTO_2D_1 = adiabaticWalls_1.Triangle(algo=smeshBuilder.MEFISTO)
isDone = adiabaticWalls_1.Compute()
constWall_1 = smesh.Mesh(constWall)
status = constWall_1.AddHypothesis(Local_Length_1)
Regular_1D_3 = constWall_1.Segment()
MEFISTO_2D_2 = constWall_1.Triangle(algo=smeshBuilder.MEFISTO)
isDone = constWall_1.Compute()
front_1 = smesh.Mesh(front)
status = front_1.AddHypothesis(Local_Length_1)
Regular_1D_4 = front_1.Segment()
MEFISTO_2D_3 = front_1.Triangle(algo=smeshBuilder.MEFISTO)
isDone = front_1.Compute()
back_1 = smesh.Mesh(back)
MEFISTO_2D_4 = back_1.Triangle(algo=smeshBuilder.MEFISTO)
Regular_1D_5 = back_1.Segment()
status = back_1.AddHypothesis(Local_Length_1)
isDone = back_1.Compute()
interface_1 = smesh.Mesh(interface)
status = interface_1.AddHypothesis(Local_Length_1)
Regular_1D_6 = interface_1.Segment()
MEFISTO_2D_5 = interface_1.Triangle(algo=smeshBuilder.MEFISTO)
isDone = interface_1.Compute()
interfaceFixed_1 = smesh.Mesh(interfaceFixed)
status = interfaceFixed_1.AddHypothesis(Local_Length_1)
Regular_1D_7 = interfaceFixed_1.Segment()
MEFISTO_2D_6 = interfaceFixed_1.Triangle(algo=smeshBuilder.MEFISTO)
isDone = interfaceFixed_1.Compute()
try:
  interfaceFixed_1.ExportSTL( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/stl/interfaceFixed.stl', 1 )
except:
  print 'ExportSTL() failed. Invalid file name?'
try:
  back_1.ExportSTL( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/stl/back.stl', 1 )
except:
  print 'ExportSTL() failed. Invalid file name?'
try:
  front_1.ExportSTL( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/stl/front.stl', 1 )
except:
  print 'ExportSTL() failed. Invalid file name?'
try:
  constWall_1.ExportSTL( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/stl/constWall.stl', 1 )
except:
  print 'ExportSTL() failed. Invalid file name?'
try:
  adiabaticWalls_1.ExportSTL( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/stl/adiabaticWalls.stl', 1 )
except:
  print 'ExportSTL() failed. Invalid file name?'
try:
  hotWall_1.ExportSTL( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/stl/hotWall.stl', 1 )
except:
  print 'ExportSTL() failed. Invalid file name?'
try:
  backgroundMesh_1.ExportUNV( r'/home/shorty/OpenFOAM/shorty-2.3.1/run/Shoorek/heatACMI/cad/backgroundMesh.unv' )
except:
  print 'ExportUNV() failed. Invalid file name?'


## Set names of Mesh objects
smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
smesh.SetName(Hexa_3D.GetAlgorithm(), 'Hexa_3D')
smesh.SetName(Quadrangle_2D.GetAlgorithm(), 'Quadrangle_2D')
smesh.SetName(MEFISTO_2D.GetAlgorithm(), 'MEFISTO_2D')
smesh.SetName(Local_Length_1, 'Local Length_1')
smesh.SetName(Local_Length_2, 'Local Length_2')
smesh.SetName(interface_1.GetMesh(), 'interface')
smesh.SetName(back_1.GetMesh(), 'back')
smesh.SetName(backgroundMesh_1.GetMesh(), 'backgroundMesh')
smesh.SetName(adiabaticWalls_1.GetMesh(), 'adiabaticWalls')
smesh.SetName(hotWall_1.GetMesh(), 'hotWall')
smesh.SetName(front_1.GetMesh(), 'front')
smesh.SetName(constWall_1.GetMesh(), 'constWall')
smesh.SetName(interfaceFixed_1.GetMesh(), 'interfaceFixed')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser(1)
