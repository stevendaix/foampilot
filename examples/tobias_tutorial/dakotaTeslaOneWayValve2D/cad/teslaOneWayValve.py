# -*- coding: utf-8 -*- ###
### This file is generated automatically by SALOME v7.7.1 with dump python functionality
###

import sys
import salome

salome.salome_init()
theStudy = salome.myStudy

import os
path = os.getcwd()

import salome_notebook
notebook = salome_notebook.NoteBook(theStudy)
sys.path.insert( 0, r'{0}'.format(path))

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
geometryFile='{0}/cad/TeslaOneWayValve.STEP'.format(path)
TelsaValveByHolzmann = geompy.ImportSTEP(geometryFile)
Background = geompy.MakeBoundingBox(TelsaValveByHolzmann, True)
geomObj_1 = geompy.MakeCDG(TelsaValveByHolzmann)
CenterPoint = geompy.MakeCDG(TelsaValveByHolzmann)
ScaleBG = geompy.MakeScaleAlongAxes(Background, CenterPoint, 1.2, 1.2, 1)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( TelsaValveByHolzmann, 'TelsaValveByHolzmann' )
geompy.addToStudy( Background, 'Background' )
geompy.addToStudy( CenterPoint, 'CenterPoint' )
geompy.addToStudy( ScaleBG, 'ScaleBG' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New(theStudy)
backgroundMesh = smesh.Mesh(ScaleBG)
Regular_1D = backgroundMesh.Segment()
Local_Length_1 = Regular_1D.LocalLength(0.001,None,1e-07)
Quadrangle_2D = backgroundMesh.Quadrangle(algo=smeshBuilder.QUADRANGLE)
Hexa_3D = backgroundMesh.Hexahedron(algo=smeshBuilder.Hexa)


## Set names of Mesh objects
smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
smesh.SetName(Hexa_3D.GetAlgorithm(), 'Hexa_3D')
smesh.SetName(Quadrangle_2D.GetAlgorithm(), 'Quadrangle_2D')
smesh.SetName(Local_Length_1, 'Local Length_1')
smesh.SetName(backgroundMesh.GetMesh(), 'backgroundMesh')

backgroundMesh.Compute()

## Export mesh
#try:
#  backgroundMesh.ExportUNV( r'{0}/cad/backgroundMesh.unv'.format(path) )
#except:
#  print 'ExportUNV() failed. Invalid file name?'

if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser(1)

#print 'Script end'
