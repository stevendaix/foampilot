from pathlib import Path
import json, numpy as np, shutil
BASE=Path(__file__).resolve().parents[2]; ROOT=BASE/'complex_analysis_raw_package'; PATCH=ROOT/'openfoam_surface_patches'; CASE=PATCH/'case_motorbike_style'; (CASE/'system').mkdir(parents=True,exist_ok=True); (CASE/'constant'/'triSurface').mkdir(parents=True,exist_ok=True)
for src in (PATCH/'constant'/'triSurface').glob('*.stl'): shutil.copy2(src,CASE/'constant'/'triSurface'/src.name)
# Bounds from source surface, expanded for the background mesh.
import vtk
r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(BASE/'medical_build_complex_source_cap4'/'capped_surface.vtp')); r.Update(); b=r.GetOutput().GetBounds(); lo=np.array([b[0],b[2],b[4]],float); hi=np.array([b[1],b[3],b[5]],float); margin=.15*(hi-lo); lo-=margin; hi+=margin
verts=[(lo[0],lo[1],lo[2]),(hi[0],lo[1],lo[2]),(hi[0],hi[1],lo[2]),(lo[0],hi[1],lo[2]),(lo[0],lo[1],hi[2]),(hi[0],lo[1],hi[2]),(hi[0],hi[1],hi[2]),(lo[0],hi[1],hi[2])]
block='''FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n\nconvertToMeters 1;\nvertices\n(\n%s\n);\nblocks\n(\n    hex (0 1 2 3 4 5 6 7) (40 80 20) simpleGrading (1 1 1)\n);\nedges ();\nboundary\n(\n    background\n    { type patch; faces ((0 4 5 1) (1 5 6 2) (2 6 7 3) (3 7 4 0) (0 3 2 1) (4 7 6 5)); }\n);\nmergePatchPairs ();\n'''%'\n'.join('    (% .8g % .8g % .8g)'%v for v in verts)
(CASE/'system'/'blockMeshDict').write_text(block)
geom=[]
for p in sorted((CASE/'constant'/'triSurface').glob('*.stl')):
 name=p.stem; patch_name='wall' if name=='wall' else name
 geom.append(f'''    {name}\n    {{\n        type triSurfaceMesh;\n        name {name};\n    }}''')
regions=[]
for p in sorted((CASE/'constant'/'triSurface').glob('*.stl')):
 name=p.stem; regions.append(f'''        {name}\n        {{\n            level (2 2);\n            patchInfo {{ type patch; inGroups ( {name} ); }}\n        }}''')
loc=np.asarray(json.loads((ROOT/'analysis_sections.json').read_text())['branches'][0]['sections'][20]['center'])
snap=f'''FoamFile\n{{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}\ncastellatedMesh true;\nsnap true;\naddLayers false;\ngeometry\n{{\n{chr(10).join(geom)}\n}}\ncastellatedMeshControls\n{{\n    maxLocalCells 1000000;\n    maxGlobalCells 5000000;\n    minRefinementCells 10;\n    maxLoadUnbalance 0.10;\n    nCellsBetweenLevels 3;\n    features ();\n    refinementSurfaces\n    {{\n{chr(10).join(regions)}\n    }}\n    refinementRegions {{}}\n    locationInMesh (% .8g % .8g % .8g);\n    allowFreeStandingZoneFaces true;\n}}\nsnapControls\n{{ nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; }}\naddLayersControls {{}}\nmeshQualityControls {{}}\nwriteFlags (scalarLevels layerSets layerFields);\nmergeTolerance 1e-6;\n'''%tuple(loc)
(CASE/'system'/'snappyHexMeshDict').write_text(snap)
feature='''FoamFile\n{ version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }\n\n'''+''.join(f'''{p.stem}.stl\n{{\n    extractionMethod extractFromSurface;\n    includedAngle 150;\n    writeObj yes;\n}}\n''' for p in sorted((CASE/'constant'/'triSurface').glob('*.stl')))
(CASE/'system'/'surfaceFeatureExtractDict').write_text(feature)
(CASE/'README.md').write_text(f'''# Complex aorta OpenFOAM surface case\n\nThis case follows the motorBike-style workflow: `blockMesh` creates a background domain and `snappyHexMesh` uses the separated STL surfaces under `constant/triSurface`.\n\nThe source surface is partitioned into `wall`, `inlet`, and `outlet_<cap_id>` files. OpenFOAM binaries were not available in the sandbox, so run `blockMesh`, `surfaceFeatureExtract` and `snappyHexMesh` in an OpenFOAM installation, then inspect `checkMesh`.\n\nThe locationInMesh is taken from a centerline station inside the aortic lumen: {tuple(loc)}.\n''')
print({'case':str(CASE),'bounds':{'lo':lo.tolist(),'hi':hi.tolist()},'locationInMesh':loc.tolist(),'stl_files':[p.name for p in sorted((CASE/'constant'/'triSurface').glob('*.stl'))]})
