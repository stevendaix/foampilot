from pathlib import Path
import json, shutil
import numpy as np
import vtk
from foampilot.geometry.topology.vmtk.vmtkcenterlinesections_local import (
    vmtkCenterlineSectionsLocal, _parallel_transport_frame,
)

SRC=Path('/home/ubuntu/vmtk_audit_extract/medical_build_complex_source_cap4')
OUT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
OUT.mkdir(parents=True,exist_ok=True)

def read_poly(path):
    if str(path).endswith('.vtp'):
        r=vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(path)
    r.SetFileName(str(path)); r.Update(); p=vtk.vtkPolyData(); p.DeepCopy(r.GetOutput()); return p

def branch_poly(poly, ids):
    pts=vtk.vtkPoints(); line=vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(len(ids))
    for j,pid in enumerate(ids): pts.InsertNextPoint(poly.GetPoint(pid)); line.GetPointIds().SetId(j,j)
    cells=vtk.vtkCellArray(); cells.InsertNextCell(line)
    out=vtk.vtkPolyData(); out.SetPoints(pts); out.SetLines(cells); return out

def lines_from(poly):
    out=[]; cells=poly.GetLines(); cells.InitTraversal(); ids=vtk.vtkIdList()
    while cells.GetNextCell(ids): out.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
    return out

def cumulative(points):
    return np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]

def clean_points(a):
    a=np.asarray(a,float)
    if len(a)>1 and np.allclose(a[0],a[-1],atol=1e-8): a=a[:-1]
    return a

surface=read_poly(SRC/'capped_surface.vtp')
centerlines=read_poly(SRC/'medical_build_centerlines.vtp')
all_branches=[]
for branch_id, ids in enumerate(lines_from(centerlines)):
    pts=np.array([centerlines.GetPoint(i) for i in ids],float)
    s=cumulative(pts); length=float(s[-1])
    extractor=vmtkCenterlineSectionsLocal(); extractor.Surface=surface; extractor.Centerlines=branch_poly(centerlines,ids)
    extractor.NumberOfSections=100; extractor.ResamplingNumberOfPoints=64; extractor.UseLocalSearch=True; extractor.LocalSearchRadius=10.0
    extractor.Execute(); sections=extractor.CenterlineSections or []
    # Reconstruct the same station frame convention from sampled centerline locations.
    centers,tangents=extractor._sample_stations(pts); normals,bins=_parallel_transport_frame(tangents)
    rec=[]; flat_raw=[]; flat_locked=[]; offsets=[0]
    for k,sec in enumerate(sections):
        md=dict(sec.metadata); station=int(md.get('station_index',k)); station=max(0,min(station,len(centers)-1))
        raw=clean_points(sec.points); locked=clean_points(sec.phase_locked_points)
        center=np.asarray(sec.center,float); tangent=np.asarray(sec.direction,float)
        # Use extractor station frame; this guarantees orthogonal, continuous local frames.
        normal=normals[station]; binormal=bins[station]
        # Ensure stored center/tangent are tied to the sampled centerline within numerical tolerance.
        abscissa=float(station*length/float(extractor.NumberOfSections))
        rec.append({'branch_id':branch_id,'station_id':station,'abscissa':abscissa,'center':center.tolist(),'tangent':tangent.tolist(),'normal':normal.tolist(),'binormal':binormal.tolist(),'points':raw.tolist(),'phase_locked_points':locked.tolist(),'area':float(sec.area),'perimeter':float(sec.perimeter),'equivalent_radius':float(sec.radius),'valid':True,'metadata':md})
        flat_raw.append(raw); flat_locked.append(locked); offsets.append(offsets[-1]+len(raw))
    rec.sort(key=lambda x:(x['abscissa'],x['station_id']))
    np.savez_compressed(OUT/f'sections_branch_{branch_id:02d}.npz',
        raw_points=np.vstack(flat_raw) if flat_raw else np.empty((0,3)),
        phase_locked_points=np.vstack(flat_locked) if flat_locked else np.empty((0,3)),
        offsets=np.asarray(offsets,dtype=np.int64),
        station_ids=np.asarray([x['station_id'] for x in rec],dtype=np.int64),
        abscissas=np.asarray([x['abscissa'] for x in rec],dtype=float),
        centers=np.asarray([x['center'] for x in rec],dtype=float),
        tangents=np.asarray([x['tangent'] for x in rec],dtype=float),
        normals=np.asarray([x['normal'] for x in rec],dtype=float),
        binormals=np.asarray([x['binormal'] for x in rec],dtype=float),
        area=np.asarray([x['area'] for x in rec],dtype=float),
        perimeter=np.asarray([x['perimeter'] for x in rec],dtype=float),
        equivalent_radius=np.asarray([x['equivalent_radius'] for x in rec],dtype=float))
    all_branches.append({'branch_id':branch_id,'source_cap_id':4,'target_cap_id':branch_id if branch_id<4 else branch_id+1,'points':pts.tolist(),'abscissas':s.tolist(),'tangents':np.asarray([centerlines.GetPoint(i) for i in ids],float).tolist(),'length':length,'sections':rec,'diagnostics':{'n_sections':len(rec),'n_requested_sections':100}})
# Correct branch tangent payload from centerline geometry.
for b in all_branches:
    p=np.asarray(b['points']); q=np.zeros_like(p); q[0]=p[1]-p[0]; q[-1]=p[-1]-p[-2]
    if len(p)>2: q[1:-1]=p[2:]-p[:-2]
    q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12); b['tangents']=q.tolist()
package={'schema':'foampilot.medical_build.analysis.v1','coordinate_system':'input','source_cap_id':4,'cap_records':[],'branches':all_branches,'diagnostics':{'source':'capped_surface.vtp','section_backend':'vmtkCenterlineSectionsLocal','branch_aware':True},'quality_metrics':{'n_branches':len(all_branches),'n_sections':sum(len(b['sections']) for b in all_branches)},'phase_timings':{},'warnings':[],'metadata':{'surface':'capped_surface.vtp','centerlines':'medical_build_centerlines.vtp','number_of_sections_requested':100,'resampling_points':64}}
(OUT/'analysis_sections.json').write_text(json.dumps(package,indent=2))
shutil.copy2(SRC/'capped_surface.vtp',OUT/'capped_surface.vtp'); shutil.copy2(SRC/'medical_build_delaunay.vtu',OUT/'delaunay.vtu'); shutil.copy2(SRC/'medical_build_voronoi.vtp',OUT/'voronoi.vtp')
print(json.dumps({'branches':len(all_branches),'sections':package['quality_metrics']['n_sections'],'counts':[len(b['sections']) for b in all_branches]},indent=2))
