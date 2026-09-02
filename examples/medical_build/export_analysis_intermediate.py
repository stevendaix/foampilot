from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('analysis_json',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True); t0=time.perf_counter(); data=json.loads(args.analysis_json.read_text());
    (out/'analysis_contract.json').write_text(json.dumps(data,indent=2)); branches=data['branches']; flat_points=[]; line_cells=[]; section_points=[]; section_cells=[]; section_meta=[]; offset=0; section_offset=0
    for b in branches:
        pts=np.asarray(b['points'],float); flat_points.extend(pts.tolist()); line_cells.append(list(range(offset,offset+len(pts)))); offset+=len(pts)
        for station,s in enumerate(b.get('sections',[])):
            contour=np.asarray(s.get('phase_locked_points') or s.get('points'),float); start=section_offset; section_points.extend(contour.tolist()); section_cells.append(list(range(start,start+len(contour)))+[start]); section_meta.append({'branch_id':int(b['branch_id']),'station_id':int(s.get('station_id',station)),'abscissa':float(s.get('abscissa',0.0))}); section_offset+=len(contour)
        contours=[np.asarray(s.get('phase_locked_points') or s.get('points'),float) for s in b.get('sections',[])]
        max_points=max((len(c) for c in contours),default=0); padded=np.full((len(contours),max_points,3),np.nan,float); counts=[]
        for j,c in enumerate(contours): padded[j,:len(c)]=c; counts.append(len(c))
        np.savez_compressed(out/f'branch_{int(b["branch_id"]):02d}.npz',points=pts,abscissas=np.asarray(b.get('abscissas',[]),float),tangents=np.asarray(b.get('tangents',[]),float),section_points_padded=padded,section_point_counts=np.asarray(counts,np.int32),section_centers=np.asarray([s['center'] for s in b.get('sections',[])],float))
    (out/'points.json').write_text(json.dumps({'centerline_points':flat_points,'section_points':section_points},indent=2)); (out/'lines.json').write_text(json.dumps({'centerlines':line_cells,'section_contours':section_cells},indent=2)); (out/'sections_metadata.json').write_text(json.dumps(section_meta,indent=2))
    try:
        import vtk
        def poly_export(path,points,cells,arrays=None):
            vp=vtk.vtkPoints(); [vp.InsertNextPoint(*map(float,p)) for p in points]; lines=vtk.vtkCellArray();
            for cell in cells:
                pl=vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(len(cell)); [pl.GetPointIds().SetId(i,int(x)) for i,x in enumerate(cell)]; lines.InsertNextCell(pl)
            poly=vtk.vtkPolyData(); poly.SetPoints(vp); poly.SetLines(lines); w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(poly); w.Write()
        poly_export(out/'centerlines.vtp',flat_points,line_cells); poly_export(out/'sections.vtp',section_points,section_cells); status='written'
    except ImportError: status='vtk unavailable'
    with (out/'centerlines.vtk').open('w') as f:
        f.write('# vtk DataFile Version 3.0\nmedical_build centerlines\nASCII\nDATASET POLYDATA\nPOINTS %d float\n'%len(flat_points)); [f.write('%.9g %.9g %.9g\n'%tuple(p)) for p in flat_points]; f.write('LINES %d %d\n'%(len(line_cells),sum(len(x)+1 for x in line_cells))); [f.write('%d %s\n'%(len(c),' '.join(map(str,c)))) for c in line_cells]
    manifest={'schema':'foampilot.medical_build.analysis_export.v1','branches':len(branches),'centerline_points':len(flat_points),'sections':len(section_cells),'section_points':len(section_points),'formats':{'analysis_contract.json':'written','points.json':'written','lines.json':'written','sections_metadata.json':'written','npz':'written','centerlines.vtp':status,'sections.vtp':status,'centerlines.vtk':'written'},'elapsed_seconds':round(time.perf_counter()-t0,6)}; (out/'intermediate_export_manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
