from pathlib import Path
import json
import numpy as np
ROOT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
data=json.loads((ROOT/'analysis_sections.json').read_text())
report={'schema':data.get('schema'),'all_ok':True,'branches':[]}
for b in data['branches']:
    p=np.asarray(b['points'],float); s=np.asarray(b['abscissas'],float); secs=b['sections']; row={'branch_id':b['branch_id'],'n_sections':len(secs),'ok':True,'checks':{}}
    row['checks']['branch_points_finite']=bool(np.isfinite(p).all()); row['checks']['abscissa_monotone']=bool(np.all(np.diff(s)>=-1e-10)); row['checks']['length_matches']=abs(float(s[-1])-float(b['length']))<1e-8
    frame_err=[]; center_err=[]; area_err=[]; point_err=[]; station=[]
    for sec in secs:
        t=np.asarray(sec['tangent']); n=np.asarray(sec['normal']); bi=np.asarray(sec['binormal']); c=np.asarray(sec['center']); pts=np.asarray(sec['phase_locked_points'])
        frame_err += [abs(np.linalg.norm(t)-1),abs(np.linalg.norm(n)-1),abs(np.linalg.norm(bi)-1),abs(np.dot(t,n)),abs(np.dot(t,bi)),abs(np.dot(n,bi))]
        station.append(sec['station_id']); area_err.append(float(sec['area'])); point_err.append(len(pts))
        # nearest centerline distance is a conservative geometric consistency check.
        center_err.append(float(np.min(np.linalg.norm(p-c[None,:],axis=1))))
    row['checks']['station_monotone']=bool(station==sorted(station) and len(set(station))==len(station)); row['checks']['frames_orthonormal']=max(frame_err,default=0.0)<1e-6; row['checks']['positive_area']=min(area_err,default=0.0)>0; row['checks']['point_counts_ge_3']=min(point_err,default=0)>=3; row['checks']['centerline_distance_max']=float(max(center_err,default=0.0))
    row['metrics']={'frame_error_max':max(frame_err,default=0.0),'centerline_distance_max':max(center_err,default=0.0),'area_min':min(area_err,default=0.0),'area_max':max(area_err,default=0.0),'point_count_min':min(point_err,default=0)}
    row['ok']=bool(all(bool(v) for k,v in row['checks'].items() if k!='centerline_distance_max') and row['checks']['centerline_distance_max']<1.0)
    report['all_ok'] = bool(report['all_ok'] and row['ok']); report['branches'].append(row)
(ROOT/'sections_validation.json').write_text(json.dumps(report,indent=2,default=lambda o: o.item() if hasattr(o,'item') else str(o)))
print(json.dumps({'all_ok':report['all_ok'],'branches':len(report['branches']),'section_counts':[r['n_sections'] for r in report['branches']],'max_centerline_distance':max(r['metrics']['centerline_distance_max'] for r in report['branches'])},indent=2))
