from pathlib import Path
import json,time,traceback
import numpy as np
import build123d as bd

ROOT=Path(__file__).resolve().parents[2] / 'case_complex'
OUT=ROOT/'build123d_complex_benchmark'; OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'analysis_sections.json').read_text())

def resample_projected(s,n=32):
    p=np.asarray(s['phase_locked_points'],float); c=np.asarray(s['center'],float); t=np.asarray(s['tangent'],float); t/=max(np.linalg.norm(t),1e-12)
    p=p-((p-c)@t)[:,None]*t[None,:]
    if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-8:p=p[:-1]
    q=np.vstack([p,p[0]]); seg=np.linalg.norm(np.diff(q,axis=0),axis=1); cum=np.r_[0,np.cumsum(seg)]
    out=[]
    for d in np.linspace(0,cum[-1],n,endpoint=False):
        k=min(max(int(np.searchsorted(cum,d,side='right')-1),0),len(p)-1); u=(d-cum[k])/max(seg[k],1e-12); out.append(q[k]*(1-u)+q[k+1]*u)
    return np.asarray(out)

def make_wire(s,n=32):
    p=resample_projected(s,n); v=[bd.Vector(*map(float,x)) for x in p]
    return bd.Wire([bd.Edge.make_line(v[i],v[(i+1)%len(v)]) for i in range(len(v))])

report={'source':'analysis_sections.json','branches':[],'union':{},'parameters':{'points_per_section':32}}
solids={}
for ruled in (False,True):
    method='loft_ruled' if ruled else 'loft_smooth'; start=time.perf_counter(); method_rows=[]
    for b in data['branches']:
        row={'branch_id':b['branch_id'],'method':method,'n_sections':len(b['sections']),'ok':False}; t=time.perf_counter()
        try:
            wires=[make_wire(s) for s in b['sections']]
            obj=bd.Solid.make_loft(wires,ruled=ruled)
            row.update(ok=bool(obj.is_valid),volume=float(obj.volume),faces=len(obj.faces()),edges=len(obj.edges()),time_s=time.perf_counter()-t)
            if obj.is_valid:
                path=OUT/f'branch_{b["branch_id"]:02d}_{method}.step'; bd.export_step(obj,path); row['step']=str(path); solids[(method,b['branch_id'])]=obj
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc)[:500],time_s=time.perf_counter()-t)
        method_rows.append(row); print(row,flush=True)
    report['branches'].extend(method_rows); report['timings_'+method]=time.perf_counter()-start

# Pairwise intersections for each method. OCC booleans are used only for measurement here.
for method in ('loft_smooth','loft_ruled'):
    objs=[(bid,solids[(method,bid)]) for bid in sorted(b['branch_id'] for b in data['branches']) if (method,bid) in solids]
    pairs=[]; union=None
    for bid,obj in objs:
        if union is None: union=obj; continue
        t=time.perf_counter()
        try:
            inter=union & obj; vol=float(inter.volume); pairs.append({'with_branch':bid,'intersection_volume':vol,'intersection_valid':bool(inter.is_valid),'time_s':time.perf_counter()-t}); union=union.fuse(obj)
        except Exception as exc: pairs.append({'with_branch':bid,'error':type(exc).__name__+': '+str(exc)[:300],'time_s':time.perf_counter()-t})
    report['union'][method]={'n_inputs':len(objs),'pairs':pairs}
    if union is not None:
        report['union'][method]['valid']=bool(union.is_valid); report['union'][method]['volume']=float(union.volume); report['union'][method]['faces']=len(union.faces())
        try: bd.export_step(union,OUT/f'aorta_union_{method}.step')
        except Exception as exc: report['union'][method]['export_error']=str(exc)
(OUT/'benchmark_report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
print(json.dumps({'report':str(OUT/'benchmark_report.json'),'n_rows':len(report['branches']),'union':report['union']},indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
