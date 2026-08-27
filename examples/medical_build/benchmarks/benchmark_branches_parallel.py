from pathlib import Path
import json,time,os,traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT=Path(__file__).resolve().parents[2] / 'case_complex'
OUT=ROOT/'build123d_branch_parallel'; OUT.mkdir(exist_ok=True)

def resample_projected(s,n=32):
    p=np.asarray(s['phase_locked_points'],float); c=np.asarray(s['center'],float); t=np.asarray(s['tangent'],float); t/=max(np.linalg.norm(t),1e-12)
    p=p-((p-c)@t)[:,None]*t[None,:]
    if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-8:p=p[:-1]
    q=np.vstack([p,p[0]]); seg=np.linalg.norm(np.diff(q,axis=0),axis=1); cum=np.r_[0,np.cumsum(seg)]; out=[]
    for d in np.linspace(0,cum[-1],n,endpoint=False):
        k=min(max(int(np.searchsorted(cum,d,side='right')-1),0),len(p)-1); u=(d-cum[k])/max(seg[k],1e-12); out.append(q[k]*(1-u)+q[k+1]*u)
    return np.asarray(out)

def worker(args):
    branch,ruled,n=args; start=time.perf_counter(); bid=int(branch['branch_id']); method='ruled' if ruled else 'smooth'; result={'branch_id':bid,'method':method,'pid':os.getpid(),'ok':False,'n_sections':len(branch['sections'])}
    try:
        import build123d as bd
        wires=[]
        for s in branch['sections']:
            p=resample_projected(s,n); v=[bd.Vector(*map(float,x)) for x in p]
            wires.append(bd.Wire([bd.Edge.make_line(v[i],v[(i+1)%len(v)]) for i in range(len(v))]))
        t=time.perf_counter(); obj=bd.Solid.make_loft(wires,ruled=ruled); result['build_time_s']=time.perf_counter()-t
        result.update(ok=bool(obj.is_valid),volume=float(obj.volume),abs_volume=abs(float(obj.volume)),faces=len(obj.faces()),edges=len(obj.edges()))
        path=OUT/f'branch_{bid:02d}_loft_{method}.step'; bd.export_step(obj,path); result['step']=str(path)
    except Exception as e:
        result.update(error=type(e).__name__+': '+str(e)[:600],traceback=traceback.format_exc(limit=3))
    result['wall_time_s']=time.perf_counter()-start
    return result

def main():
    data=json.loads((ROOT/'analysis_sections.json').read_text()); tasks=[(b,r,32) for r in (False,True) for b in data['branches']]; start=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=min(4,len(tasks))) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for f in as_completed(futures):
            row=f.result(); rows.append(row); print(row,flush=True)
    rows.sort(key=lambda x:(x['method'],x['branch_id']))
    report={'source':'analysis_sections.json','parallel':True,'workers':4,'points_per_section':32,'wall_time_s':time.perf_counter()-start,'branches':rows,'summary':{}}
    for method in ('smooth','ruled'):
        rs=[r for r in rows if r['method']==method]; report['summary'][method]={'n':len(rs),'valid':sum(bool(r['ok']) for r in rs),'negative_volume':[r['branch_id'] for r in rs if r.get('volume',0)<0],'total_build_time_s':sum(r.get('build_time_s',0) for r in rs)}
    (OUT/'parallel_benchmark_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))
if __name__=='__main__': main()
