from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[2] / 'case_complex'
inv=json.loads((root/'raw_inventory.json').read_text())
results=[]
for path in sorted(root.glob('branch_*.npz')):
    d=np.load(path); row={'file':path.name,'ok':True,'arrays':{}}
    points=d['points']; row['points_shape']=list(points.shape); row['finite_points']=bool(np.isfinite(points).all())
    for name in d.files:
        a=np.asarray(d[name]); stats={'shape':list(a.shape),'finite':bool(np.isfinite(a).all())}
        if name in ('TracePCoords','EdgePCoordArray'):
            stats['min']=float(np.min(a)); stats['max']=float(np.max(a)); stats['in_01']=bool(np.min(a)>=-1e-8 and np.max(a)<=1+1e-8)
        row['arrays'][name]=stats
        row['ok'] &= stats['finite'] and stats.get('in_01',True)
    row['ok'] &= row['finite_points'] and len(points)>=2
    results.append(row)
out={'branches':len(results),'all_ok':all(r['ok'] for r in results),'results':results,'inventory':inv}
(root/'raw_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'branches':len(results),'all_ok':out['all_ok'],'point_counts':[r['points_shape'][0] for r in results]},indent=2))
