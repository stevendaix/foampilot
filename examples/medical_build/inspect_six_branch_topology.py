from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def cluster(points,tol):
    groups=[]
    for i,p in enumerate(points):
        for g in groups:
            if np.linalg.norm(p-g['center'])<=tol:
                g['indices'].append(i); g['center']=np.mean([points[j] for j in g['indices']],axis=0); break
        else: groups.append({'center':np.asarray(p,float),'indices':[i]})
    return groups

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path',type=Path); ap.add_argument('--tol',type=float,default=1.0); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); data=json.loads(args.path.read_text()); endpoints=[]
    for b in data['branches']:
        if not b['sections']: continue
        endpoints += [{'branch_id':b['branch_id'],'side':'first','point':np.asarray(b['sections'][0]['center'],float),'tangent':np.asarray(b['sections'][0]['tangent'],float)}, {'branch_id':b['branch_id'],'side':'last','point':np.asarray(b['sections'][-1]['center'],float),'tangent':np.asarray(b['sections'][-1]['tangent'],float)}]
    groups=cluster([e['point'] for e in endpoints],args.tol); rows=[]; terminal=[]; junction=[]
    for i,g in enumerate(groups):
        members=[endpoints[j] for j in g['indices']]; sides=sorted({m['side'] for m in members}); row={'node_id':i,'center':g['center'].tolist(),'degree':len(members),'sides':sides,'members':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()} for m in members]}; rows.append(row)
        if len(members)==1 or len(sides)==1: terminal.append(i)
        else: junction.append(i)
    report={'source':str(args.path),'endpoint_tolerance':args.tol,'branch_count':len(data['branches']),'endpoint_count':len(endpoints),'node_count':len(rows),'terminal_nodes':terminal,'junction_nodes':junction,'nodes':rows}; args.out.write_text(json.dumps(report,indent=2)); print(json.dumps({'branch_count':len(data['branches']),'node_count':len(rows),'terminal_nodes':terminal,'junction_nodes':junction},indent=2))
if __name__=='__main__': main()
