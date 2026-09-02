"""Manual STL reconstruction from ordered section points.

The algorithm is deliberately independent of OCC: every pair of consecutive
closed contours is resampled to a common number of points and connected with
planar triangles. End sections are capped with a fan. The output is then
checked for duplicate vertices, boundary edges and non-manifold edges.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def clean_loop(points):
    p=np.asarray(points,dtype=float)
    if len(p)>1 and np.linalg.norm(p[0]-p[-1])<1e-8: p=p[:-1]
    keep=[p[0]]
    for x in p[1:]:
        if np.linalg.norm(x-keep[-1])>1e-8: keep.append(x)
    return np.asarray(keep)


def resample_loop(points, n):
    p=clean_loop(points); q=np.vstack([p,p[0]]); lengths=np.linalg.norm(np.diff(q,axis=0),axis=1); cumulative=np.r_[0,np.cumsum(lengths)]
    if cumulative[-1]<=1e-12: raise ValueError("degenerate section contour")
    result=[]
    for d in np.linspace(0,cumulative[-1],n,endpoint=False):
        k=min(max(int(np.searchsorted(cumulative,d,side="right")-1),0),len(p)-1); u=(d-cumulative[k])/max(lengths[k],1e-12); result.append(q[k]*(1-u)+q[k+1]*u)
    return np.asarray(result)


def orient_next(prev, current):
    n=len(prev); costs=[np.mean(np.linalg.norm(prev-np.roll(current,k,axis=0),axis=0)) for k in range(n)]
    current=np.roll(current,int(np.argmin(costs)),axis=0)
    if np.mean(np.linalg.norm(prev-current[::-1],axis=1)) < np.mean(np.linalg.norm(prev-current,axis=1)): current=current[::-1]
    return current


def reconstruct_branch(sections, n_points=32):
    contours=[]
    for section in sections:
        raw=section.get("phase_locked_points") or section.get("points")
        c=resample_loop(raw,n_points)
        if contours: c=orient_next(contours[-1],c)
        contours.append(c)
    vertices=[]; triangles=[]
    for c in contours: vertices.extend(c.tolist())
    for s in range(len(contours)-1):
        a=s*n_points; b=(s+1)*n_points
        for k in range(n_points):
            j=(k+1)%n_points
            triangles.extend([(a+k,a+j,b+j),(a+k,b+j,b+k)])
    # cap each terminal with a fan, preserving the contour vertices.
    for s,reverse in ((0,True),(len(contours)-1,False)):
        c=np.mean(contours[s],axis=0); center=len(vertices); vertices.append(c.tolist()); base=s*n_points
        for k in range(n_points):
            j=(k+1)%n_points; triangles.append((center,base+j,base+k) if reverse else (center,base+k,base+j))
    return np.asarray(vertices,float),np.asarray(triangles,np.int64)


def write_binary_stl(vertices, triangles, path):
    import struct
    with Path(path).open("wb") as f:
        f.write(b"medical_build manual section STL".ljust(80,b" ")); f.write(struct.pack("<I",len(triangles)))
        for tri in triangles:
            a,b,c=vertices[tri]; normal=np.cross(b-a,c-a); normal=normal/max(np.linalg.norm(normal),1e-12)
            f.write(struct.pack("<3f",*normal)); f.write(struct.pack("<9f",*(list(a)+list(b)+list(c)))); f.write(struct.pack("<H",0))


def quality(vertices, triangles):
    edges={}
    for a,b,c in triangles:
        for x,y in ((a,b),(b,c),(c,a)): edges[tuple(sorted((int(x),int(y))))]=edges.get(tuple(sorted((int(x),int(y)))),0)+1
    return {"vertices":int(len(vertices)),"triangles":int(len(triangles)),"boundary_edges":int(sum(v==1 for v in edges.values())),"nonmanifold_edges":int(sum(v>2 for v in edges.values())),"edge_histogram":{str(k):int(list(edges.values()).count(k)) for k in sorted(set(edges.values()))}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("contract",type=Path); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--points",type=int,default=32); args=ap.parse_args(); data=json.loads(args.contract.read_text()); args.output.mkdir(parents=True,exist_ok=True); report={"branches":[]}
    all_v=[]; all_t=[]
    for branch in data["branches"]:
        v,t=reconstruct_branch(branch["sections"],args.points); stl=args.output/f"branch_{int(branch['branch_id']):02d}.stl"; write_binary_stl(v,t,stl); report["branches"].append({"branch_id":branch["branch_id"],"stl":str(stl),**quality(v,t)}); all_t.append(t+sum(len(x) for x in all_v)); all_v.append(v)
    vertices=np.vstack(all_v); triangles=np.vstack(all_t); combined=args.output/"aorta_manual_sections.stl"; write_binary_stl(vertices,triangles,combined); report["combined"]={"stl":str(combined),**quality(vertices,triangles)}; (args.output/"manual_stl_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
