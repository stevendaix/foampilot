from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def load(path):
    m=trimesh.load_mesh(path,process=False); m.process(validate=True); return m

def distances(a,b):
    ta=cKDTree(np.asarray(a.vertices)); tb=cKDTree(np.asarray(b.vertices))
    da=tb.query(np.asarray(a.vertices),workers=-1)[0]; db=ta.query(np.asarray(b.vertices),workers=-1)[0]
    return {'a_to_b_mean':float(np.mean(da)),'a_to_b_rms':float(np.sqrt(np.mean(da**2))),'a_to_b_p95':float(np.percentile(da,95)),'a_to_b_max':float(np.max(da)),'b_to_a_mean':float(np.mean(db)),'b_to_a_rms':float(np.sqrt(np.mean(db**2))),'b_to_a_p95':float(np.percentile(db,95)),'b_to_a_max':float(np.max(db)),'symmetric_mean':float((np.mean(da)+np.mean(db))/2),'symmetric_max':float(max(np.max(da),np.max(db)))}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('reference',type=Path); ap.add_argument('candidate',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); ref=load(args.reference); cand=load(args.candidate); r={'reference':str(args.reference),'candidate':str(args.candidate),'reference_vertices':len(ref.vertices),'candidate_vertices':len(cand.vertices),'distances_vertex_approx':distances(ref,cand),'reference_volume':float(ref.volume),'candidate_volume':float(cand.volume),'reference_area':float(ref.area),'candidate_area':float(cand.area)}; args.output.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
