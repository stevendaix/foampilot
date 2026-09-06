from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import trimesh

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stl',type=Path); ap.add_argument('--output',type=Path); args=ap.parse_args()
    m=trimesh.load_mesh(args.stl, process=False)
    m.process(validate=True)
    edges=np.sort(np.asarray(m.edges),axis=1)
    _, counts=np.unique(edges, axis=0, return_counts=True)
    boundary=int(np.sum(counts==1)); nonmanifold=int(np.sum(counts>2))
    components=m.split(only_watertight=False)
    report={'file':str(args.stl),'vertices':int(len(m.vertices)),'faces':int(len(m.faces)),'components':int(len(components)),'component_faces':[int(len(x.faces)) for x in components],'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,'watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'finite':bool(np.isfinite(m.vertices).all()),'volume':float(m.volume),'area':float(m.area),'bounds':np.asarray(m.bounds).tolist()}
    text=json.dumps(report,indent=2)
    if args.output: args.output.write_text(text)
    print(text)
if __name__=='__main__': main()
