from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np

def main():
 root=Path(sys.argv[1]); report={}
 for p in sorted(root.glob('branch_*.npz')):
  z=np.load(p,allow_pickle=False); item={}
  for k in z.files:
   a=z[k]; item[k]={'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()) if np.issubdtype(a.dtype,np.number) else None,'min':float(np.nanmin(a)) if np.issubdtype(a.dtype,np.number) and a.size else None,'max':float(np.nanmax(a)) if np.issubdtype(a.dtype,np.number) and a.size else None}
  report[p.name]=item
 print(json.dumps(report,indent=2)); Path(sys.argv[2]).write_text(json.dumps(report,indent=2))
if __name__=='__main__': main()
