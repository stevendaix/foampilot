from pathlib import Path
import json
import numpy as np
root=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
summary=[]
for p in sorted(root.glob('branch_*.npz')):
 d=np.load(p,allow_pickle=True)
 row={'file':p.name,'points':int(len(d['points']))}
 for k in ['Blanking','GroupIds','CenterlineIds','TractIds','MaximumInscribedSphereRadius']:
  a=np.asarray(d[k]).reshape(-1)
  row[k]={'unique':np.unique(a).tolist() if k!='MaximumInscribedSphereRadius' else None,'min':float(np.min(a)),'max':float(np.max(a)),'mean':float(np.mean(a))}
 summary.append(row)
out=root/'complex_arrays_summary.json'; out.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
