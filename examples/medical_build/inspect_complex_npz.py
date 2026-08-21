from pathlib import Path
import numpy as np
root=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
for p in sorted(root.glob('branch_*.npz')):
    d=np.load(p, allow_pickle=True)
    print(p.name)
    for k in d.files:
        a=d[k]
        print(' ',k, getattr(a,'shape',None), getattr(a,'dtype',None))
