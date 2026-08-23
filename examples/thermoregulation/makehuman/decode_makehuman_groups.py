import os
from pathlib import Path
import numpy as np
p=Path(os.getenv('MAKEHUMAN_BASE_NPZ', '/usr/share/makehuman-community/data/3dobjs/base.npz'))
d=np.load(p,allow_pickle=True)
for k in ['fgstr','fgidx','group']:
    print(k, d[k].shape, d[k].dtype, repr(d[k][:30]))
raw=np.asarray(d['fgstr']).tobytes()
idx=np.asarray(d['fgidx'],dtype=int)
print('raw bytes length',len(raw),'sample',repr(raw[:500]))
print('decoded groups:')
for i,start in enumerate(idx):
    end=idx[i+1] if i+1<len(idx) else len(raw)
    chunk=raw[start:end].rstrip(b'\x00')
    print(i,start,end,repr(chunk))
