from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

cases = [
    ('all_groups', Path('/home/ubuntu/foampilot/openfoam_runs/meshio_openfoam_case')),
    ('body_only', Path('/home/ubuntu/foampilot/openfoam_runs/meshio_body_only_case')),
]
for label, case in cases:
    rows=list(csv.DictReader((case/'zone_mapping_openfoam.csv').open(encoding='utf-8')))
    a=np.array([float(r['area_m2']) for r in rows]); z=np.array([int(r['zone_id']) for r in rows])
    print(label, 'faces',len(rows),'area',f'{a.sum():.12f}','min_face_area',a.min(),'max_face_area',a.max())
    print('  ratios',','.join(f'{i}:{a[z==i].sum()/a.sum():.6f}' for i in sorted(set(z))))
