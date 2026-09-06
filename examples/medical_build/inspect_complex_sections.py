import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); print('branches',len(d['branches']))
for b in d['branches']:
    secs=b.get('sections',[]); counts=[]
    for s in secs:
        x=s.get('phase_locked_points') or s.get('points'); counts.append(len(x))
    print(b['branch_id'], 'sections',len(secs),'min_points',min(counts) if counts else 0,'max_points',max(counts) if counts else 0,'total_points',sum(counts))
