from pathlib import Path
import csv
import re
import numpy as np

CASE = Path(__file__).resolve().parent
COMMS = CASE / 'comms'
SEGMENTS = ['Head','Neck','Chest','Back','Pelvis','LShoulder','LArm','LHand','RShoulder','RArm','RHand','LThigh','LLeg','LFoot','RThigh','RLeg','RFoot']

def points(path):
    text = path.read_text()
    values = []
    for row in re.findall(r'\(([^()]*)\)', text):
        tokens = row.split()
        if len(tokens) != 3:
            continue
        try:
            values.append([float(v) for v in tokens])
        except ValueError:
            continue
    return np.asarray(values, dtype=float)

def faces(path):
    out=[]
    for line in path.read_text().splitlines():
        m=re.match(r'\s*\d+\(([^()]*)\)', line)
        if m: out.append([int(v) for v in m.group(1).split()])
    return out

def classify(c):
    lo, hi = c.min(axis=0), c.max(axis=0); span=np.maximum(hi-lo,1e-12)
    vertical=int(np.argmax(span)); rem=[a for a in range(3) if a != vertical]
    lateral=rem[int(np.argmax(span[rem]))]; depth=rem[0] if rem[1] == lateral else rem[1]
    p=np.zeros_like(c); p[:,0]=(c[:,lateral]-(lo[lateral]+hi[lateral])/2)/span[lateral]; p[:,1]=(c[:,depth]-(lo[depth]+hi[depth])/2)/span[depth]; p[:,2]=(c[:,vertical]-lo[vertical])/span[vertical]
    x,y,z=p.T; result=np.full(len(c),'Pelvis',dtype=object); result[z>=.84]='Head'; result[(z>=.76)&(z<.84)]='Neck'; result[(z>=.56)&(z<.76)&(y>=0)]='Chest'; result[(z>=.56)&(z<.76)&(y<0)]='Back'
    upper=(z>=.47)&(z<.72)&(np.abs(x)>.16); lower=(z>=.27)&(z<.47)&(np.abs(x)>.10); feet=(z<.13)&(np.abs(x)>.08); hands=(z>=.28)&(z<.58)&(np.abs(x)>.36)
    result[upper&(x<0)]='LShoulder'; result[upper&(x>=0)]='RShoulder'; result[lower&(x<0)]='LThigh'; result[lower&(x>=0)]='RThigh'; result[feet&(x<0)]='LFoot'; result[feet&(x>=0)]='RFoot'; result[hands&(x<0)]='LHand'; result[hands&(x>=0)]='RHand'
    arm=(np.abs(x)>.22)&~hands&~feet&(z>=.13)&(z<.58); result[arm&(x<0)]='LArm'; result[arm&(x>=0)]='RArm'; leg=(np.abs(x)>.08)&~feet&(z<.42); result[leg&(x<0)]='LLeg'; result[leg&(x>=0)]='RLeg'
    return result, {'vertical':'xyz'[vertical],'lateral':'xyz'[lateral],'depth':'xyz'[depth]}

P=points(COMMS/'patchPoints'); F=faces(COMMS/'patchFaces'); C=np.asarray([P[f].mean(axis=0) for f in F]); labels,axes=classify(C); rows=[]
for i,(face,centroid,label) in enumerate(zip(F,C,labels)):
    v=P[face]; area=sum(0.5*np.linalg.norm(np.cross(v[j]-v[0],v[j+1]-v[0])) for j in range(1,len(v)-1)); rows.append({'face_id':i,'zone_id':SEGMENTS.index(str(label)),'jos3_name':str(label),'area_m2':f'{area:.12g}','cx_m':f'{centroid[0]:.12g}','cy_m':f'{centroid[1]:.12g}','cz_m':f'{centroid[2]:.12g}'})
out=CASE/'zone_mapping_openfoam.csv'; out.write_text('');
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print({'faces':len(F),'areas_m2':float(sum(float(r['area_m2']) for r in rows)),'axes':axes,'zone_counts':{s:int(sum(r['jos3_name']==s for r in rows)) for s in SEGMENTS},'output':str(out)})
