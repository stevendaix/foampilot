from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('output',type=Path); args=ap.parse_args(); src=json.loads(args.input.read_text()); branches=[]
    for i,b in enumerate(src['branches']):
        sections=b['sections']; points=[s['center'] for s in sections]
        branches.append({'branch_id':int(b['branch_id']),'source_cap_id':int(1000+i*2),'target_cap_id':int(1001+i*2),'points':points,'sections':sections})
    out={'schema':'foampilot.medical_build.analysis.v1','coordinate_system':'input','source_cap_id':1000,'cap_records':[],'branches':branches,'diagnostics':{'source':'python_dense_sections'},'quality_metrics':{'n_branches':len(branches),'n_sections':sum(len(b['sections']) for b in branches)}}
    args.output.write_text(json.dumps(out,indent=2)); print({'branches':len(branches),'sections':out['quality_metrics']['n_sections']})
if __name__=='__main__': main()
