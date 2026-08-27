from __future__ import annotations
import argparse,json,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('step_dir',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); import build123d as bd
 shapes=[]; metrics=[]; t0=time.perf_counter()
 for p in sorted(list(args.step_dir.glob('branch_*.step'))+list(args.step_dir.glob('branch_*/build123d_branch_*.step'))):
  t=time.perf_counter(); shape=bd.import_step(str(p)); valid=getattr(shape,'is_valid',None); valid=valid() if callable(valid) else valid; metrics.append({'file':p.name,'load_seconds':round(time.perf_counter()-t,6),'is_valid':None if valid is None else bool(valid),'volume':float(shape.volume)}); shapes.append(shape)
 fused=shapes[0]; steps=[]
 for i,shape in enumerate(shapes[1:],1):
  t=time.perf_counter()
  try: fused=fused.fuse(shape); steps.append({'branch':i,'seconds':round(time.perf_counter()-t,6),'status':'ok','is_valid':bool(fused.is_valid),'volume':float(fused.volume)})
  except Exception as exc: steps.append({'branch':i,'seconds':round(time.perf_counter()-t,6),'status':'failed','error':repr(exc)}); break
 args.output.parent.mkdir(parents=True,exist_ok=True); bd.export_step(fused,str(args.output.with_suffix('.step'))); bd.export_stl(fused,str(args.output.with_suffix('.stl'))); report={'inputs':metrics,'fusions':steps,'total_seconds':round(time.perf_counter()-t0,6),'final_is_valid':bool(fused.is_valid),'final_volume':float(fused.volume),'final_solid_count':len(fused.solids()) if hasattr(fused,'solids') else None}; args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
