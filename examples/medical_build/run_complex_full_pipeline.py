from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from foampilot.geometry.medical_build.reconstruction import Build123dReconstruction, normalize_sections
from foampilot.geometry.medical_build.models import ReconstructionSpec

def timed(report, name, fn):
    t=time.perf_counter(); value=fn(); report[name]={"seconds":round(time.perf_counter()-t,6),"status":"ok"}; return value

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("sections",type=Path); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--points",type=int,default=32); ap.add_argument("--branch",type=int,default=None); ap.add_argument("--cad-only",action="store_true"); ap.add_argument("--manual-only",action="store_true"); ap.add_argument("--skip-cad-stl",action="store_true"); args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True); report={"input":str(args.sections),"output":str(out),"steps":{}}
    data=timed(report,"load_sections",lambda: json.loads(args.sections.read_text()))
    branches=[b for b in data["branches"] if args.branch is None or int(b["branch_id"])==args.branch]; report["branches"]=len(branches); report["sections_total"]=sum(len(b.get("sections",[])) for b in branches)
    def stl():
        from section_stl_reconstruction import reconstruct_branch,write_binary_stl,quality
        result=[]
        for b in branches:
            v,t=reconstruct_branch(b["sections"],args.points); path=out/f"manual_branch_{int(b['branch_id']):02d}.stl"; write_binary_stl(v,t,path); result.append({"branch_id":b["branch_id"],"path":str(path),**quality(v,t)})
        (out/"manual_stl_branch_metrics.json").write_text(json.dumps(result,indent=2)); return result
    manual=[] if args.cad_only else timed(report,"manual_stl_by_branch",stl)
    report["manual_stl_quality"]={"branches":len(manual),"boundary_edges":sum(x["boundary_edges"] for x in manual),"nonmanifold_edges":sum(x["nonmanifold_edges"] for x in manual)}
    def cad():
        import build123d as bd
        recon=Build123dReconstruction(); result=[]; solids=[]
        for b in branches:
            sections=[]
            for s in b["sections"]:
                pts=np.asarray(s.get("phase_locked_points") or s.get("points"),float); tangent=np.asarray(s.get("tangent",s.get("direction",[0,0,1])),float)
                sections.append({"branch_id":b["branch_id"],"center":s["center"],"points":pts,"phase_locked_points":pts,"tangent":tangent,"equivalent_radius":s.get("equivalent_radius",0.0)})
            shape=recon.build(sections,ReconstructionSpec(metadata={"project_profiles":True,"max_section_points":args.points}))
            valid=getattr(shape,"is_valid",None); valid=valid() if callable(valid) else valid; step=out/f"build123d_branch_{int(b['branch_id']):02d}.step"; stl=out/f"build123d_branch_{int(b['branch_id']):02d}.stl"; bd.export_step(shape,str(step)); stl_value=None if args.skip_cad_stl else str(stl); (None if args.skip_cad_stl else bd.export_stl(shape,str(stl))); result.append({"branch_id":b["branch_id"],"step":str(step),"stl":stl_value,"is_valid":None if valid is None else bool(valid),"volume":float(shape.volume)}); solids.append(shape)
        if len(solids)>1:
            compound=bd.Compound(solids); bd.export_step(compound,str(out/"build123d_all_branches_compound.step")); bd.export_stl(compound,str(out/"build123d_all_branches_compound.stl"))
        (out/"build123d_branch_metrics.json").write_text(json.dumps(result,indent=2)); return result
    try:
        cad_metrics=timed(report,"build123d_by_branch",cad) if not args.manual_only else []
        report["build123d_quality"]={"branches":len(cad_metrics),"invalid":sum(not x["is_valid"] for x in cad_metrics),"negative_volume":sum(x["volume"]<0 for x in cad_metrics)}
    except Exception as exc:
        report["steps"]["build123d_by_branch"]={"seconds":None,"status":"failed","error":repr(exc)}
        report["build123d_quality"]={"branches":0,"invalid":None,"negative_volume":None,"error":repr(exc)}
    report["total_seconds"]=round(sum(float(v.get("seconds") or 0) for v in report.values() if isinstance(v,dict) and "seconds" in v),6); (out/"complex_full_pipeline_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
