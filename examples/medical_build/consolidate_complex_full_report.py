from __future__ import annotations
import json,sys
from pathlib import Path

def main():
    root=Path(sys.argv[1]); manual=Path(sys.argv[2]); out=Path(sys.argv[3]); result={"branches":[],"totals":{}}
    for i in range(8):
        cad=json.loads((root/f"branch_{i}/build123d_branch_metrics.json").read_text())[0]
        crep=json.loads((root/f"branch_{i}/complex_full_pipeline_report.json").read_text())
        mrep=json.loads((manual/f"branch_{i}/complex_full_pipeline_report.json").read_text())
        mfile=manual/f"branch_{i}/manual_branch_{i:02d}.stl"
        result["branches"].append({"branch_id":i,"section_count":crep["sections_total"],"manual_stl":{"path":str(mfile),"bytes":mfile.stat().st_size,"seconds":mrep.get("manual_stl_by_branch",{}).get("seconds"),"boundary_edges":mrep.get("manual_stl_quality",{}).get("boundary_edges"),"nonmanifold_edges":mrep.get("manual_stl_quality",{}).get("nonmanifold_edges")},"build123d":{"step":cad["step"],"step_bytes":Path(cad["step"]).stat().st_size,"stl":cad.get("stl"),"is_valid":cad["is_valid"],"volume":cad["volume"],"seconds":crep.get("build123d_by_branch",{}).get("seconds")}})
    result["totals"]={"branches":8,"sections":sum(x["section_count"] for x in result["branches"]),"manual_stl_seconds":sum(x["manual_stl"]["seconds"] for x in result["branches"]),"build123d_seconds":sum(x["build123d"]["seconds"] for x in result["branches"]),"all_occ_valid":all(x["build123d"]["is_valid"] for x in result["branches"]),"manual_boundary_edges":sum(x["manual_stl"]["boundary_edges"] for x in result["branches"]),"manual_nonmanifold_edges":sum(x["manual_stl"]["nonmanifold_edges"] for x in result["branches"]),"negative_volumes":[x["branch_id"] for x in result["branches"] if x["build123d"]["volume"]<0]}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
