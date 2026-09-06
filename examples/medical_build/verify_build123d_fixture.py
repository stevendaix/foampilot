from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from foampilot.geometry.medical_build.models import ReconstructionSpec
from foampilot.geometry.medical_build.reconstruction import Build123dReconstruction

def main():
    d=json.loads(Path(__file__).with_name("minimal_analysis_contract.json").read_text())
    sections=[]
    for s in d["branches"][0]["sections"]:
        p=np.asarray(s["phase_locked_points"],float)
        sections.append({"branch_id":0,"center":np.asarray(s["center"],float),"points":p,"phase_locked_points":p,"tangent":np.array([0.,0.,1.]),"equivalent_radius":1.0})
    shape=Build123dReconstruction().build(sections,ReconstructionSpec(metadata={"project_profiles":True,"max_section_points":32}))
    valid=getattr(shape,"is_valid",None); valid=valid() if callable(valid) else valid
    result={"type":type(shape).__name__,"is_valid":None if valid is None else bool(valid)}
    try: result["volume"]=float(shape.volume)
    except Exception: pass
    print(json.dumps(result,indent=2))
    if result["is_valid"] is False: raise SystemExit(2)
if __name__=="__main__": main()
