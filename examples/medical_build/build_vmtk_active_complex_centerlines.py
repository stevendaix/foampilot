from pathlib import Path
import json
import numpy as np
import pyvista as pv

ROOT=Path('/home/ubuntu/foampilot_pr_repo')
NPZ_ROOT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
VTP=ROOT/'examples/medical_build/case_complex/analysis/centerlines.vtp'
OUT=ROOT/'examples/medical_build/outputs'

def main():
    cl=pv.read(VTP); rows=[]; active=[]; blanked=[]
    for i in range(8):
        d=np.load(NPZ_ROOT/f'branch_{i:02d}.npz',allow_pickle=True)
        blank=int(np.asarray(d['Blanking']).reshape(-1)[0]); group=int(np.asarray(d['GroupIds']).reshape(-1)[0])
        row={'cell_id':i,'group_id':group,'blanking':blank,'points':int(len(d['points'])),'radius_min':float(np.min(d['MaximumInscribedSphereRadius'])),'radius_max':float(np.max(d['MaximumInscribedSphereRadius']))}
        rows.append(row)
        (active if blank==0 else blanked).append(i)
    active_mesh=cl.extract_cells(active); blanked_mesh=cl.extract_cells(blanked)
    OUT.mkdir(parents=True,exist_ok=True); active_mesh.save(OUT/'complex_vmtk_nonblanked_centerlines.vtu')
    report={'input':str(VTP),'vmtk_nonblanked_cells':active,'vmtk_blankeds_cells':blanked,'rows':rows,'rule':'Use only Blanking=0 for branch sections; handle Blanking=1 with bifurcation sections'}
    (OUT/'complex_vmtk_blanking_report.json').write_text(json.dumps(report,indent=2))
    pl=pv.Plotter(off_screen=True,window_size=(1800,900)); pl.set_background('white'); pl.add_mesh(cl,color='lightgray',line_width=5,render_lines_as_tubes=True,label='all centerlines')
    pl.add_mesh(active_mesh,color='green',line_width=9,render_lines_as_tubes=True,label='VMTK active Blanking=0')
    pl.add_mesh(blanked_mesh,color='orange',line_width=7,render_lines_as_tubes=True,label='VMTK bifurcation/blanked')
    pl.add_legend(bcolor='white',face='rectangle'); pl.add_text('Complex VMTK centerlines: Blanking-aware topology',font_size=15,color='black'); pl.camera_position='iso'; pl.show(screenshot=str(OUT/'complex_vmtk_blanking_comparison.png'),auto_close=True)
    print(json.dumps({'nonblanked':active,'blanked':blanked},indent=2))
if __name__=='__main__': main()
