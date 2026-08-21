"""Second medical_build example: asymmetric local aneurysm, isolated from the reference case.

The reference NPZ files are read-only. LocalDeformationSpec performs the primary
Gaussian radial enlargement; an example-only angular modulation makes the sac
asymmetric without changing the production API or reference data.
"""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import pyvista as pv
from foampilot.geometry.medical_build.analysis_data import GeometryAnalysisData, BranchRecord, SectionRecord
from foampilot.geometry.medical_build.deformation import LocalDeformationSpec, apply_local_deformation, deformation_report

ROOT=Path(__file__).resolve().parents[2]
NPZ_ROOT=Path('/home/ubuntu/vmtk_audit_extract/complex_analysis_raw_package')
OUT=ROOT/'examples/medical_build/outputs/asymmetric_aneurysm_example'; OUT.mkdir(parents=True, exist_ok=True)
N_RING=24; STEP=4

def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v,axis=1,keepdims=True); return v/np.maximum(n,1e-12)

def make_analysis():
    data=GeometryAnalysisData(coordinate_system='complex_vmtk_input', source_cap_id=4, metadata={'example':'asymmetric_aneurysm','reference_immutable':True})
    for bid in range(8):
        z=np.load(NPZ_ROOT/f'branch_{bid:02d}.npz')
        p=np.asarray(z['points'],float); t=unit(np.asarray(z['FrenetTangent'],float)); n=unit(np.asarray(z['ParallelTransportNormals'],float)); b=unit(np.asarray(z['ParallelTransportBinormals'],float)); r=np.asarray(z['MaximumInscribedSphereRadius'],float).reshape(-1); s=np.asarray(z['Abscissas'],float).reshape(-1)
        keep=np.arange(0,len(p),STEP)
        if keep[-1] != len(p)-1: keep=np.r_[keep,len(p)-1]
        sections=[]
        for sid,i in enumerate(keep):
            ang=np.linspace(0,2*np.pi,N_RING,endpoint=False)
            # Elliptical baseline gives a non-circular anatomy while remaining valid.
            pts=p[i] + (r[i]*np.cos(ang))[:,None]*n[i] + (0.76*r[i]*np.sin(ang))[:,None]*b[i]
            phase=pts.copy(); area=float(np.pi*0.76*r[i]**2); per=float(2*np.pi*r[i]*0.88)
            sections.append(SectionRecord(branch_id=bid,station_id=sid,abscissa=float(s[i]),center=p[i],tangent=t[i],normal=n[i],binormal=b[i],points=pts,phase_locked_points=phase,area=area,perimeter=per,equivalent_radius=float(np.sqrt(area/np.pi)),metadata={'source_npz':f'branch_{bid:02d}.npz','source_index':int(i)}))
        data.branches.append(BranchRecord(branch_id=bid,source_cap_id=4,target_cap_id=bid,points=p[keep],abscissas=s[keep],tangents=t[keep],length=float(s[-1]-s[0]),sections=sections))
    data.validate(); return data

def asymmetric_lobe(data, branch_id=2, center=None, sigma=10.0, lobe=0.22, angle=0.35):
    """Example-only angular modulation layered on the API deformation output."""
    changed=0
    for br in data.branches:
        if br.branch_id != branch_id: continue
        s0=center if center is not None else float(np.median([x.abscissa for x in br.sections]))
        for sec in br.sections:
            g=math.exp(-0.5*((sec.abscissa-s0)/sigma)**2)
            if g < 1e-5: continue
            q=sec.phase_locked_points-sec.center; x=q@sec.normal; y=q@sec.binormal; th=np.arctan2(y,x)
            factor=1.0 + lobe*g*np.maximum(0.0,np.cos(th-angle))
            sec.points=sec.center + q*factor[:,None]
            sec.phase_locked_points=sec.center + (sec.phase_locked_points-sec.center)*factor[:,None]
            sec.metadata['asymmetric_angular_factor']=float(g)
            changed+=1
    data.metadata['asymmetric_lobe']={'branch_id':branch_id,'center_abscissa':center,'sigma':sigma,'lobe_amplitude':lobe,'angle_rad':angle,'changed_sections':changed}
    data.validate(); return data

def mesh_from_sections(data):
    pts=[]; faces=[]; branch_ids=[]; offset=0
    for br in data.branches:
        for a,c in zip(br.sections[:-1],br.sections[1:]):
            # Each branch is emitted as connected ring strips; no junction union is claimed.
            for ring in (a.points,c.points): pts.extend(ring.tolist())
            for j in range(N_RING):
                k=(j+1)%N_RING; faces.append([4,offset+j,offset+k,offset+N_RING+k,offset+N_RING+j]); branch_ids.append(br.branch_id)
            offset += 2*N_RING
    arr=np.asarray(pts,float); poly=pv.PolyData(arr,np.asarray(faces,dtype=np.int64).ravel()); poly.cell_data['BranchId']=np.asarray(branch_ids,dtype=np.int32); return poly

if __name__=='__main__':
    ref=make_analysis()
    spec=LocalDeformationSpec(branch_ids=(2,),center_abscissa=float(np.median([x.abscissa for x in ref.branches[2].sections])),sigma=12.0,amplitude=0.85,junction_protection=8.0,max_scale=2.0)
    deformed=apply_local_deformation(ref,spec)
    deformed=asymmetric_lobe(deformed,branch_id=2,sigma=12.0,lobe=0.28,angle=0.45)
    ref.save_json(OUT/'reference_analysis.json'); deformed.save_json(OUT/'deformed_analysis.json')
    (OUT/'deformation_report.json').write_text(json.dumps(deformation_report(deformed),indent=2))
    mesh=mesh_from_sections(deformed); mesh.save(OUT/'asymmetric_aneurysm_sections.vtp'); mesh.save(OUT/'asymmetric_aneurysm_sections.stl')
    pl=pv.Plotter(off_screen=True,window_size=(1500,900)); pl.set_background('white')
    base=mesh_from_sections(ref); pl.add_mesh(base,color='lightgray',opacity=0.20,show_edges=False)
    pl.add_mesh(mesh,scalars='BranchId',cmap='viridis',show_edges=False)
    pl.add_text('Asymmetric local aneurysm — branch 2',font_size=14,color='black'); pl.camera_position='iso'; pl.reset_camera(); pl.screenshot(str(OUT/'asymmetric_aneurysm_before_after.png')); pl.close()
    print(json.dumps({'output':str(OUT),'reference_branches':len(ref.branches),'deformation':deformation_report(deformed),'mesh_points':mesh.n_points,'mesh_cells':mesh.n_cells},indent=2))
