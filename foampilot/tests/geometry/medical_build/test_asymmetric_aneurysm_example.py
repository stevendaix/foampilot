import numpy as np

from foampilot.geometry.medical_build.deformation import LocalDeformationSpec, apply_local_deformation
from foampilot.geometry.medical_build.analysis_data import GeometryAnalysisData, BranchRecord, SectionRecord


def _analysis():
    sections=[]
    for i,s in enumerate((0.,5.,10.,15.)):
        c=np.array([s,0.,0.]); t=np.array([1.,0.,0.]); n=np.array([0.,1.,0.]); b=np.array([0.,0.,1.]); a=np.linspace(0,2*np.pi,16,endpoint=False)
        p=c+np.cos(a)[:,None]*n+np.sin(a)[:,None]*b
        sections.append(SectionRecord(1,i,s,c,t,n,b,p,p.copy(),np.pi,2*np.pi,1.0))
    return GeometryAnalysisData(branches=[BranchRecord(1,0,1,np.array([[s,0,0] for s in (0.,5.,10.,15.)]),np.array((0.,5.,10.,15.)),np.tile([1.,0.,0.],(4,1)),15.,sections)])


def test_local_deformation_preserves_reference_and_targets_branch():
    ref=_analysis()
    before=np.stack([s.points.copy() for s in ref.branches[0].sections])
    spec=LocalDeformationSpec(branch_ids=(1,),center_abscissa=7.5,sigma=4.,amplitude=.5,junction_protection=1.)
    out=apply_local_deformation(ref,spec)
    assert np.allclose(np.stack([s.points for s in ref.branches[0].sections]),before)
    assert any(not np.allclose(a.points,b) for a,b in zip(out.branches[0].sections,before))
    assert out.metadata['local_deformation']['branch_ids']==[1]
