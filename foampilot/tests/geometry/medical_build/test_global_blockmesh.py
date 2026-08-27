import numpy as np
from foampilot.geometry.medical_build.global_blockmesh import GlobalBlockMesh
def test_global_connectivity_validation():
    m=GlobalBlockMesh(tolerance=1e-8)
    # Two adjacent hexes sharing one face must form one component.
    m.add_block([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    m.add_block([[1,0,0],[2,0,0],[2,1,0],[1,1,0],[1,0,1],[2,0,1],[2,1,1],[1,1,1]])
    r=m.validate()
    assert r['connected'] and r['nonmanifold_faces']==0 and r['ok'], r

    # A disconnected block must be rejected when global connectivity is required.
    n=GlobalBlockMesh(tolerance=1e-8)
    n.add_block([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    n.add_block([[3,0,0],[4,0,0],[4,1,0],[3,1,0],[3,0,1],[4,0,1],[4,1,1],[3,1,1]])
    r=n.validate()
    assert not r['connected'] and not r['ok'], r
