"""Direct global blockMesh topology primitives for MedicalBuild.

This module deliberately separates global topology from Classy Blocks and
Build123d. A branch may be prepared independently, but the final dictionary
must pass global face and connectivity checks before being used by OpenFOAM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np

@dataclass
class BlockRecord:
    vertices: Tuple[int, int, int, int, int, int, int, int]
    zone: str = "fluid"
    label: str = ""

@dataclass
class GlobalBlockMesh:
    tolerance: float = 1e-7
    scale: float = 1.0
    vertices: List[np.ndarray] = field(default_factory=list)
    blocks: List[BlockRecord] = field(default_factory=list)
    boundary: Dict[str, List[Tuple[int, int, int, int]]] = field(default_factory=dict)
    merge_patch_pairs: List[Tuple[str, str]] = field(default_factory=list)
    _vertex_buckets: Dict[Tuple[int, int, int], List[int]] = field(default_factory=dict, repr=False)

    def _key(self, p: Iterable[float]) -> Tuple[int, int, int]:
        q=np.asarray(p,dtype=float)
        return tuple(np.rint(q/self.tolerance).astype(np.int64).tolist())

    def vertex(self, point: Iterable[float], *, merge: bool = True) -> int:
        p=np.asarray(point,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all(): raise ValueError("invalid vertex")
        key=self._key(p)
        if merge:
            for idx in self._vertex_buckets.get(key,[]):
                if np.linalg.norm(self.vertices[idx]-p)<=self.tolerance:
                    return idx
        idx=len(self.vertices); self.vertices.append(p); self._vertex_buckets.setdefault(key,[]).append(idx); return idx

    def add_block(self, points: Iterable[Iterable[float]], *, zone: str="fluid", label: str="", merge: bool=True) -> int:
        pts=np.asarray(list(points),dtype=float)
        if pts.shape!=(8,3): raise ValueError("a hexahedral block requires 8 points")
        ids=tuple(self.vertex(p,merge=merge) for p in pts)
        if len(set(ids))!=8: raise ValueError("degenerate block with repeated vertices")
        block=BlockRecord(ids,zone,label); self.blocks.append(block); return len(self.blocks)-1

    @staticmethod
    def _faces(ids: Tuple[int,...]) -> List[Tuple[int,int,int,int]]:
        a,b,c,d,e,f,g,h=ids
        return [(a,b,c,d),(e,h,g,f),(a,e,f,b),(b,f,g,c),(c,g,h,d),(d,h,e,a)]

    @staticmethod
    def _canonical(face: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
        rotations=[face[i:]+face[:i] for i in range(4)]
        r=tuple(reversed(face)); rotations += [r[i:]+r[:i] for i in range(4)]
        return min(rotations)

    def face_usage(self) -> Dict[Tuple[int,int,int,int], int]:
        usage: Dict[Tuple[int,int,int,int],int]={}
        for block in self.blocks:
            for face in self._faces(block.vertices):
                key=self._canonical(face); usage[key]=usage.get(key,0)+1
        return usage

    def connected_components(self) -> List[List[int]]:
        face_to_blocks: Dict[Tuple[int,int,int,int],List[int]]={}
        for i,b in enumerate(self.blocks):
            for f in self._faces(b.vertices): face_to_blocks.setdefault(self._canonical(f),[]).append(i)
        adj=[set() for _ in self.blocks]
        for owners in face_to_blocks.values():
            for a in owners:
                for b in owners:
                    if a!=b: adj[a].add(b)
        unseen=set(range(len(self.blocks))); comps=[]
        while unseen:
            root=unseen.pop(); stack=[root]; comp=[root]
            while stack:
                a=stack.pop()
                for b in adj[a]:
                    if b in unseen: unseen.remove(b); stack.append(b); comp.append(b)
            comps.append(comp)
        return comps

    def validate(self, *, require_connected: bool=True) -> Dict[str,object]:
        usage=self.face_usage(); internal=sum(v==2 for v in usage.values()); boundary=sum(v==1 for v in usage.values()); invalid=[k for k,v in usage.items() if v>2]
        comps=self.connected_components(); bad=[]
        for i,b in enumerate(self.blocks):
            p=np.asarray([self.vertices[j] for j in b.vertices]);
            # Signed volume proxy using the six tetrahedra from vertex 0.
            vol=0.0
            for j in (1,2,3): vol += np.dot(p[j]-p[0],np.cross(p[4]-p[0],p[j+1]-p[0]))/6.0
            if abs(vol)<=self.tolerance**3: bad.append(i)
        ok=not bad and not invalid and (not require_connected or len(comps)==1)
        return {'ok':bool(ok),'n_vertices':len(self.vertices),'n_blocks':len(self.blocks),'internal_faces':internal,'boundary_faces':boundary,'nonmanifold_faces':len(invalid),'degenerate_blocks':bad,'components':[len(c) for c in comps],'connected':len(comps)==1}

    def write(self, path: str | Path) -> Path:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            f.write('FoamFile\n{ version 2.0; format ascii; class dictionary; object blockMeshDict; }\n\n')
            f.write(f'scale {self.scale};\n\nvertices\n(\n')
            for p in self.vertices: f.write('    (% .12g % .12g % .12g)\n'%tuple(p))
            f.write(');\n\nblocks\n(\n')
            for b in self.blocks: f.write('    hex (%s) (%d %d %d) simpleGrading (1 1 1)\n'%(' '.join(map(str,b.vertices)),1,1,1))
            f.write(');\n\nedges ();\n\nboundary\n(\n')
            for name,faces in self.boundary.items():
                f.write(f'    {name}\n    {{ type patch; faces\n    (\n')
                for face in faces: f.write('        (%s)\n'%(' '.join(map(str,face))))
                f.write('    ); }\n')
            f.write(');\n\nmergePatchPairs\n(\n')
            for a,b in self.merge_patch_pairs: f.write(f'    ({a} {b})\n')
            f.write(');\n')
        return path
