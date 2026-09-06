from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def polygon_area_3d(points):
    p=np.asarray(points,float)
    if len(p)<3: return 0.0
    # Newell vector area; valid for a planar section up to numerical tolerance.
    s=np.zeros(3)
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        s += np.cross(a,b)
    return 0.5*float(np.linalg.norm(s))

def section_volume(sections):
    vals=[]
    for s in sections:
        a=polygon_area_3d(s['points']); c=np.asarray(s['center'],float); vals.append((c,a))
    vol=0.0; length=0.0
    for (c0,a0),(c1,a1) in zip(vals,vals[1:]):
        dl=float(np.linalg.norm(c1-c0)); length+=dl
        vol += dl*(a0+a1+math.sqrt(max(a0*a1,0.0)))/3.0
    return vol,length,[a for _,a in vals]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('sections',type=Path); ap.add_argument('--stl',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    d=json.loads(args.sections.read_text()); branch_rows=[]; all_ids=[]
    for b in d['branches']:
        v,l,areas=section_volume(b['sections']); ids=list(b['centerline_point_ids']); all_ids.extend(ids)
        sections=b['sections']; first_center=sections[0]['center'] if sections else None; last_center=sections[-1]['center'] if sections else None
        branch_rows.append({'branch_id':b['branch_id'],'section_count':len(areas),'volume_section_integral':v,'centerline_length':l,'area_min':min(areas,default=0),'area_max':max(areas,default=0),'area_mean':float(np.mean(areas)) if areas else 0,'first_point_id':ids[0] if ids else None,'last_point_id':ids[-1] if ids else None,'first_center':first_center,'last_center':last_center})
    report={'source':str(args.sections),'surface_points':d.get('surface_points'),'surface_cells':d.get('surface_cells'),'centerline_points':d.get('centerline_points'),'branch_count':len(branch_rows),'branches':branch_rows,'sum_branch_volume_section_integral':sum(x['volume_section_integral'] for x in branch_rows),'shared_centerline_point_ids':sorted({i for i in set(all_ids) if all_ids.count(i)>1})}
    if args.stl:
        import trimesh
        m=trimesh.load_mesh(args.stl,process=False); m.process(validate=True)
        report['stl']= {'file':str(args.stl),'volume':float(m.volume),'watertight':bool(m.is_watertight),'components':len(m.split(only_watertight=False)),'area':float(m.area)}
        report['relative_difference_stl_vs_sum_sections']=(report['stl']['volume']/report['sum_branch_volume_section_integral']-1.0) if report['sum_branch_volume_section_integral'] else None
    try:
        import networkx as nx
        g=nx.Graph()
        for b in branch_rows:
            u=('endpoint',b['branch_id'],'first'); v=('endpoint',b['branch_id'],'last'); g.add_node(u,position=b['first_center']); g.add_node(v,position=b['last_center']); g.add_edge(u,v,branch_id=b['branch_id'],length=b['centerline_length'],volume=b['volume_section_integral'])
        endpoint_pairs=[]
        for i,a in enumerate(branch_rows):
            for j,b in enumerate(branch_rows):
                if j<=i: continue
                for ka,pa in [('first',a['first_center']),('last',a['last_center'])]:
                    for kb,pb in [('first',b['first_center']),('last',b['last_center'])]:
                        if pa is None or pb is None: continue
                        dist=float(np.linalg.norm(np.asarray(pa)-np.asarray(pb)))
                        endpoint_pairs.append({'a':a['branch_id'],'a_end':ka,'b':b['branch_id'],'b_end':kb,'distance':dist})
        endpoint_pairs.sort(key=lambda x:x['distance'])
        # Add only close endpoint contacts; threshold is reported explicitly.
        threshold=5.0
        for x in endpoint_pairs:
            if x['distance']<=threshold:
                g.add_edge(('endpoint',x['a'],x['a_end']),('endpoint',x['b'],x['b_end']),distance=x['distance'],kind='spatial_contact')
        report['networkx']={'nodes':g.number_of_nodes(),'edges':g.number_of_edges(),'connected_components':nx.number_connected_components(g),'degrees':{str(k):int(v) for k,v in g.degree()},'is_tree':bool(nx.is_tree(g)),'spatial_contact_threshold':threshold,'closest_endpoint_pairs':endpoint_pairs[:20]}
    except ImportError:
        report['networkx']={'installed':False}
    args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
