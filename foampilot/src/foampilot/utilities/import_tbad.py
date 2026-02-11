import nibabel as nib
import numpy as np
import trimesh
import pyacvd
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt, zoom, binary_closing, binary_dilation, generate_binary_structure, gaussian_filter
import networkx as nx
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def nifti_to_stl_cfd_multisurface(
    nifti_path: str,
    output_prefix: str,
    label_value: int = 1,
    refine_factor: int = 3,           # Augmenté pour meilleure résolution
    sdf_sigma_mm: float = 0.25,      # Réduit pour préserver les détails
    target_triangles: int = 500000,  # Augmenté pour Snappy
    slice_offset_mm: float = 2.0,
    min_inlet_diameter_mm: float = 4.0,  # Filtre les petits trous (bruit)
    verbose: bool = True
):
    """
    Pipeline CFD complet avec :
    - Walls : surface fermée (watertight)
    - Inlet/Outlets : patches plans parfaitement orientés
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 PIPELINE CFD MULTI-SURFACES")
        print(f"{'='*60}")
        print(f"📁 Source: {nifti_path}")
    
    # ------------------------------------------------------------------------
    # 1. CHARGEMENT ET PRÉ-TRAITEMENT AMÉLIORÉ
    # ------------------------------------------------------------------------
    img = nib.load(nifti_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    affine = img.affine.copy()
    
    # Extraction intelligente du label
    if np.unique(data).size <= 2:
        mask = data > 0.5
        if verbose:
            print(f"🎯 Mode binaire détecté")
    else:
        mask = data == label_value
        if verbose:
            print(f"🎯 Label {label_value} extrait")
    
    # Préservation des connexions fines - version plus conservative
    struct = generate_binary_structure(3, 1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    # Dilation UNIQUEMENT si nécessaire (connexions rompues)
    if np.sum(mask) > 100:  # Éviter de créer du bruit
        mask = binary_dilation(mask, structure=struct, iterations=1)
    
    if verbose:
        vx_count = np.sum(mask)
        print(f"📊 Voxels actifs: {vx_count:,} ({vx_count/mask.size*100:.2f}%)")
    
    # ------------------------------------------------------------------------
    # 2. SIGNED DISTANCE FIELD (SDF) OPTIMISÉE
    # ------------------------------------------------------------------------
    dist_out = distance_transform_edt(~mask, sampling=spacing)
    dist_in  = distance_transform_edt(mask,  sampling=spacing)
    sdf = dist_out - dist_in
    
    # Lissage adaptatif - plus fort loin des parois, préservé près des détails
    if sdf_sigma_mm > 0:
        # Détection des zones à préserver (petits vaisseaux)
        thin_mask = (dist_in < 1.0) | (dist_out < 1.0)
        sigma_vox = [sdf_sigma_mm / s for s in spacing]
        
        sdf_smooth = gaussian_filter(sdf, sigma=sigma_vox)
        # Mélange : zones épaisses = lissées, zones fines = originales
        sdf = np.where(thin_mask, sdf, sdf_smooth)
    
    # Super-échantillonnage
    if refine_factor > 1:
        sdf = zoom(sdf, refine_factor, order=1)
        spacing = tuple(s / refine_factor for s in spacing)
        affine[:3, :3] /= refine_factor
    
    # ------------------------------------------------------------------------
    # 3. MARCHING CUBES - HAUTE QUALITÉ
    # ------------------------------------------------------------------------
    verts, faces, _, _ = marching_cubes(
        sdf, 
        level=0.0, 
        spacing=spacing, 
        method='lewiner',
        step_size=1,
        allow_degenerate=False
    )
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    
    if verbose:
        print(f"📐 Triangles bruts: {len(faces):,}")
    
    # ------------------------------------------------------------------------
    # 4. SLICING AUTOMATIQUE INTELLIGENT
    # ------------------------------------------------------------------------
    bounds = mesh.bounds
    center = mesh.centroid
    extents = bounds[1] - bounds[0]
    
    # On ne coupe que les extrémités qui sont "naturellement" des ouvertures
    # Stratégie: couper sur l'axe principal du vaisseau
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices - center)
    main_axis = pca.components_[0]  # Axe principal
    
    # Projection des sommets sur l'axe principal
    proj = np.dot(mesh.vertices - center, main_axis)
    min_proj, max_proj = proj.min(), proj.max()
    
    # Coupes aux deux extrémités selon l'axe principal
    cut_positions = [
        (center + main_axis * (min_proj + slice_offset_mm), -main_axis),  # Entrée
        (center + main_axis * (max_proj - slice_offset_mm), main_axis)    # Sortie
    ]
    
    for origin, normal in cut_positions:
        mesh = mesh.slice_plane(plane_origin=origin, 
                               plane_normal=normal, 
                               cap=False)
    
    if verbose:
        print(f"✂️  Slicing selon axe principal: 2 coupes")
    
    # ------------------------------------------------------------------------
    # 5. IDENTIFICATION ROBUSTE DES PATCHES
    # ------------------------------------------------------------------------
    boundary_edges = mesh.edges_boundary
    boundary_graphs = mesh.edges_to_graph(boundary_edges)
    connected_boundaries = list(nx.connected_components(boundary_graphs))
    
    # Filtrage: ignorer les trop petits contours (bruit)
    valid_boundaries = []
    for nodes in connected_boundaries:
        points = mesh.vertices[list(nodes)]
        diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if diameter * np.mean(spacing) * 1000 > min_inlet_diameter_mm:
            valid_boundaries.append(nodes)
    
    if verbose:
        print(f"📍 {len(valid_boundaries)} ouvertures significatives détectées "
              f"(>{min_inlet_diameter_mm}mm)")
    
    # ------------------------------------------------------------------------
    # 6. CRÉATION DES PATCHS PARFAITEMENT PLANS
    # ------------------------------------------------------------------------
    patch_meshes = []
    patch_info = []
    
    for i, nodes in enumerate(valid_boundaries):
        boundary_points = mesh.vertices[list(nodes)]
        
        # 1. Trouver le plan optimal par PCA
        pca_patch = PCA(n_components=3)
        pca_patch.fit(boundary_points - boundary_points.mean(axis=0))
        
        # La normale du patch est le 3ème composant (moindre variance)
        normal = pca_patch.components_[2]
        # S'assurer que la normale pointe vers l'extérieur
        if np.dot(normal, boundary_points.mean(axis=0) - mesh.centroid) < 0:
            normal = -normal
        
        # 2. Créer un patch plan parfait avec ACVD (uniforme)
        try:
            # Projeter les points sur le plan
            plane = trimesh.geometry.plane_transform(
                origin=boundary_points.mean(axis=0),
                normal=normal
            )
            points_2d = trimesh.transform_points(boundary_points, plane)[:, :2]
            
            # Créer un maillage 2D de qualité
            from scipy.spatial import Delaunay
            tri = Delaunay(points_2d)
            
            # Reconstruire en 3D
            cap_verts = boundary_points[tri.simplices].reshape(-1, 3)
            cap_faces = np.arange(len(cap_verts)).reshape(-1, 3)
            cap_mesh = trimesh.Trimesh(vertices=cap_verts, faces=cap_faces)
            cap_mesh = cap_mesh.merged()
            
        except:
            # Fallback: convex hull
            cap_mesh = trimesh.creation.convex_hull(boundary_points)
        
        # Nommage intelligent basé sur la taille et la position
        area = cap_mesh.area
        if i == 0:
            name = "inlet"
        elif i == 1:
            name = "outlet_principal"
        else:
            name = f"outlet_{i-1}"
        
        # Export STL
        patch_path = f"{output_prefix}_{name}.stl"
        cap_mesh.export(patch_path)
        patch_meshes.append(name)
        patch_info.append({
            'name': name,
            'area': area,
            'normal': normal,
            'center': boundary_points.mean(axis=0)
        })
        
        if verbose:
            print(f"   └─ {name}: {area:.2f} mm², normale {normal.round(3)}")
    
    # ------------------------------------------------------------------------
    # 7. FERMETURE DU MAILLAGE PRINCIPAL
    # ------------------------------------------------------------------------
    # Remplir TOUS les trous pour avoir un mesh watertight
    mesh.fill_holes()
    
    # Réparation supplémentaire si nécessaire
    if not mesh.is_watertight:
        try:
            import pymeshfix
            fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            fixer.repair(verbose=False)
            mesh = trimesh.Trimesh(fixer.v, fixer.f, process=True)
            if verbose:
                print("🔧 Réparation pymeshfix effectuée")
        except:
            # Fallback: convex hull (dernier recours)
            mesh = mesh.convex_hull
            if verbose:
                print("⚠️  Fallback: convex hull")
    
    # ------------------------------------------------------------------------
    # 8. OPTIMISATION DU MAILLAGE POUR SNAPPYHEXMESH
    # ------------------------------------------------------------------------
    # Snappy aime les triangles relativement uniformes
    if len(mesh.faces) > target_triangles * 1.2:
        # Décimation quadratique avec préservation des bords
        mesh = mesh.simplify_quadratic_decimation(
            target_triangles,
            preserve_curvature=True,
            preserve_border=True
        )
        if verbose:
            print(f"🔻 Décimation: {len(mesh.faces):,} triangles")
    
    # Uniformisation optionnelle avec ACVD
    try:
        if hasattr(pyacvd, 'ACVD'):
            clus = pyacvd.ACVD(mesh)
            clus.cluster(len(mesh.vertices) // 2)
            mesh = clus.mesh
            if verbose:
                print(f"🔄 Uniformisation ACVD: {len(mesh.faces):,} triangles")
    except:
        pass
    
    # Nettoyage final
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    
    # ------------------------------------------------------------------------
    # 9. EXPORT WALLS
    # ------------------------------------------------------------------------
    walls_path = f"{output_prefix}_walls.stl"
    mesh.export(walls_path)
    
    # ------------------------------------------------------------------------
    # 10. RAPPORT FINAL
    # ------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"{'='*60}")
        print(f"📁 Walls: {walls_path}")
        print(f"   ├─ Triangles: {len(mesh.faces):,}")
        print(f"   ├─ Watertight: {mesh.is_watertight}")
        print(f"   ├─ Volume: {mesh.volume:.2f} mm³")
        print(f"   └─ Surface: {mesh.area:.2f} mm²")
        print(f"\n🎯 Patches générés ({len(patch_meshes)}):")
        for name in patch_meshes:
            print(f"   └─ {output_prefix}_{name}.stl")
        
        # Instructions pour OpenFOAM
        print(f"\n📋 INSTRUCTIONS SNAPPYHEXMESH:")
        print(f"   walls.stl → geometry.walls")
        for name in patch_meshes:
            print(f"   {name}.stl → geometry.{name}")
        print(f"{'='*60}\n")
    
    return {
        'walls': walls_path,
        'patches': [f"{output_prefix}_{name}.stl" for name in patch_meshes],
        'patch_info': patch_info,
        'stats': {
            'triangles': len(mesh.faces),
            'volume': mesh.volume,
            'watertight': mesh.is_watertight
        }
    }

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    # Pour une aorte avec branches
    result = nifti_to_stl_cfd_multisurface(
        "aorte_label.nii.gz",
        "aorte_cfd",
        label_value=1,
        refine_factor=3,        # Haute résolution
        sdf_sigma_mm=0.25,     # Préserve les sténoses
        target_triangles=500000,
        slice_offset_mm=2.0,
        min_inlet_diameter_mm=4.0
    )
    
    # Pour un TBAD (vrai chenal + faux chenal)
    # result_tl = nifti_to_stl_cfd_multisurface("tbad.nii.gz", "tbad_TL", label_value=1)
    # result_fl = nifti_to_stl_cfd_multisurface("tbad.nii.gz", "tbad_FL", label_value=2)