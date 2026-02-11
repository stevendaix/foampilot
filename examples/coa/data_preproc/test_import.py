"""
PIPELINE CFD COMPLET POUR TBAD - NIfTI → STL CFD-ready avec visualisations
Méthode unique : nifti_to_stl_cfd_multisurface avec visualisations avancées
"""

from pathlib import Path
import trimesh
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt, zoom, binary_closing, binary_dilation, generate_binary_structure, gaussian_filter
import networkx as nx
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION PV
# ============================================================================
pv.set_plot_theme('document')
pv.OFF_SCREEN = False  # Mettre True pour génération automatique sans affichage

# ============================================================================
# CŒUR DU PIPELINE - CONVERSION NIFTI → STL CFD
# ============================================================================

def nifti_to_stl_cfd_multisurface(
    nifti_path: str,
    output_prefix: str,
    label_value: int = 1,
    refine_factor: int = 3,
    sdf_sigma_mm: float = 0.25,
    target_triangles: int = 400000,
    slice_offset_mm: float = 2.0,
    min_inlet_diameter_mm: float = 4.0,
    verbose: bool = True
):
    """
    Pipeline CFD complet avec walls + patches.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 PIPELINE CFD MULTI-SURFACES")
        print(f"{'='*60}")
        print(f"📁 Source: {Path(nifti_path).name}")
        print(f"🏷️  Label: {label_value}")

    # ------------------------------------------------------------------------
    # 1. CHARGEMENT ET PRÉ-TRAITEMENT
    # ------------------------------------------------------------------------
    img = nib.load(nifti_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    affine = img.affine.copy()

    # Extraction du label
    if np.unique(data).size <= 2:
        mask = data > 0.5
    else:
        mask = data == label_value

    # Préservation des connexions fines
    struct = generate_binary_structure(3, 1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    if np.sum(mask) > 100:
        mask = binary_dilation(mask, structure=struct, iterations=1)

    if verbose:
        vx_count = np.sum(mask)
        print(f"📊 Voxels actifs: {vx_count:,} ({vx_count/mask.size*100:.2f}%)")

    # ------------------------------------------------------------------------
    # 2. SIGNED DISTANCE FIELD (SDF)
    # ------------------------------------------------------------------------
    dist_out = distance_transform_edt(~mask, sampling=spacing)
    dist_in  = distance_transform_edt(mask,  sampling=spacing)
    sdf = dist_out - dist_in

    # Lissage adaptatif
    if sdf_sigma_mm > 0:
        thin_mask = (dist_in < 1.0) | (dist_out < 1.0)
        sigma_vox = [sdf_sigma_mm / s for s in spacing]
        sdf_smooth = gaussian_filter(sdf, sigma=sigma_vox)
        sdf = np.where(thin_mask, sdf, sdf_smooth)

    # Super-échantillonnage
    if refine_factor > 1:
        sdf = zoom(sdf, refine_factor, order=1)
        spacing = tuple(s / refine_factor for s in spacing)
        affine[:3, :3] /= refine_factor

    # ------------------------------------------------------------------------
    # 3. MARCHING CUBES
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
    # 4. SLICING INTELLIGENT - AXE PRINCIPAL
    # ------------------------------------------------------------------------
    center = mesh.centroid
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices - center)
    main_axis = pca.components_[0]

    proj = np.dot(mesh.vertices - center, main_axis)
    min_proj, max_proj = proj.min(), proj.max()

    cut_positions = [
        (center + main_axis * (min_proj + slice_offset_mm), -main_axis),
        (center + main_axis * (max_proj - slice_offset_mm), main_axis)
    ]

    for origin, normal in cut_positions:
        mesh = mesh.slice_plane(plane_origin=origin, 
                               plane_normal=normal, 
                               cap=False)

    if verbose:
        print(f"✂️  Slicing: 2 coupes selon axe principal")

    # ------------------------------------------------------------------------
    # 5. IDENTIFICATION DES PATCHES
    # ------------------------------------------------------------------------
    boundary_edges = mesh.edges_boundary
    boundary_graphs = mesh.edges_to_graph(boundary_edges)
    connected_boundaries = list(nx.connected_components(boundary_graphs))

    valid_boundaries = []
    for nodes in connected_boundaries:
        points = mesh.vertices[list(nodes)]
        diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        if diameter * np.mean(spacing) * 1000 > min_inlet_diameter_mm:
            valid_boundaries.append(nodes)

    if verbose:
        print(f"📍 {len(valid_boundaries)} ouvertures significatives")

    # ------------------------------------------------------------------------
    # 6. CRÉATION DES PATCHS PLANS
    # ------------------------------------------------------------------------
    patch_meshes = []
    patch_info = []

    for i, nodes in enumerate(valid_boundaries):
        boundary_points = mesh.vertices[list(nodes)]

        # Plan optimal par PCA
        pca_patch = PCA(n_components=3)
        pca_patch.fit(boundary_points - boundary_points.mean(axis=0))
        normal = pca_patch.components_[2]
        
        # Orientation vers l'extérieur
        if np.dot(normal, boundary_points.mean(axis=0) - mesh.centroid) < 0:
            normal = -normal

        # Création du patch
        try:
            plane = trimesh.geometry.plane_transform(
                origin=boundary_points.mean(axis=0),
                normal=normal
            )
            points_2d = trimesh.transform_points(boundary_points, plane)[:, :2]

            from scipy.spatial import Delaunay
            tri = Delaunay(points_2d)

            cap_verts = boundary_points[tri.simplices].reshape(-1, 3)
            cap_faces = np.arange(len(cap_verts)).reshape(-1, 3)
            cap_mesh = trimesh.Trimesh(vertices=cap_verts, faces=cap_faces)
            cap_mesh = cap_mesh.merged()
        except:
            cap_mesh = trimesh.creation.convex_hull(boundary_points)

        # Nommage
        area = cap_mesh.area
        if i == 0:
            name = "inlet"
        elif i == 1:
            name = "outlet_principal"
        else:
            name = f"outlet_{i-1}"

        # Export
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
            print(f"   └─ {name}: {area*1e6:.2f} mm², normale {normal.round(3)}")

    # ------------------------------------------------------------------------
    # 7. FERMETURE DU MAILLAGE PRINCIPAL
    # ------------------------------------------------------------------------
    mesh.fill_holes()

    if not mesh.is_watertight:
        try:
            import pymeshfix
            fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            fixer.repair(verbose=False)
            mesh = trimesh.Trimesh(fixer.v, fixer.f, process=True)
            if verbose:
                print("🔧 Réparation pymeshfix effectuée")
        except:
            mesh = mesh.convex_hull
            if verbose:
                print("⚠️  Fallback: convex hull")

    # ------------------------------------------------------------------------
    # 8. OPTIMISATION POUR SNAPPYHEXMESH
    # ------------------------------------------------------------------------
    if len(mesh.faces) > target_triangles * 1.2:
        mesh = mesh.simplify_quadratic_decimation(
            target_triangles,
            preserve_curvature=True,
            preserve_border=True
        )
        if verbose:
            print(f"🔻 Décimation: {len(mesh.faces):,} triangles")

    # Nettoyage final
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    # ------------------------------------------------------------------------
    # 9. EXPORT WALLS
    # ------------------------------------------------------------------------
    walls_path = f"{output_prefix}_walls.stl"
    mesh.export(walls_path)

    if verbose:
        print(f"\n✅ Walls: {Path(walls_path).name}")
        print(f"   ├─ Triangles: {len(mesh.faces):,}")
        print(f"   ├─ Watertight: {mesh.is_watertight}")
        print(f"   └─ Volume: {mesh.volume*1000:.2f} mm³")

    return {
        'walls': walls_path,
        'patches': [f"{output_prefix}_{name}.stl" for name in patch_meshes],
        'patch_info': patch_info,
        'stats': {
            'triangles': len(mesh.faces),
            'volume': mesh.volume,
            'watertight': mesh.is_watertight,
            'spacing': spacing,
            'label': label_value
        }
    }


# ============================================================================
# VISUALISATIONS POUR TBAD
# ============================================================================

def visualize_tbad_results(result_tl, result_fl, output_dir):
    """
    Visualisation complète des résultats TBAD (TL + FL)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🖼️  GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    
    # Chargement des maillages
    mesh_tl = trimesh.load(result_tl['walls'])
    mesh_fl = trimesh.load(result_fl['walls'])
    
    # Conversion PyVista
    pv_tl = pv.wrap(mesh_tl)
    pv_fl = pv.wrap(mesh_fl)
    
    # Mise à l'échelle (mm -> m pour OpenFOAM)
    scale_factor = 0.001
    pv_tl.points *= scale_factor
    pv_fl.points *= scale_factor
    
    viz_files = {}
    
    # ------------------------------------------------------------------------
    # 1. VISUALISATION COMBINÉE TL + FL
    # ------------------------------------------------------------------------
    p1 = pv.Plotter(window_size=[1400, 900], off_screen=True)
    p1.add_mesh(pv_tl, color="deepskyblue", opacity=0.7, 
                label="True Lumen (TL)", smooth_shading=True)
    p1.add_mesh(pv_fl, color="salmon", opacity=0.5, 
                label="False Lumen (FL)", smooth_shading=True)
    
    # Ajout des patches
    for i, patch_info in enumerate(result_tl['patch_info']):
        patch_path = result_tl['patches'][i]
        patch_mesh = trimesh.load(patch_path)
        pv_patch = pv.wrap(patch_mesh)
        pv_patch.points *= scale_factor
        p1.add_mesh(pv_patch, color="lime", opacity=0.9,
                   label=f"TL {patch_info['name']}" if i == 0 else None,
                   show_edges=True)
    
    for i, patch_info in enumerate(result_fl['patch_info']):
        patch_path = result_fl['patches'][i]
        patch_mesh = trimesh.load(patch_path)
        pv_patch = pv.wrap(patch_mesh)
        pv_patch.points *= scale_factor
        p1.add_mesh(pv_patch, color="gold", opacity=0.9,
                   label=f"FL {patch_info['name']}" if i == 0 else None,
                   show_edges=True)
    
    p1.add_legend(face='line', font_size=10, bcolor='white')
    p1.add_text("TBAD - True Lumen + False Lumen", font_size=14, position='upper_edge')
    p1.add_axes()
    path = output_dir / "01_tbad_combined.png"
    p1.screenshot(str(path), window_size=[1400, 900])
    viz_files['combined'] = path
    print(f"   ✅ {path.name}")
    p1.close()
    
    # ------------------------------------------------------------------------
    # 2. CARTE DE COURBURE
    # ------------------------------------------------------------------------
    p2 = pv.Plotter(shape=(1, 2), window_size=[1600, 700], off_screen=True)
    
    p2.subplot(0, 0)
    pv_tl_curv = pv_tl.compute_curvature(curv_type='mean')
    p2.add_mesh(pv_tl_curv, scalars='Mean_Curvature', cmap='RdBu_r',
                smooth_shading=True, scalar_bar_args={'title': 'Courbure (TL)'},
                clim=[-100, 100])
    p2.add_text("True Lumen - Courbure", font_size=12)
    
    p2.subplot(0, 1)
    pv_fl_curv = pv_fl.compute_curvature(curv_type='mean')
    p2.add_mesh(pv_fl_curv, scalars='Mean_Curvature', cmap='RdBu_r',
                smooth_shading=True, scalar_bar_args={'title': 'Courbure (FL)'},
                clim=[-100, 100])
    p2.add_text("False Lumen - Courbure", font_size=12)
    
    path = output_dir / "02_tbad_curvature.png"
    p2.screenshot(str(path), window_size=[1600, 700])
    viz_files['curvature'] = path
    print(f"   ✅ {path.name}")
    p2.close()
    
    # ------------------------------------------------------------------------
    # 3. COUPE TRANSVERSALE
    # ------------------------------------------------------------------------
    p3 = pv.Plotter(window_size=[1200, 800], off_screen=True)
    
    center_tl = pv_tl.center
    tl_slice = pv_tl.slice_plane(origin=center_tl, normal=[1, 0, 0])
    fl_slice = pv_fl.slice_plane(origin=center_tl, normal=[1, 0, 0])
    
    p3.add_mesh(tl_slice, color="blue", line_width=3, label="TL Wall")
    p3.add_mesh(fl_slice, color="red", line_width=3, label="FL Wall")
    p3.add_legend()
    p3.add_text("Coupe transversale - Dissection", font_size=14)
    p3.add_axes()
    
    path = output_dir / "03_tbad_cross_section.png"
    p3.screenshot(str(path), window_size=[1200, 800])
    viz_files['cross_section'] = path
    print(f"   ✅ {path.name}")
    p3.close()
    
    # ------------------------------------------------------------------------
    # 4. PROFILS DE DIAMÈTRE
    # ------------------------------------------------------------------------
    def compute_diameters(mesh, axis=0, n_slices=30):
        bounds = mesh.bounds
        positions = np.linspace(bounds[0][axis], bounds[1][axis], n_slices)
        diameters = []
        valid_positions = []
        
        for pos in positions:
            slice_mesh = mesh.section(plane_origin=[pos, 0, 0], 
                                     plane_normal=[1, 0, 0])
            if slice_mesh and len(slice_mesh.vertices) > 2:
                bbox = slice_mesh.bounding_box.extents
                diam = np.mean(bbox[1:3]) * 1000  # mm
                diameters.append(diam)
                valid_positions.append(pos * 1000)  # mm
        
        return np.array(valid_positions), np.array(diameters)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TL
    pos_tl, diam_tl = compute_diameters(mesh_tl)
    axes[0].plot(pos_tl, diam_tl, 'b-o', linewidth=2, markersize=4, label='TL')
    axes[0].fill_between(pos_tl, 0, diam_tl, alpha=0.3, color='blue')
    axes[0].set_xlabel('Position (mm)')
    axes[0].set_ylabel('Diamètre (mm)')
    axes[0].set_title('True Lumen - Profil de diamètre')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # FL
    pos_fl, diam_fl = compute_diameters(mesh_fl)
    axes[1].plot(pos_fl, diam_fl, 'r-o', linewidth=2, markersize=4, label='FL')
    axes[1].fill_between(pos_fl, 0, diam_fl, alpha=0.3, color='red')
    axes[1].set_xlabel('Position (mm)')
    axes[1].set_ylabel('Diamètre (mm)')
    axes[1].set_title('False Lumen - Profil de diamètre')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    path = output_dir / "04_tbad_diameters.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    viz_files['diameters'] = path
    print(f"   ✅ {path.name}")
    plt.close()
    
    # ------------------------------------------------------------------------
    # 5. PATCHES AVEC NORMALES
    # ------------------------------------------------------------------------
    p4 = pv.Plotter(window_size=[1400, 900], off_screen=True)
    
    p4.add_mesh(pv_tl, color="lightgray", opacity=0.2, label="TL")
    p4.add_mesh(pv_fl, color="lightgray", opacity=0.1, label="FL")
    
    colors = ['green', 'orange', 'purple', 'brown']
    for idx, (patch_path, patch_info) in enumerate(zip(result_tl['patches'], result_tl['patch_info'])):
        patch_mesh = trimesh.load(patch_path)
        pv_patch = pv.wrap(patch_mesh)
        pv_patch.points *= scale_factor
        p4.add_mesh(pv_patch, color=colors[idx % len(colors)], 
                   opacity=0.8, label=f"TL {patch_info['name']}")
        
        center = patch_info['center'] * scale_factor
        normal = patch_info['normal']
        p4.add_arrows(center, normal * 0.008, color='red', mag=0.005)
    
    p4.add_legend()
    p4.add_text("Patches et normales - Conditions limites CFD", font_size=14)
    p4.add_axes()
    
    path = output_dir / "05_tbad_patches_normals.png"
    p4.screenshot(str(path), window_size=[1400, 900])
    viz_files['patches'] = path
    print(f"   ✅ {path.name}")
    p4.close()
    
    print(f"\n✅ {len(viz_files)} visualisations générées")
    return viz_files


# ============================================================================
# RAPPORT CFD
# ============================================================================

def generate_cfd_report(result_tl, result_fl, output_path):
    """
    Génère un rapport CFD complet pour SnappyHexMesh
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT DE PRÉPARATION CFD - DISSECTION AORTIQUE (TBAD)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. INFORMATIONS GÉNÉRALES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Date: {np.datetime64('now')}\n")
        f.write(f"Type: TBAD (True Lumen + False Lumen)\n\n")
        
        f.write("2. STATISTIQUES TRUE LUMEN (TL)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  • Fichier walls: {Path(result_tl['walls']).name}\n")
        f.write(f"  • Triangles: {result_tl['stats']['triangles']:,}\n")
        f.write(f"  • Volume: {result_tl['stats']['volume']*1000:.2f} mm³\n")
        f.write(f"  • Watertight: {result_tl['stats']['watertight']}\n")
        f.write(f"  • Patches: {len(result_tl['patches'])}\n\n")
        
        f.write("3. STATISTIQUES FALSE LUMEN (FL)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  • Fichier walls: {Path(result_fl['walls']).name}\n")
        f.write(f"  • Triangles: {result_fl['stats']['triangles']:,}\n")
        f.write(f"  • Volume: {result_fl['stats']['volume']*1000:.2f} mm³\n")
        f.write(f"  • Watertight: {result_fl['stats']['watertight']}\n")
        f.write(f"  • Patches: {len(result_fl['patches'])}\n\n")
        
        f.write("4. CONFIGURATION SNAPPYHEXMESH\n")
        f.write("-" * 40 + "\n")
        f.write("geometry {\n")
        f.write(f"    {Path(result_tl['walls']).name} {{\n")
        f.write("        type triSurfaceMesh;\n")
        f.write("        name TL;\n")
        f.write("    }\n")
        f.write(f"    {Path(result_fl['walls']).name} {{\n")
        f.write("        type triSurfaceMesh;\n")
        f.write("        name FL;\n")
        f.write("    }\n\n")
        
        for info in result_tl['patch_info']:
            patch_file = [p for p in result_tl['patches'] if info['name'] in p][0]
            f.write(f"    {Path(patch_file).name} {{\n")
            f.write("        type triSurfaceMesh;\n")
            f.write(f"        name TL_{info['name']};\n")
            f.write("    }\n")
        
        f.write("\n")
        for info in result_fl['patch_info']:
            patch_file = [p for p in result_fl['patches'] if info['name'] in p][0]
            f.write(f"    {Path(patch_file).name} {{\n")
            f.write("        type triSurfaceMesh;\n")
            f.write(f"        name FL_{info['name']};\n")
            f.write("    }\n")
        
        f.write("}\n\n")
        
        f.write("5. RECOMMANDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("  • Résolution de maillage: 0.2-0.5 mm près des parois\n")
        f.write("  • Couches limites: 5-10 couches avec expansion 1.2\n")
        f.write("  • Raffinement local: x2 au niveau de la dissection\n")
        f.write("  • Conditions limites:\n")
        
        for info in result_tl['patch_info']:
            if 'inlet' in info['name']:
                f.write(f"      - TL_{info['name']}: velocity inlet\n")
            else:
                f.write(f"      - TL_{info['name']}: pressure outlet\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"   ✅ {output_path.name}")
    return output_path


# ============================================================================
# PIPELINE PRINCIPAL TBAD
# ============================================================================

def process_tbad_pipeline(
    nifti_path: str,
    output_dir: str = "cfd_output",
    refine_factor: int = 3,
    target_triangles_tl: int = 400000,
    target_triangles_fl: int = 300000,
    sdf_sigma_mm: float = 0.25,
    generate_viz: bool = True
):
    """
    Pipeline complet TBAD : conversion + visualisations + rapport
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("🏥 PIPELINE CFD POUR DISSECTION AORTIQUE (TBAD)")
    print("="*80)
    print(f"📁 Source: {Path(nifti_path).name}")
    print(f"📁 Output: {output_dir}")
    
    # ------------------------------------------------------------------------
    # 1. Traitement True Lumen (TL)
    # ------------------------------------------------------------------------
    print("\n" + "🔵"*40)
    print("🔵 TRAITEMENT DU TRUE LUMEN (TL) - Label 1")
    print("🔵"*40)
    
    result_tl = nifti_to_stl_cfd_multisurface(
        nifti_path=nifti_path,
        output_prefix=str(output_dir / "tbad_TL"),
        label_value=1,
        refine_factor=refine_factor,
        sdf_sigma_mm=sdf_sigma_mm,
        target_triangles=target_triangles_tl,
        slice_offset_mm=2.0,
        min_inlet_diameter_mm=4.0,
        verbose=True
    )
    
    # ------------------------------------------------------------------------
    # 2. Traitement False Lumen (FL)
    # ------------------------------------------------------------------------
    print("\n" + "🔴"*40)
    print("🔴 TRAITEMENT DU FALSE LUMEN (FL) - Label 2")
    print("🔴"*40)
    
    result_fl = nifti_to_stl_cfd_multisurface(
        nifti_path=nifti_path,
        output_prefix=str(output_dir / "tbad_FL"),
        label_value=2,
        refine_factor=refine_factor,
        sdf_sigma_mm=sdf_sigma_mm,
        target_triangles=target_triangles_fl,
        slice_offset_mm=2.0,
        min_inlet_diameter_mm=4.0,
        verbose=True
    )
    
    # ------------------------------------------------------------------------
    # 3. Visualisations
    # ------------------------------------------------------------------------
    if generate_viz:
        viz_files = visualize_tbad_results(result_tl, result_fl, output_dir)
    
    # ------------------------------------------------------------------------
    # 4. Rapport CFD
    # ------------------------------------------------------------------------
    report_path = generate_cfd_report(
        result_tl, 
        result_fl, 
        output_path=output_dir / "cfd_report.txt"
    )
    
    # ------------------------------------------------------------------------
    # 5. Résumé final
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("✅ PIPELINE TBAD TERMINÉ AVEC SUCCÈS")
    print("="*80)
    
    print(f"\n📁 Fichiers générés dans: {output_dir}")
    print("\n📂 WALLS:")
    print(f"   └─ {Path(result_tl['walls']).name}")
    print(f"   └─ {Path(result_fl['walls']).name}")
    
    print("\n🎯 PATCHES:")
    for f in sorted(output_dir.glob("*.stl")):
        if "wall" not in f.name:
            size_kb = f.stat().st_size / 1024
            print(f"   └─ {f.name} ({size_kb:.1f} KB)")
    
    if generate_viz:
        print("\n🖼️  VISUALISATIONS:")
        for f in sorted(output_dir.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            print(f"   └─ {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n📋 RAPPORT:")
    print(f"   └─ {report_path.name}")
    
    print("\n" + "="*80)
    print("🚀 PRÊT POUR SNAPPYHEXMESH !")
    print("="*80 + "\n")
    
    return {
        'tl': result_tl,
        'fl': result_fl,
        'output_dir': output_dir
    }


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------------
    # À MODIFIER SELON VOTRE SYSTÈME
    NIFTI_PATH = Path.cwd().parent / "imageTBAD" / "126_label.nii.gz"
    OUTPUT_DIR = "tbad_cfd_results"
    
    # Recherche automatique si le fichier n'existe pas
    if not NIFTI_PATH.exists():
        nifti_files = list(Path.cwd().parent.rglob("*126_label*.nii*"))
        if nifti_files:
            NIFTI_PATH = nifti_files[0]
        else:
            nifti_files = list(Path.cwd().parent.rglob("*label*.nii*"))
            if nifti_files:
                NIFTI_PATH = nifti_files[0]
                print(f"⚠️  Fichier 126 non trouvé, utilisation de: {NIFTI_PATH.name}")
            else:
                raise FileNotFoundError("Aucun fichier NIfTI trouvé")
    
    # ------------------------------------------------------------------------
    # EXÉCUTION
    # ------------------------------------------------------------------------
    results = process_tbad_pipeline(
        nifti_path=str(NIFTI_PATH),
        output_dir=OUTPUT_DIR,
        refine_factor=3,           # Résolution x3
        target_triangles_tl=400000, # TL plus complexe
        target_triangles_fl=300000, # FL plus simple
        sdf_sigma_mm=0.25,        # Préserve les détails
        generate_viz=True         # Génère les visualisations
    )
    
    # ------------------------------------------------------------------------
    # VISUALISATION INTERACTIVE FINALE (optionnelle)
    # ------------------------------------------------------------------------
    if not pv.OFF_SCREEN:
        print("\n🎬 Lancement de la visualisation interactive...")
        
        mesh_tl = trimesh.load(results['tl']['walls'])
        mesh_fl = trimesh.load(results['fl']['walls'])
        
        pv_tl = pv.wrap(mesh_tl)
        pv_fl = pv.wrap(mesh_fl)
        
        pv_tl.points *= 0.001
        pv_fl.points *= 0.001
        
        p = pv.Plotter(title="TBAD - Résultat Final CFD", window_size=[1400, 900])
        p.add_mesh(pv_tl, color="deepskyblue", opacity=0.7, smooth_shading=True, label="True Lumen")
        p.add_mesh(pv_fl, color="salmon", opacity=0.5, smooth_shading=True, label="False Lumen")
        
        # Ajout d'un patch pour exemple
        if results['tl']['patches']:
            patch = pv.wrap(trimesh.load(results['tl']['patches'][0]))
            patch.points *= 0.001
            p.add_mesh(patch, color="lime", opacity=0.9, show_edges=True, label="Inlet/Outlet")
        
        p.add_legend(face='line', font_size=12, bcolor='white')
        p.add_text("TBAD - Prêt pour SnappyHexMesh", font_size=16, position='upper_edge')
        p.add_axes()
        p.show()