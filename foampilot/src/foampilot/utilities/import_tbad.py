import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv

def save_nifti_to_obj(nifti_path, output_obj_path):
    # 1. Charger le fichier de segmentation
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # 2. Appliquer Marching Cubes pour extraire la surface
    # On suppose que l'aorte est le label > 0
    verts, faces, normals, values = measure.marching_cubes(data, level=0.5)

    # 3. Sauvegarder au format OBJ
    with open(output_obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # +1 car les indices OBJ commencent à 1
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")



def prepare_tbad_mesh(nifti_label_path, output_stl_path, labels=(1, 2)):
    """
    Convert TBAD segmentation NIfTI → STL surface (with correct physical scale)
    """
    img = nib.load(nifti_label_path)
    data = img.get_fdata()

    # voxel spacing (mm)
    spacing = img.header.get_zooms()[:3]

    # extract aorta (TL + FL by default)
    binary_mask = np.isin(data, labels).astype(np.uint8)

    # marching cubes with physical scaling
    verts, faces, normals, values = measure.marching_cubes(
        binary_mask,
        level=0.5,
        spacing=spacing
    )

    # PyVista mesh
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, pv_faces)

    mesh.save(output_stl_path)
    return output_stl_path




import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh
from pathlib import Path


def extract_geometry_from_nifti(
    nii_path,
    output_stl,
    label_value=1,
    zoom_factor=2,
    sdf_sigma_mm=0.6,
):
    """
    Reconstruct a smooth geometric surface from a voxelized NIfTI segmentation
    using a signed distance field (SDF).

    Parameters
    ----------
    nii_path : str | Path
        Path to *_label.nii or *.nii.gz
    output_stl : str | Path
        Output STL path
    label_value : int
        Label to extract (e.g. 1=true lumen, 2=false lumen)
    zoom_factor : int
        SDF upsampling factor (2 is usually enough)
    sdf_sigma_mm : float
        Gaussian smoothing of SDF in mm (TBAD-safe: 0.5–0.8)
    """

    nii_path = Path(nii_path)
    output_stl = Path(output_stl)

    print(f"Loading NIfTI: {nii_path.name}")
    img = nib.load(str(nii_path))
    data = img.get_fdata()

    spacing = img.header.get_zooms()[:3]
    print(f"Voxel spacing (mm): {spacing}")

    # ------------------------------------------------------------------
    # 1. Binary label extraction (NO interpolation here)
    # ------------------------------------------------------------------
    binary = (data == label_value)

    if not np.any(binary):
        raise ValueError(f"Label {label_value} not found in volume")

    # ------------------------------------------------------------------
    # 2. Signed Distance Field (physical units, mm)
    # ------------------------------------------------------------------
    print("Computing signed distance field (SDF)...")

    dist_out = ndimage.distance_transform_edt(~binary, sampling=spacing)
    dist_in  = ndimage.distance_transform_edt(binary,  sampling=spacing)

    sdf = dist_out - dist_in

    # ------------------------------------------------------------------
    # 3. Light SDF smoothing (geometry-safe)
    # ------------------------------------------------------------------
    print(f"Smoothing SDF (sigma = {sdf_sigma_mm} mm)...")

    sigma_vox = [sdf_sigma_mm / s for s in spacing]
    sdf = ndimage.gaussian_filter(sdf, sigma=sigma_vox)

    # ------------------------------------------------------------------
    # 4. SDF super-resolution (THIS is where zoom is allowed)
    # ------------------------------------------------------------------
    if zoom_factor > 1:
        print(f"Upsampling SDF (x{zoom_factor})...")
        sdf = ndimage.zoom(sdf, zoom_factor, order=3)
        spacing = tuple(s / zoom_factor for s in spacing)

    # ------------------------------------------------------------------
    # 5. Marching Cubes on SDF = 0
    # ------------------------------------------------------------------
    print("Extracting implicit surface...")
    verts, faces, _, _ = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=spacing,
    )

    # ------------------------------------------------------------------
    # 6. Apply affine (world coordinates)
    # ------------------------------------------------------------------
    affine = img.affine.copy()
    affine[:3, :3] /= zoom_factor

    verts = (np.c_[verts, np.ones(len(verts))] @ affine.T)[:, :3]

    # ------------------------------------------------------------------
    # 7. Mesh cleanup (minimal – geometry already smooth)
    # ------------------------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=True,
    )

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.rezero()

    mesh.export(output_stl)
    print(f"✔ Geometry exported: {output_stl}")