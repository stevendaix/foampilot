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
