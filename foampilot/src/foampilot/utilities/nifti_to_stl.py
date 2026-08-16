"""Generic NIfTI to STL conversion utilities.

This module provides basic medical image-to-mesh conversion without
study-specific parameters or workflows. For TBAD-specific extraction,
see the examples/coa/data_preproc/ module.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import trimesh
from skimage.measure import marching_cubes

logger = logging.getLogger(__name__)


def nifti_to_stl(
    nifti_path: Union[str, Path],
    output_path: Union[str, Path],
    label_value: Optional[int] = None,
    smoothing_sigma: float = 0.0,
    decimate_target: Optional[int] = None,
    bin_threshold: float = 0.5,
) -> dict:
    """Convert a NIfTI segmentation mask to an STL surface mesh.

    Args:
        nifti_path: Path to the NIfTI file (.nii or .nii.gz).
        output_path: Path to write the output STL file.
        label_value: Optional label value to extract from multi-label NIfTI.
            If None, uses bin_threshold on the data.
        smoothing_sigma: Gaussian smoothing sigma in voxels applied to the
            signed distance field before marching cubes. 0 disables smoothing.
        decimate_target: Optional target face count for quadric decimation.
            If None, no decimation is applied.
        bin_threshold: Threshold for binary masks when label_value is None.

    Returns:
        Dictionary with mesh statistics: vertices, faces, watertight, etc.
    """
    import nibabel as nib

    nifti_path = Path(nifti_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load NIfTI
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.int16 if label_value is not None else np.float32)
    spacing = img.header.get_zooms()[:3]

    # Extract mask
    if label_value is not None:
        mask = data == label_value
    else:
        mask = data > bin_threshold

    if not np.any(mask):
        raise ValueError(f"Empty mask extracted from {nifti_path}")

    # Apply optional smoothing via distance transform
    if smoothing_sigma > 0:
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        dist_out = distance_transform_edt(~mask, sampling=spacing)
        dist_in = distance_transform_edt(mask, sampling=spacing)
        sdf = dist_out - dist_in
        sdf_smooth = gaussian_filter(sdf, sigma=smoothing_sigma)
        # Use smoothed SDF near walls, preserve near thin structures
        thin_mask = (dist_in < 1.0) | (dist_out < 1.0)
        sdf = np.where(thin_mask, sdf, sdf_smooth)
        mask = sdf > 0

    # Marching cubes
    try:
        verts, faces, normals, values = marching_cubes(mask, level=0.5, spacing=spacing)
    except Exception as exc:
        raise RuntimeError(f"marching_cubes failed: {exc}") from exc

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Optional decimation
    if decimate_target is not None and len(mesh.faces) > decimate_target:
        from foampilot.mesh.quality.stl_ops import decimate_stl

        result = decimate_stl(
            input_path=None,
            output_path=output_path,
            target_faces=decimate_target,
        )
        # decimate_stl expects file paths; reload mesh for stats
        mesh = trimesh.load(str(output_path), process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
    else:
        mesh.export(str(output_path))

    stats = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area": float(mesh.area),
        "bbox": mesh.bounds.tolist(),
    }

    logger.info("NIfTI->STL: %s -> %s (%d faces)", nifti_path.name, output_path.name, stats["faces"])
    return stats


def nifti_to_stl_multisurface(
    nifti_path: Union[str, Path],
    output_dir: Union[str, Path],
    labels: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Extract multiple surfaces from a multi-label NIfTI.

    Args:
        nifti_path: Path to the NIfTI file.
        output_dir: Directory to write STL files.
        labels: Optional dict mapping label_value -> output_name.
            If None, extracts all unique labels.
        **kwargs: Additional arguments passed to nifti_to_stl().

    Returns:
        Dictionary mapping output names to mesh statistics.
    """
    import nibabel as nib

    nifti_path = Path(nifti_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj)
    unique_labels = np.unique(data)
    if labels is None:
        labels = {int(lbl): f"label_{int(lbl)}" for lbl in unique_labels if lbl != 0}

    results = {}
    for lbl, name in labels.items():
        if lbl not in unique_labels:
            logger.warning("Label %d not found in %s, skipping", lbl, nifti_path)
            continue
        out_path = output_dir / f"{name}.stl"
        try:
            stats = nifti_to_stl(nifti_path, out_path, label_value=int(lbl), **kwargs)
            results[name] = stats
        except Exception as exc:
            logger.warning("Failed to extract label %d: %s", lbl, exc)
            results[name] = {"error": str(exc)}

    return results
