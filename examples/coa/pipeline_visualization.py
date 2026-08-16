#!/usr/bin/env python3
"""
Visualization utilities for TBAD pipeline.
Generates images after cleaning, meshing, and results.
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import trimesh

logger = logging.getLogger(__name__)


def visualize_stl(stl_path: Path, output_path: Path, title: str = "STL Geometry"):
    """Visualize STL file with matplotlib."""
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    
    fig = plt.figure(figsize=(12, 8))
    
    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                     triangles=mesh.faces, alpha=0.8, edgecolor='none', cmap='viridis')
    ax1.set_title(f'{title} - 3D View')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    
    # Top view
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.tripcolor(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.faces, 
                  facecolors='lightblue', edgecolors='gray', linewidth=0.1)
    ax2.set_title('Top View (XY)')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.axis('equal')
    
    # Side view
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.tripcolor(mesh.vertices[:, 0], mesh.vertices[:, 2], triangles=mesh.faces,
                  facecolors='lightcoral', edgecolors='gray', linewidth=0.1)
    ax3.set_title('Side View (XZ)')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.axis('equal')
    
    # Front view
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.tripcolor(mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces,
                  facecolors='lightgreen', edgecolors='gray', linewidth=0.1)
    ax4.set_title('Front View (YZ)')
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Image saved: {output_path}")


def visualize_mesh(mesh_file: Path, output_path: Path, title: str = "Volume Mesh"):
    """Visualize volume mesh from Gmsh .msh file."""
    try:
        import gmsh
        
        gmsh.initialize()
        gmsh.model.add("visualization")
        
        # Load mesh
        gmsh.open(str(mesh_file))
        
        # Get mesh data
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        
        types, elems, nodes = gmsh.model.mesh.getElements(dim=3)
        n_elems = sum(len(e) for e in elems) if elems else 0
        
        # Get surface elements for visualization
        surf_types, surf_elems, surf_nodes = gmsh.model.mesh.getElements(dim=2)
        n_surf_elems = sum(len(e) for e in surf_elems) if surf_elems else 0
        
        gmsh.finalize()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Summary
        axes[0].text(0.5, 0.5, f"{title}\n\nNodes: {n_nodes:,}\nVol Elements: {n_elems:,}\nSurf Elements: {n_surf_elems:,}",
                ha='center', va='center', fontsize=12, transform=axes[0].transAxes)
        axes[0].set_title('Mesh Statistics')
        axes[0].axis('off')
        
        # Patch info
        axes[1].text(0.5, 0.5, f"Mesh File:\n{mesh_file.name}\n\nSize: {mesh_file.stat().st_size / 1024 / 1024:.1f} MB",
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title('Mesh File Info')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Mesh image saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not visualize mesh: {e}")


def visualize_results(case_dir: Path, output_path: Path):
    """Visualize OpenFOAM results if available."""
    try:
        # Check if results exist
        time_dirs = sorted([d for d in (case_dir / "0").iterdir() 
                           if d.is_dir() and d.name.replace('.', '').isdigit()],
                          key=lambda x: float(x.name))
        
        if not time_dirs:
            logger.warning("No time directories found for visualization")
            return
        
        fig, axes = plt.subplots(1, min(4, len(time_dirs)), figsize=(16, 4))
        if len(time_dirs) == 1:
            axes = [axes]
        
        for idx, time_dir in enumerate(time_dirs[:4]):
            ax = axes[idx] if len(axes) > 1 else axes
            ax.text(0.5, 0.5, f"Time: {time_dir.name}", 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f"t = {time_dir.name}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Results image saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not visualize results: {e}")


def check_mesh_patches(mesh_dir: Path, output_path: Path):
    """Check and visualize mesh patches."""
    try:
        boundary_file = mesh_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            logger.warning("No boundary file found")
            return
        
        content = boundary_file.read_text()
        
        # Parse patches
        patches = []
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and not line.startswith('//') and not line.startswith('}') and not line.startswith('{'):
                patch_name = line
                # Look for type in next lines
                patch_type = "unknown"
                for j in range(i+1, min(i+10, len(lines))):
                    if 'type' in lines[j]:
                        patch_type = lines[j].split('type')[-1].strip().rstrip(';').strip()
                        break
                patches.append((patch_name, patch_type))
            i += 1
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        if patches:
            y_pos = np.arange(len(patches))
            patch_names = [p[0] for p in patches]
            patch_types = [p[1] for p in patches]
            
            ax.barh(y_pos, [1] * len(patches), color='lightblue', edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(patch_names)
            ax.set_xlim(0, 2)
            ax.set_xlabel('Patch')
            ax.set_title(f'Mesh Patches ({len(patches)} found)')
            
            # Add type labels
            for i, (name, ptype) in enumerate(patches):
                ax.text(1.5, i, ptype, ha='center', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No patches found\nin boundary file', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Mesh Patches')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Patches visualization saved: {output_path}")
        logger.info(f"  Found {len(patches)} patches: {[p[0] for p in patches]}")
        
    except Exception as e:
        logger.warning(f"Could not check patches: {e}")


def generate_pipeline_report(patient_dir: Path, output_path: Path):
    """Generate a summary report image with all pipeline stages."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Stage 1: STL
        tl_stl = patient_dir / "tbad_TL_walls.stl"
        if tl_stl.exists():
            mesh = trimesh.load(tl_stl)
            if isinstance(mesh, trimesh.Trimesh):
                axes[0].tripcolor(mesh.vertices[:, 0], mesh.vertices[:, 1], 
                                 triangles=mesh.faces, facecolors='lightblue')
                axes[0].set_title(f'Step 1: STL\n{len(mesh.faces):,} faces')
                axes[0].axis('equal')
        
        # Stage 2: CAD
        axes[1].text(0.5, 0.5, 'Step 2: CAD\nCenterlines + Loft', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Step 2: CAD Reconstruction')
        axes[1].axis('off')
        
        # Stage 3: Mesh
        mesh_file = patient_dir / "mesh" / "mesh.msh"
        if mesh_file.exists():
            axes[2].text(0.5, 0.5, f'Step 3: Mesh\n{mesh_file.name}', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Step 3: Volume Mesh')
        else:
            axes[2].text(0.5, 0.5, 'Step 3: Mesh\n(not generated)', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Step 3: Volume Mesh')
        axes[2].axis('off')
        
        # Stage 4: OpenFOAM
        of_dir = patient_dir / "openfoam"
        if of_dir.exists():
            axes[3].text(0.5, 0.5, f'Step 4: OpenFOAM\n{of_dir.name}', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Step 4: OpenFOAM Case')
        else:
            axes[3].text(0.5, 0.5, 'Step 4: OpenFOAM\n(not generated)', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Step 4: OpenFOAM Case')
        axes[3].axis('off')
        
        # Summary
        summary_file = patient_dir / "pipeline_summary.json"
        if summary_file.exists():
            import json
            with open(summary_file) as f:
                summary = json.load(f)
            
            summary_text = "Pipeline Summary:\n"
            summary_text += f"Patient: {summary.get('patient_id', 'N/A')}\n"
            if 'mesh' in summary:
                summary_text += f"Mesh: {summary['mesh'].get('elements', 'N/A')} elements\n"
            if 'openfoam' in summary:
                summary_text += f"OpenFOAM: Ready\n"
            
            axes[4].text(0.1, 0.5, summary_text, ha='left', va='center', 
                        transform=axes[4].transAxes, fontsize=10)
            axes[4].set_title('Summary')
        else:
            axes[4].text(0.5, 0.5, 'No summary\navailable', 
                        ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('Summary')
        axes[4].axis('off')
        
        # Placeholder for results
        axes[5].text(0.5, 0.5, 'Results\n(pending)', 
                    ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title('CFD Results')
        axes[5].axis('off')
        
        plt.suptitle(f'TBAD Pipeline - Patient {patient_dir.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Report saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate report: {e}")
