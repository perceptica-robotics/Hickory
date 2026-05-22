import numpy as np
import time
import os
import sys
import argparse
import trimesh
import mesh2sdf
import pyvista as pv

# Assuming these libraries are in the same directory
from MPS import mps
from superquadrics import superquadric
from utils import *


def mesh_to_sdf_data(mesh_path, grid_resolution=100, normalize=False, level=2):
    """
    Logic from the original mesh2sdf_convert.py.
    Converts a Mesh file into the SDF data array format required by main.py.
    Now also saves the intermediate watertight mesh to a PLY file.
    """
    if not os.path.isfile(mesh_path):
        raise ValueError(f"The input file does not exist: {mesh_path}")

    print(f'Loading original mesh from {mesh_path}...')
    mesh = trimesh.load(mesh_path, force='mesh')

    # Default parameter logic from original script
    mesh_scale = 0.8
    size = grid_resolution
    fix_level = level / size

    # Normalize mesh (temporary processing for SDF computation)
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    
    print('Converting to watertight mesh and computing SDF...')
    # Generate watertight mesh and SDF
    # watertight_mesh is a trimesh object
    sdf, watertight_mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=fix_level, return_mesh=True)

    # --- SAVE WATERTIGHT MESH ---
    # Construct output filename for the mesh
    dir_name = os.path.dirname(mesh_path)
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    watertight_path = os.path.join(dir_name, f"{base_name}_watertight.ply")
    
    # Check if we need to restore scale before saving mesh (for visualization consistency)
    if not normalize:
        # If not normalizing, we want the saved mesh to match the original object's scale
        watertight_mesh.vertices = watertight_mesh.vertices / scale + center
        
    watertight_mesh.export(watertight_path)
    print(f"Saved watertight mesh to: {watertight_path}")
    # ----------------------------

    # Construct Header (Grid Config) and Body (SDF Data)
    if normalize:
        grid_config = np.array([
            [grid_resolution], 
            [-1], [1], 
            [-1], [1], 
            [-1], [1]
        ])
        final_sdf = sdf
    else:
        # Restore coordinate system scale for the SDF configuration
        grid_config = np.array([
            [grid_resolution], 
            [-1 / scale + center[0]], [1 / scale + center[0]], 
            [-1 / scale + center[1]], [1 / scale + center[1]], 
            [-1 / scale + center[2]], [1 / scale + center[2]]
        ])
        # Restore SDF value scale
        final_sdf = sdf / scale

    # Reshape data to match main.py's expected format
    writevoxel = np.reshape(np.swapaxes(final_sdf, 0, 2), (grid_resolution**3, 1))
    combined_data = np.append(grid_config, writevoxel).flatten()
    
    print('SDF computation complete.')
    return combined_data


def run_mps_visualization(sdf_data, output_sq_path=None):
    """
    Runs the MPS algorithm and visualizes/saves the results.
    """
    # Simulate data structure after reading CSV
    sdf = sdf_data
    voxelGrid = {}

    # Setup voxel grid
    voxelGrid['size'] = np.ones(3, dtype=int) * int(sdf[0])
    voxelGrid['range'] = sdf[1:7]
    sdf = sdf[7:]

    voxelGrid['x'] = np.linspace(voxelGrid['range'][0], voxelGrid['range'][1], int(voxelGrid['size'][0]))
    voxelGrid['y'] = np.linspace(voxelGrid['range'][2], voxelGrid['range'][3], int(voxelGrid['size'][1]))
    voxelGrid['z'] = np.linspace(voxelGrid['range'][4], voxelGrid['range'][5], int(voxelGrid['size'][2]))

    # Meshgrid
    x, y, z = np.meshgrid(voxelGrid['x'], voxelGrid['y'], voxelGrid['z'], indexing='ij')
    points = np.stack((x, y, z), axis=3)
    voxelGrid['points'] = points.reshape((-1, 3), order='F').T

    voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
    voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
    voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
    voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

    # Clamp SDF
    sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation'])
    print('sdf.shape: ', sdf.shape)
    print('voxelGrid["points"].shape: ', voxelGrid['points'].shape)

    start_time = time.time()

    # Run MPS
    print("Running MPS...")
    components = mps(sdf, voxelGrid)
    
    elapsed_time = time.time() - start_time
    print(f"MPS Elapsed time: {elapsed_time} seconds")

    # PyVista plotter
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Container for merging all SQ meshes
    combined_sq_mesh = None

    print(f"Reconstructing {len(components)} superquadrics...")
    
    for quadric in components:
        sq = superquadric(
            quadric[0:2],   # shape epsilon1, epsilon2
            quadric[2:5],   # scales a, b, c
            quadric[5:8],   # position x,y,z
            quadric[8:11]   # orientation
        )

        # Generate mesh for this superquadric
        mesh = superquadric_to_mesh(sq, resolution=80)
        
        # Ensure it is a PyVista object (PolyData)
        mesh = pv.wrap(mesh)

        # Add to visualization
        plotter.add_mesh(mesh, color="lightblue", opacity=1.0)
        
        # Merge into combined mesh for saving
        if combined_sq_mesh is None:
            combined_sq_mesh = mesh
        else:
            # Add meshes (creates an UnstructuredGrid)
            combined_sq_mesh = combined_sq_mesh + mesh

    # --- SAVE SUPERQUADRICS MESH ---
    if output_sq_path and combined_sq_mesh is not None:
        # FIX: Convert UnstructuredGrid back to Surface (PolyData) to support .ply
        surface_mesh = combined_sq_mesh.extract_surface()
        surface_mesh.save(output_sq_path)
        print(f"Saved combined Superquadric mesh to: {output_sq_path}")
    # -------------------------------

    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a mesh into SDF, run MPS, and save output meshes.')
    
    # Required argument: Mesh file path
    parser.add_argument('mesh_path', help='Path to the input mesh file (e.g., .obj, .stl).')
    
    # Optional arguments
    parser.add_argument('--grid_resolution', type=int, default=64, help='Voxel grid resolution (default: 100)')
    parser.add_argument('--normalize', action='store_true', help='Normalize mesh before processing (default: False)')
    parser.add_argument('--level', type=float, default=2, help='Watertighting thicken level (default: 2)')

    args = parser.parse_args()

    # Define output path for superquadrics based on input filename
    dir_name = os.path.dirname(args.mesh_path)
    base_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
    sq_output_path = os.path.join(dir_name, f"{base_name}_sq.ply")

    try:
        # 1. Convert Mesh -> SDF Data (and save watertight mesh)
        sdf_data = mesh_to_sdf_data(
            args.mesh_path, 
            grid_resolution=args.grid_resolution, 
            normalize=args.normalize, 
            level=args.level
        )
        
        # 2. Run MPS, Visualize, and Save SQ mesh
        run_mps_visualization(sdf_data, output_sq_path=sq_output_path)
        
    except Exception as e:
        print(f"Error: {e}")
