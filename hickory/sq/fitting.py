import numpy as np
import time
import gc
import argparse
import ctypes
import pyvista as pv
from mps.MPS import mps
from mps.superquadrics import superquadric
from mps.utils import *
import os


def _candidate_libstdcxx_paths():
    candidates = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "lib", "libstdc++.so.6"))

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = os.path.dirname(os.path.dirname(conda_exe))
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
        if env_name:
            candidates.append(os.path.join(conda_root, "envs", env_name, "lib", "libstdc++.so.6"))

    return [path for path in candidates if os.path.isfile(path)]


def _preload_conda_libstdcxx():
    for lib_path in _candidate_libstdcxx_paths():
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            return lib_path
        except OSError:
            continue
    return None


_PRELOADED_LIBSTDCXX = _preload_conda_libstdcxx()

try:
    import mesh2sdf
except ImportError as exc:
    extra = ""
    if "GLIBCXX_" in str(exc):
        extra = (
            " The process is missing a new enough libstdc++. "
            f"Tried preloading: {_PRELOADED_LIBSTDCXX or 'none'}."
        )
    raise ImportError(f"Failed to import mesh2sdf.{extra}") from exc


def _build_voxel_points(x_coords, y_coords, z_coords):
    """Build 3xN voxel points with the same ordering as meshgrid(...).reshape(order='F')."""
    nx = int(x_coords.size)
    ny = int(y_coords.size)
    nz = int(z_coords.size)
    px = np.tile(x_coords, ny * nz)
    py = np.tile(np.repeat(y_coords, nx), nz)
    pz = np.repeat(z_coords, nx * ny)
    return np.vstack((px, py, pz))


def _load_mesh_for_sdf(mesh_path):
    """Load a triangle mesh with Open3D and return vertices/faces arrays."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "SQ_fitting.py now uses open3d for mesh loading. Install it with `pip install open3d`."
        ) from exc

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None:
        raise ValueError(f"Failed to load mesh: {mesh_path}")

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh became empty after cleanup: {mesh_path}")

    return vertices, faces


def mesh_to_sdf_data(mesh_path, grid_resolution=96, normalize=False, level=2):
    """ Converts a Mesh file into the SDF data array format. """
    if not os.path.isfile(mesh_path):
        raise ValueError(f"The input file does not exist: {mesh_path}")

    # print(f'Loading original mesh from {mesh_path}...')
    if grid_resolution < 16:
        raise ValueError("grid_resolution is too small; use >= 16.")
    if grid_resolution > 128:
        raise ValueError("grid_resolution > 128 is blocked to avoid OOM/system crash risk.")

    vertices, faces = _load_mesh_for_sdf(mesh_path)

    mesh_scale = 0.8
    size = grid_resolution
    fix_level = level / size

    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    
    # print('Converting to watertight mesh and computing SDF...')
    sdf, watertight_mesh = mesh2sdf.compute(
        vertices, faces, size, fix=True, level=fix_level, return_mesh=True)

    # --- SAVE WATERTIGHT MESH ---
    # dir_name = os.path.dirname(mesh_path)
    # base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    # watertight_path = os.path.join(dir_name, f"{base_name}_watertight.ply")
    
    # if not normalize:
    #     watertight_mesh.vertices = watertight_mesh.vertices / scale + center
        
    # watertight_mesh.export(watertight_path)
    # print(f"Saved watertight mesh to: {watertight_path}")
    # ----------------------------

    if normalize:
        grid_config = np.array([
            [grid_resolution], 
            [-1], [1], 
            [-1], [1], 
            [-1], [1]
        ])
        final_sdf = sdf
    else:
        grid_config = np.array([
            [grid_resolution], 
            [-1 / scale + center[0]], [1 / scale + center[0]], 
            [-1 / scale + center[1]], [1 / scale + center[1]], 
            [-1 / scale + center[2]], [1 / scale + center[2]]
        ])
        final_sdf = sdf / scale

    # Critical formatting step to match MPS requirements
    writevoxel = np.reshape(np.swapaxes(final_sdf, 0, 2), (grid_resolution**3, 1))
    # Use concatenate to avoid the extra copy behavior of np.append.
    combined_data = np.concatenate([grid_config.reshape(-1), writevoxel.reshape(-1)], axis=0)
    
    # print('SDF computation complete.')
    return combined_data

def load_sdf_from_npy(npy_path, bounds=None):
    """
    Loads a 3D numpy array (.npy) containing SDF values and formats it 
    for the run_mps_visualization function.
    
    Args:
        npy_path: Path to .npy file.
        bounds: List of 6 floats [xmin, xmax, ymin, ymax, zmin, zmax].
                If None, assumes unit cube [-1, 1] normalized.
    """
    print(f"Loading SDF directly from {npy_path}...")
    sdf_3d = np.load(npy_path)

    if sdf_3d.ndim != 3:
        raise ValueError(f"Input NPY must be a 3D array. Got shape {sdf_3d.shape}")
    
    if sdf_3d.shape[0] != sdf_3d.shape[1] or sdf_3d.shape[0] != sdf_3d.shape[2]:
        print("Warning: SDF volume is not a cube. MPS might expect cubic grids.")

    grid_resolution = sdf_3d.shape[0]

    # Set default physical bounds if none provided (Normalized Unit Cube)
    if bounds is None:
        print("No bounds provided. Assuming normalized coordinates [-1, 1].")
        bounds = [-1, 1, -1, 1, -1, 1]

    # Construct Grid Configuration Header (Size + Bounds)
    grid_config = np.array([
        [grid_resolution], 
        [bounds[0]], [bounds[1]], 
        [bounds[2]], [bounds[3]], 
        [bounds[4]], [bounds[5]]
    ])

    # Replicate the specific axis swapping used in mesh_to_sdf_data
    # This ensures x,y,z indexing aligns with the MPS algorithm
    writevoxel = np.reshape(np.swapaxes(sdf_3d, 0, 2), (grid_resolution**3, 1))
    
    # Combine header and body
    combined_data = np.append(grid_config, writevoxel).flatten()
    
    return combined_data

def _prepare_mps_input(sdf_data):
    """Helper to parse the raw SDF data array into components for MPS."""
    sdf = np.asarray(sdf_data)
    voxelGrid = {}
    voxelGrid['size'] = np.ones(3, dtype=int) * int(sdf[0])
    voxelGrid['range'] = np.asarray(sdf[1:7], dtype=np.float32)
    sdf = sdf[7:]

    voxelGrid['x'] = np.linspace(voxelGrid['range'][0], voxelGrid['range'][1], int(voxelGrid['size'][0]), dtype=np.float32)
    voxelGrid['y'] = np.linspace(voxelGrid['range'][2], voxelGrid['range'][3], int(voxelGrid['size'][1]), dtype=np.float32)
    voxelGrid['z'] = np.linspace(voxelGrid['range'][4], voxelGrid['range'][5], int(voxelGrid['size'][2]), dtype=np.float32)

    # Lower peak memory than meshgrid + stack while preserving point order.
    voxelGrid['points'] = _build_voxel_points(voxelGrid['x'], voxelGrid['y'], voxelGrid['z']).astype(np.float32, copy=False)

    voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
    voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
    voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
    voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

    sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation']).astype(np.float32, copy=False)
    return sdf, voxelGrid

# ================= MODIFIED FUNCTION =================
def run_mps_visualization(sdf_data, output_sq_path=None, output_params_path=None):
    """ Runs the MPS algorithm and visualizes/saves the results. """
    sdf, voxelGrid = _prepare_mps_input(sdf_data)
    print('sdf.shape: ', sdf.shape)
    
    # start_time = time.time()
    # 'components' is a list of arrays, each containing 11 parameters
    components = mps(sdf, voxelGrid)
    # print(f"MPS Elapsed time: {time.time() - start_time} seconds")

    # Free large SDF data as soon as possible
    del sdf, voxelGrid
    gc.collect()

    # --- SAVE PARAMETERS ---
    if output_params_path and len(components) > 0:
        # Convert list of 1D arrays to a single 2D numpy array (N x 11)
        params_array = np.array(components)
        np.save(output_params_path, params_array)
        print(f"Saved {len(components)} Superquadric parameters to: {output_params_path}")
        print("Format per row: [e1, e2, ax, ay, az, rx, ry, rz, tx, ty, tz]")
    # -----------------------

    plotter = pv.Plotter()
    plotter.set_background("white")
    meshes = []

    for quadric in components:
        sq = superquadric(
            quadric[0:2], quadric[2:5], quadric[5:8], quadric[8:11]
        )
        mesh = superquadric_to_mesh(sq, resolution=50)
        mesh = pv.wrap(mesh)
        plotter.add_mesh(mesh, color="lightblue", opacity=1.0)
        meshes.append(mesh)

    if output_sq_path and meshes:
        if hasattr(pv, "append_polydata"):
            combined_sq_mesh = pv.append_polydata(meshes)
        else:
            combined_sq_mesh = meshes[0].copy()
            for mesh in meshes[1:]:
                combined_sq_mesh = combined_sq_mesh.merge(mesh, inplace=False)
        surface_mesh = combined_sq_mesh.extract_surface().triangulate()
        surface_mesh.save(output_sq_path)
        # print(f"Saved combined Superquadric mesh to: {output_sq_path}")

    plotter.show()

    # Cleanup plotter resources
    plotter.close()
    del plotter, meshes, components
    gc.collect()
# =====================================================


def run_mps_from_sdf_data(sdf_data, output_sq_path=None, output_params_path=None, sq_mesh_resolution=50):
    """Runs MPS from precomputed sdf_data without visualization."""
    start_time = time.time()
    sdf, voxelGrid = _prepare_mps_input(sdf_data)
    components = mps(sdf, voxelGrid)
    print(f"MPS Elapsed time: {time.time() - start_time} seconds")

    del sdf, voxelGrid
    gc.collect()

    if output_params_path and len(components) > 0:
        params_array = np.array(components)
        np.save(output_params_path, params_array)
        print(f"{len(components)} Superquadrics found.")

    if output_sq_path:
        meshes = []
        for quadric in components:
            sq = superquadric(
                quadric[0:2], quadric[2:5], quadric[5:8], quadric[8:11]
            )
            mesh = superquadric_to_mesh(sq, resolution=sq_mesh_resolution)
            mesh = pv.wrap(mesh)
            meshes.append(mesh)

        if meshes:
            if hasattr(pv, "append_polydata"):
                combined_sq_mesh = pv.append_polydata(meshes)
            else:
                combined_sq_mesh = meshes[0].copy()
                for mesh in meshes[1:]:
                    combined_sq_mesh = combined_sq_mesh.merge(mesh, inplace=False)
            combined_sq_mesh.save(output_sq_path)
            del meshes, combined_sq_mesh

    del components
    gc.collect()


def run_mps(
    mesh_path,
    output_sq_path=None,
    output_params_path=None,
    grid_resolution=96,
    normalize=False,
    level=2,
    sq_mesh_resolution=50,
):
    """Runs the MPS algorithm without visualization and saves results."""
    sdf_data = mesh_to_sdf_data(
        mesh_path,
        grid_resolution=grid_resolution,
        normalize=normalize,
        level=level,
    )
    start_time = time.time()
    sdf, voxelGrid = _prepare_mps_input(sdf_data)
    components = mps(sdf, voxelGrid)
    print(f"MPS Elapsed time: {time.time() - start_time} seconds")

    # Free large SDF data
    del sdf, sdf_data, voxelGrid
    gc.collect()

    if output_params_path and len(components) > 0:
        params_array = np.array(components)
        np.save(output_params_path, params_array)
        print(f"{len(components)} Superquadrics found for {mesh_path}.")

    if output_sq_path:
        meshes = []
        for quadric in components:
            sq = superquadric(
                quadric[0:2], quadric[2:5], quadric[5:8], quadric[8:11]
            )
            mesh = superquadric_to_mesh(sq, resolution=sq_mesh_resolution)
            mesh = pv.wrap(mesh)
            meshes.append(mesh)

        if meshes:
            if hasattr(pv, "append_polydata"):
                combined_sq_mesh = pv.append_polydata(meshes)
            else:
                combined_sq_mesh = meshes[0].copy()
                for mesh in meshes[1:]:
                    combined_sq_mesh = combined_sq_mesh.merge(mesh, inplace=False)
            # surface_mesh = combined_sq_mesh.extract_surface().triangulate()
            # surface_mesh.save(output_sq_path)
            combined_sq_mesh.save(output_sq_path)
            # print(f"Saved combined Superquadric mesh to: {output_sq_path}")
            del meshes, combined_sq_mesh

    # Final cleanup of mesh objects
    del components
    gc.collect()


def fit_sq_params_for_mesh(
    mesh_path: str,
    sq_params_path: str,
    grid_resolution: int = 64,
    level: float = 2.0,
    sq_mesh_resolution: int = 50,
) -> None:
    run_mps(
        mesh_path,
        output_sq_path=None,
        output_params_path=sq_params_path,
        grid_resolution=max(16, int(grid_resolution)),
        normalize=False,
        level=float(level),
        sq_mesh_resolution=max(16, int(sq_mesh_resolution)),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Mesh or SDF .npy into MPS.')
    
    # Required argument: Mesh or NPY file path
    parser.add_argument('input_path', help='Path to input file (.obj, .stl, .ply) OR SDF volume (.npy).')
    
    # Options for Mesh input
    parser.add_argument('--grid_resolution', type=int, default=64, help='Grid resolution (for mesh input only)')
    parser.add_argument('--normalize', action='store_true', help='Normalize mesh coordinates (for mesh input only)')
    parser.add_argument('--level', type=float, default=2, help='Watertighting thicken level (for mesh input only)')
    parser.add_argument('--visualize', action='store_true', help='Enable PyVista visualization (can be memory-heavy).')
    parser.add_argument(
        '--sq-mesh-resolution',
        type=int,
        default=50,
        help='Resolution for SQ output mesh generation (lower is safer/faster).',
    )
    parser.add_argument(
        '--output-params-path',
        type=str,
        default=None,
        help='Optional output path for SQ params .npy (default: next to input mesh).',
    )

    # Options for SDF input
    # Example usage: --bounds -1 1 -1 1 -1 1
    parser.add_argument('--bounds', type=float, nargs=6, default=None, 
                        help='Physical bounds [xmin xmax ymin ymax zmin zmax] (for .npy input only)')

    args = parser.parse_args()

    # Define output path
    dir_name = os.path.dirname(args.input_path)
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]
    sq_output_path = os.path.join(dir_name, f"{base_name}_sq.vtk")
    
    # New path for parameters
    sq_params_path = args.output_params_path or os.path.join(dir_name, f"{base_name}_sq_params.npy")
    
    file_ext = os.path.splitext(args.input_path)[1].lower()

    try:
        # Branch logic based on file extension
        if file_ext == '.npy':
            # Case 1: Load pre-computed SDF directly
            sdf_data = load_sdf_from_npy(args.input_path, bounds=args.bounds)
        else:
            # Case 2: Convert Mesh to SDF
            sdf_data = mesh_to_sdf_data(
                args.input_path, 
                grid_resolution=args.grid_resolution, 
                normalize=args.normalize, 
                level=args.level
            )
        
        # 2. Run MPS
        if args.visualize:
            run_mps_visualization(
                sdf_data,
                output_sq_path=sq_output_path,
                output_params_path=sq_params_path,
            )
        else:
            # Fast/safe default path: no GUI and no heavy SQ mesh building unless explicitly requested.
            if file_ext == '.npy':
                run_mps_from_sdf_data(
                    sdf_data,
                    output_sq_path=None,
                    output_params_path=sq_params_path,
                    sq_mesh_resolution=max(16, int(args.sq_mesh_resolution)),
                )
            else:
                run_mps(
                    args.input_path,
                    output_sq_path=None,
                    output_params_path=sq_params_path,
                    grid_resolution=args.grid_resolution,
                    normalize=args.normalize,
                    level=args.level,
                    sq_mesh_resolution=max(16, int(args.sq_mesh_resolution)),
                )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
