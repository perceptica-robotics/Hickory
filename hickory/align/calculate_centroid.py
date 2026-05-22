import numpy as np
import trimesh
from scipy.special import beta
from numpy.linalg import norm
import os
import time

# --- 1. Centroid calculation from SQ ---
def compute_sq_centroid(file_path):
    """
    Parses (N, 11) SQ parameters to compute the exact object centroid.
    Only returns the centroid coordinate.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    data = np.load(file_path)
    
    # Slicing
    e1, e2 = data[:, 0], data[:, 1]
    ax, ay, az = data[:, 2], data[:, 3], data[:, 4]
    sq_translations = data[:, 8:11] 
    
    # Volume weights
    # Calculation is purely analytical (O(1) with respect to surface resolution)
    term_scale = 2.0 * ax * ay * az
    term_shape = e1 * e2
    term_beta = beta(e1/2 + 1, e1) * beta(e2/2, e2/2)
    volumes = np.nan_to_num(term_scale * term_shape * term_beta)
    
    # Weighted Centroid
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return np.mean(sq_translations, axis=0)
        
    weighted_positions = volumes[:, np.newaxis] * sq_translations
    object_centroid = np.sum(weighted_positions, axis=0) / total_volume
    
    return object_centroid

# --- 2. Centroid calculation from mesh ---
def compute_mesh_properties(mesh_path):
    """
    Loads mesh and returns centroid AND bounding box diagonal size.
    """
    # Loading and processing a mesh involves parsing thousands of faces
    mesh = trimesh.load(mesh_path)
    
    # mesh.extents returns [len_x, len_y, len_z]
    bbox_diagonal = norm(mesh.extents)
    
    # mesh.center_mass requires volume integration over surface triangles
    return mesh.center_mass, bbox_diagonal

# --- 3. Combined Evaluation (Accuracy + Speed) ---
def evaluate_centroid_accuracy(mesh_path, sq_file):
    print(f"\nEvaluating: {os.path.basename(mesh_path)}")
    print(f"SQ File:    {os.path.basename(sq_file)}")
    
    # --- Measure SQ Time ---
    start_time_sq = time.perf_counter()
    com_sq = compute_sq_centroid(sq_file)
    end_time_sq = time.perf_counter()
    
    if com_sq is None: return
    time_sq_ms = (end_time_sq - start_time_sq) * 1000.0

    # --- Measure Mesh Time ---
    start_time_mesh = time.perf_counter()
    com_mesh, obj_size = compute_mesh_properties(mesh_path)
    end_time_mesh = time.perf_counter()
    
    time_mesh_ms = (end_time_mesh - start_time_mesh) * 1000.0
    
    # --- Calculate Errors ---
    abs_error = norm(com_sq - com_mesh)
    
    if obj_size > 0:
        rel_error = abs_error / obj_size
    else:
        rel_error = float('inf')
    
    # --- Output Report ---
    print("-" * 65)
    print(f"{'Metric':<20} | {'SQ Estimated':<15} | {'Mesh GT':<15}")
    print("-" * 65)
    print(f"{'Centroid X':<20} | {com_sq[0]:<15.4f} | {com_mesh[0]:<15.4f}")
    print(f"{'Centroid Y':<20} | {com_sq[1]:<15.4f} | {com_mesh[1]:<15.4f}")
    print(f"{'Centroid Z':<20} | {com_sq[2]:<15.4f} | {com_mesh[2]:<15.4f}")
    print("-" * 65)
    print(f"Object Size (Diag): {obj_size:.4f} m")
    print(f"Absolute Error:     {abs_error:.4f} m")
    print(f"Relative Error:     {rel_error:.2%}")
    print("-" * 65)
    
    # --- Time Comparison ---
    print(f"{'Computation Time':<20} | {'Time (ms)':<15} | {'Speed Factor':<15}")
    print("-" * 65)
    print(f"{'SQ Calculation':<20} | {time_sq_ms:<15.4f} | {'1.0'}")
    print(f"{'Mesh Calculation':<20} | {time_mesh_ms:<15.4f} | {time_mesh_ms/time_sq_ms:<15.1f}")
    print("-" * 65)


# --- Usage ---
if __name__ == "__main__":
    evaluate_centroid_accuracy("output_sam3d/bear_8.glb", "output_sam3d/bear_8_sq_params.npy")