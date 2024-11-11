import numpy as np
import trimesh
import pyvista as pv
from joblib import Parallel, delayed

def compute_primitive_invariants_from_points(points):
    """
    Compute primitive moments (IP1 to IP7) from a set of points.

    Args:
        points (np.ndarray): Nx3 array of point coordinates.

    Returns:
        dict: Dictionary containing primitive moments.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    moment2 = np.sum(centered**2, axis=0)
    moment3 = np.sum(centered**3, axis=0)
    
    # Compute primitive moments
    IP1 = moment2.sum()
    IP2 = moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2]
    IP3 = (moment2[0] * moment2[1] * moment2[2]) - np.sum(centered[:, 0] * centered[:, 1] * centered[:, 2])
    IP4 = (moment3[0] + moment3[1]) * (moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2])
    IP5 = moment2[0] * moment3[1] - moment2[1] * moment3[2] + moment2[2] * moment3[0]
    IP6 = np.sum(moment3**2)
    IP7 = moment2[0] * (moment3[1] + moment3[2]) - moment3[0] * moment2[1]
    
    return {
        "IP1": IP1,
        "IP2": IP2,
        "IP3": IP3,
        "IP4": IP4,
        "IP5": IP5,
        "IP6": IP6,
        "IP7": IP7,
    }

def normalize_invariants(invariants):
    """
    Normalize primitive moments to achieve scale invariance.

    Args:
        invariants (dict): Dictionary of primitive moments.

    Returns:
        dict: Normalized primitive moments.
    """
    normalized = {}
    for key, value in invariants.items():
        try:
            # Extract the numerical part of the key, e.g., "IP1" -> 1
            order = int(key[2:])
            # Normalize by IP1 raised to half the order
            normalized[key] = value / invariants["IP1"]**(order / 2)
        except (ValueError, KeyError) as e:
            print(f"Error normalizing key {key}: {e}")
            normalized[key] = 0  # Assign a default value or handle as needed
    return normalized

def get_reference_torus_primitive_invariants():
    """
    Create a reference torus and compute its normalized primitive moments.

    Returns:
        dict: Normalized primitive moments of the reference torus.
    """
    # Create a torus to represent a tire
    major_radius = 1.0
    minor_radius = 0.3
    torus_mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)
    voxel_pitch = 0.4  # Adjust the pitch as needed
    voxelized_torus = torus_mesh.voxelized(pitch=voxel_pitch)
    
    # Compute primitive moments
    invariants = compute_primitive_invariants_from_points(voxelized_torus.points)
    
    # Normalize the moments
    normalized_invariants = normalize_invariants(invariants)
    
    return normalized_invariants

def load_and_preprocess_mesh(mesh_file):
    """
    Load a mesh file and voxelize it.

    Args:
        mesh_file (str): Path to the mesh file.

    Returns:
        trimesh.VoxelGrid: Voxelized mesh.
    """
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh.remove_unreferenced_vertices()
    # Voxelize the mesh
    voxel_pitch = 0.1  # Adjust the pitch as needed
    voxelized = mesh.voxelized(pitch=voxel_pitch)
    return voxelized

def extract_voxels_within_neighborhood(voxel_grid, center, neighborhood_size):
    """
    Extract voxels within a cubic neighborhood around a center point.

    Args:
        voxel_grid (trimesh.VoxelGrid): Voxel grid.
        center (np.ndarray): 1x3 array representing the center point.
        neighborhood_size (float): Size of the cubic neighborhood.

    Returns:
        tuple: (selected_voxel_centers, selected_indices) or (None, None) if no voxels are found.
    """
    # Get the positions of filled voxels
    filled_voxel_centers = voxel_grid.points
    # Define a cube neighborhood
    half_size = neighborhood_size / 2
    mask = np.all(np.abs(filled_voxel_centers - center) <= half_size, axis=1)
    selected_voxel_centers = filled_voxel_centers[mask]
    selected_indices = np.where(mask)[0]
    if selected_voxel_centers.size == 0:
        return None, None
    return selected_voxel_centers, selected_indices

def compare_invariants(target_invariants, selected_invariants):
    """
    Compare normalized primitive moments of a region with the reference.

    Args:
        target_invariants (dict): Normalized primitive moments of the reference torus.
        selected_invariants (dict): Normalized primitive moments of the selected region.

    Returns:
        float: Sum of absolute differences between the invariants.
    """
    error = 0.0
    for key in target_invariants:
        error += abs(target_invariants[key] - selected_invariants.get(key, 0))
    return error

def generate_grid_points_voxel(voxel_grid, step_size=0.5):
    """
    Generate grid points within the voxel grid bounds.

    Args:
        voxel_grid (trimesh.VoxelGrid): Voxel grid.
        step_size (float): Step size for the grid.

    Returns:
        np.ndarray: Array of grid points.
    """
    bbox_min, bbox_max = voxel_grid.bounds
    x = np.arange(bbox_min[0], bbox_max[0], step_size)
    y = np.arange(bbox_min[1], bbox_max[1], step_size)
    z = np.arange(bbox_min[2], bbox_max[2], step_size)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return grid

def find_best_torus_grid_search(voxel_grid, target_invariants, grid_points, neighborhood_size, error_threshold=0.05):
    """
    Perform a grid search to find regions resembling a torus based on primitive moments.

    Args:
        voxel_grid (trimesh.VoxelGrid): Voxel grid.
        target_invariants (dict): Normalized primitive moments of the reference torus.
        grid_points (np.ndarray): Array of grid points to search.
        neighborhood_size (float): Size of the neighborhood to consider around each grid point.
        error_threshold (float): Maximum allowable error for a match.

    Returns:
        list: List of detected torus regions with their properties.
    """
    detected = []
    filled_voxel_centers = voxel_grid.points
    filled_indices = np.arange(len(filled_voxel_centers))
    
    def process_point(center):
        selected_voxel_centers, selected_indices = extract_voxels_within_neighborhood(voxel_grid, center, neighborhood_size)
        if selected_voxel_centers is not None and len(selected_voxel_centers) >= 30:
            selected_invariants = compute_primitive_invariants_from_points(selected_voxel_centers)
            normalized_selected = normalize_invariants(selected_invariants)
            error = compare_invariants(target_invariants, normalized_selected)
            if error < error_threshold:
                return {
                    'voxel_centers': selected_voxel_centers,
                    'selected_indices': selected_indices,
                    'position': center,
                    'error': error
                }
        return None
    
    # Parallel processing for efficiency
    results = Parallel(n_jobs=-1)(delayed(process_point)(pt) for pt in grid_points)
    
    for res in results:
        if res:
            # Check for overlap with previous detections
            overlap = False
            for d in detected:
                overlap_indices = np.intersect1d(res['selected_indices'], d['selected_indices'])
                if len(overlap_indices) > 0:
                    overlap = True
                    break
            if not overlap:
                detected.append(res)
    
    return detected

def visualize_tori(voxel_grid, detected):
    """
    Visualize the original voxel grid and the detected torus regions.

    Args:
        voxel_grid (trimesh.VoxelGrid): Original voxel grid.
        detected (list): List of detected torus regions.
    """
    pv_voxels = pv.PolyData(voxel_grid.points)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_voxels, color='lightgrey', opacity=0.3, label='Original Voxel Grid')
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta']
    for idx, torus in enumerate(detected):
        pv_torus_voxels = pv.PolyData(torus['voxel_centers'])
        plotter.add_mesh(pv_torus_voxels, color=colors[idx % len(colors)], opacity=0.8, label=f'Tire {idx+1}')
    
    plotter.add_legend()
    plotter.show()

def main():
    # Path to the vehicle mesh file
    mesh_file = "vehicle.obj"  # Replace with your mesh file path
    
    # Load and voxelize the vehicle mesh
    voxel_grid = load_and_preprocess_mesh(mesh_file)
    
    # Estimate neighborhood size based on expected tire size
    # You can adjust this based on your specific requirements
    neighborhood_size = 1.0  # Example value; adjust as needed
    
    # Generate target invariants for reference torus shape
    target_invariants = get_reference_torus_primitive_invariants()
    
    print("Reference Torus Normalized Primitive Moments:")
    for key, value in target_invariants.items():
        print(f"{key}: {value:.4f}")
    
    # Generate grid points for searching within the voxel grid
    grid_points = generate_grid_points_voxel(voxel_grid, step_size=0.5)
    
    print(f"Total grid points to search: {len(grid_points)}")
    
    # Find and visualize detected tires
    detected_tori = find_best_torus_grid_search(
        voxel_grid=voxel_grid,
        target_invariants=target_invariants,
        grid_points=grid_points,
        neighborhood_size=neighborhood_size,
        error_threshold=20  # Adjust this threshold based on experimentation
    )
    
    if detected_tori:
        print(f"Detected {len(detected_tori)} tire(s).")
        visualize_tori(voxel_grid, detected_tori)
    else:
        print("No tires detected.")

if __name__ == "__main__":
    main()