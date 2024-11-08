import numpy as np
import trimesh
from scipy.optimize import differential_evolution
import pyvista as pv
import warnings

# Suppress trimesh warnings for cleaner output
warnings.filterwarnings('ignore')

def compute_primitive_invariants(vertices):
    """
    Compute scaling-free, rotation, and translation invariant primitive moments 
    for a 3D object based on a set of generating functions.
    
    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates.
    
    Returns:
        dict: Dictionary containing the computed primitive invariants.
    """
    # Center the vertices to achieve translation invariance
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # Compute raw moments up to the third order
    moment2 = np.sum(centered_vertices**2, axis=0)  # Second-order moments
    moment3 = np.sum(centered_vertices**3, axis=0)  # Third-order moments

    # Define primitive invariants by combining moments
    primitives = {
        "IP1": moment2[0] + moment2[1] + moment2[2],
        "IP2": moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2],
        "IP3": moment2[0] * moment2[1] * moment2[2] - np.sum(centered_vertices[:, 0] * centered_vertices[:, 1] * centered_vertices[:, 2]),
        "IP4": (moment3[0] + moment3[1]) * (moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2]),
        "IP5": moment2[0] * moment3[1] - moment2[1] * moment3[2] + moment2[2] * moment3[0],
        "IP6": moment3[0]**2 + moment3[1]**2 + moment3[2]**2,
        "IP7": moment2[0] * (moment3[1] + moment3[2]) - moment3[0] * moment2[1],
    }

    return primitives

def get_reference_cylinder_primitive_invariants():
    """
    Compute primitive invariants for a reference cylinder for comparison.
    
    Returns:
        dict: Primitive invariants for the reference cylinder.
    """
    # Create a reference cylinder
    cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
    vertices = cylinder.vertices

    # Compute primitive invariants
    primitives = compute_primitive_invariants(vertices)

    return primitives

def load_and_preprocess_mesh(mesh_file):
    """
    Load and preprocess the mesh: remove unreferenced vertices, center, and normalize.

    Parameters:
        mesh_file (str): Path to the mesh file.

    Returns:
        trimesh.Trimesh: Preprocessed mesh.
    """
    mesh = trimesh.load(mesh_file, force='mesh')




    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    # Center the mesh
    mesh.vertices -= mesh.centroid

    # Normalize the mesh
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= scale

    return mesh

def extract_submesh_within_cylinder(mesh, center, radius, height, orientation=np.array([0, 1, 0])):
    """
    Extract a submesh within a specified cylinder aligned along a given orientation.

    Parameters:
        mesh (trimesh.Trimesh): The original mesh.
        center (array-like): (x, y, z) coordinates of the cylinder center.
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        orientation (np.ndarray): Orientation vector of the cylinder axis.

    Returns:
        trimesh.Trimesh or None: The extracted submesh or None if invalid.
    """
    # Normalize the orientation vector
    orientation = orientation / np.linalg.norm(orientation)

    # Compute rotation matrix to align the cylinder axis with the y-axis
    y_axis = np.array([0, 1, 0])
    if np.allclose(orientation, y_axis):
        rotation_matrix = np.eye(3)
    else:
        v = np.cross(orientation, y_axis)
        s = np.linalg.norm(v)
        c = np.dot(orientation, y_axis)
        if s == 0:
            rotation_matrix = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))

    # Rotate vertices to align with y-axis
    rotated_vertices = (mesh.vertices - center).dot(rotation_matrix)

    # Define cylinder bounds
    lower_y, upper_y = -height / 2, height / 2
    within_radius = (rotated_vertices[:, 0] ** 2 + rotated_vertices[:, 2] ** 2) <= radius ** 2
    within_height = (rotated_vertices[:, 1] >= lower_y) & (rotated_vertices[:, 1] <= upper_y)
    within_cylinder = within_radius & within_height
    indices = np.where(within_cylinder)[0]

    if indices.size == 0:
        return None

    # Extract faces where all vertices are within the cylinder
    faces = mesh.faces
    mask = np.all(np.isin(faces, indices), axis=1)
    sub_faces = faces[mask]

    if sub_faces.size == 0:
        return None

    sub_vertices = rotated_vertices[indices]

    # Reindex faces
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    reindexed_faces = np.vectorize(index_map.get)(sub_faces)

    try:
        sub_mesh = trimesh.Trimesh(vertices=sub_vertices, faces=reindexed_faces, process=False)
        return sub_mesh
    except Exception as e:
        print(f"Error creating submesh: {e}")
        return None

def loss_function(params, mesh, target_invariants):
    """
    Calculate the loss based on how closely the submesh invariants match target invariants.

    Parameters:
        params (array-like): [x, y, z, radius, height] parameters of the cylinder.
        mesh (trimesh.Trimesh): The original mesh.
        target_invariants (dict): Target primitive invariants.

    Returns:
        float: The computed loss.
    """
    x, y, z, radius, height = params
    position = np.array([x, y, z])

    # Extract submesh within the cylinder
    orientation = np.array([0, 1, 0])  # Assuming cylinders aligned along y-axis
    sub_mesh = extract_submesh_within_cylinder(mesh, position, radius, height, orientation)

    if sub_mesh is None or len(sub_mesh.vertices) < 50:
        return 1e6  # Penalize invalid or small submeshes

    # Compute primitive invariants of the submesh
    invariants = compute_primitive_invariants(sub_mesh.vertices)

    # Compute loss as the sum of normalized absolute differences
    loss = 0.0
    for key in target_invariants:
        if 'mean' in key:
            invariant_key = key.replace('_mean', '')
            loss += np.abs(invariants[invariant_key] - target_invariants[key]['mean']) / target_invariants[key]['std']

    return loss

def get_reference_cylinder_invariants():
    """
    Compute the primitive invariants of a reference cylinder.

    Returns:
        dict: Primitive invariants with mean and standard deviation for each invariant.
    """
    primitives = get_reference_cylinder_primitive_invariants()
    # Assuming some standard deviation based on expected variation
    # These values can be adjusted based on empirical observations
    target_invariants = {
        'IP1': {'mean': primitives['IP1'], 'std': 0.1},
        'IP2': {'mean': primitives['IP2'], 'std': 0.1},
        'IP3': {'mean': primitives['IP3'], 'std': 0.1},
        'IP4': {'mean': primitives['IP4'], 'std': 0.1},
        'IP5': {'mean': primitives['IP5'], 'std': 0.1},
        'IP6': {'mean': primitives['IP6'], 'std': 0.1},
        'IP7': {'mean': primitives['IP7'], 'std': 0.1},
    }
    return target_invariants

def find_best_cylinders(mesh, target_invariants, bounds, max_cylinders=5, error_threshold=10.0):
    """
    Detect multiple cylinders in the mesh that match the target invariants using optimization.

    Parameters:
        mesh (trimesh.Trimesh): The original mesh.
        target_invariants (dict): Target primitive invariants with mean and standard deviation.
        bounds (list of tuples): Optimization bounds for [x, y, z, radius, height].
        max_cylinders (int): Maximum number of cylinders to detect.
        error_threshold (float): Maximum acceptable loss to consider a detection valid.

    Returns:
        list of dict: Detected cylinders with their mesh and moments.
    """
    detected = []
    search_mesh = mesh.copy()

    for i in range(max_cylinders):
        print(f"Optimizing for cylinder {i+1}...")
        result = differential_evolution(
            loss_function,
            bounds,
            args=(search_mesh, target_invariants),
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True,
            disp=False
        )

        if result.fun > error_threshold:
            print(f"No further cylinders detected. Last error: {result.fun:.4f}")
            break

        x, y, z, radius, height = result.x
        position = np.array([x, y, z])
        print(f"Detected cylinder {i+1}: Position = {position}, Radius = {radius:.4f}, Height = {height:.4f}, Error = {result.fun:.4f}")

        # Extract the submesh corresponding to the detected cylinder
        orientation = np.array([0, 1, 0])  # Assuming cylinders aligned along y-axis
        sub_mesh = extract_submesh_within_cylinder(search_mesh, position, radius, height, orientation)

        if sub_mesh is None or len(sub_mesh.vertices) < 50:
            print("Failed to extract submesh.")
            continue

        # Compute primitive invariants
        invariants = compute_primitive_invariants(sub_mesh.vertices)

        detected.append({
            'mesh': sub_mesh,
            'position': position,
            'radius': radius,
            'height': height,
            'invariants': invariants,
            'error': result.fun
        })
        print(f"Submesh Invariants: {invariants}")

        # Remove the detected cylinder from the search_mesh to find other cylinders
        search_mesh = subtract_mesh(search_mesh, sub_mesh)

        if search_mesh.is_empty:
            print("No more mesh left to search.")
            break

    return detected

def subtract_mesh(original_mesh, sub_mesh):
    """
    Subtract a submesh from the original mesh to avoid re-detection.

    Parameters:
        original_mesh (trimesh.Trimesh): The original mesh.
        sub_mesh (trimesh.Trimesh): The submesh to subtract.

    Returns:
        trimesh.Trimesh: The remaining mesh after subtraction.
    """
    try:
        remaining = original_mesh.difference(sub_mesh, engine='scad')
        return remaining
    except Exception as e:
        print(f"Error subtracting mesh: {e}")
        return original_mesh

def visualize_cylinders(original_mesh, detected_cylinders):
    """
    Visualize the original mesh and detected cylinders with distinct colors.

    Parameters:
        original_mesh (trimesh.Trimesh): The original vehicle mesh.
        detected_cylinders (list of dict): Detected cylinders with their meshes and properties.
    """
    plotter = pv.Plotter()

    # Add original mesh in light gray
    vehicle_pv = pv.wrap(original_mesh)
    plotter.add_mesh(vehicle_pv, color='lightgray', opacity=0.3, label='Vehicle')

    # Define distinct colors for detected cylinders
    color_palette = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink']

    for idx, cylinder in enumerate(detected_cylinders):
        color = color_palette[idx % len(color_palette)]
        cyl_mesh = cylinder['mesh']
        # Convert trimesh to pyvista PolyData
        if not cyl_mesh.is_watertight:
            cyl_mesh = cyl_mesh.convex_hull
        pv_faces = np.hstack([np.full((cyl_mesh.faces.shape[0], 1), 3), cyl_mesh.faces]).astype(np.int32)
        cyl_pv = pv.PolyData(cyl_mesh.vertices, pv_faces)
        plotter.add_mesh(cyl_pv, color=color, label=f"Cylinder {idx+1}")

    # Enhance visualization
    plotter.add_legend()
    plotter.add_title("Detected Cylinders in Vehicle Mesh")
    plotter.show()

def main():
    # Define target invariants (computed from a reference cylinder)
    target_invariants = get_reference_cylinder_invariants()

    # Specify the path to your mesh file
    mesh_file = "vehicle.obj"  # Replace with your mesh file path

    # Load and preprocess the mesh
    mesh = load_and_preprocess_mesh(mesh_file)
    if mesh is None:
        return

    # Define optimization bounds based on the mesh's bounding box
    bbox_min, bbox_max = mesh.bounds
    # Assuming the cylinder radius and height should be within reasonable fractions of the mesh size
    radius_bounds = (0.01, 0.2)  # Adjust based on expected cylinder sizes
    height_bounds = (0.05, 0.5)  # Adjust based on expected cylinder heights
    bounds = [
        (bbox_min[0], bbox_max[0]),  # x
        (bbox_min[1], bbox_max[1]),  # y
        (bbox_min[2], bbox_max[2]),  # z
        radius_bounds,                # radius
        height_bounds                 # height
    ]

    print(f"Normalized Mesh Bounding Box: Min {bbox_min}, Max {bbox_max}")
    print(f"Optimization Bounds:")
    print(f"  X: {bounds[0]}")
    print(f"  Y: {bounds[1]}")
    print(f"  Z: {bounds[2]}")
    print(f"  Radius: {bounds[3]}")
    print(f"  Height: {bounds[4]}")

    # Detect cylinders using optimization
    detected_cylinders = find_best_cylinders(
        mesh=mesh,
        target_invariants=target_invariants,
        bounds=bounds,
        max_cylinders=10,
        error_threshold=10.0  # Adjusted threshold based on desired sensitivity
    )

    if not detected_cylinders:
        print("No cylinders detected.")
        return

    # Visualize the detected cylinders
    visualize_cylinders(mesh, detected_cylinders)

if __name__ == "__main__":
    main()