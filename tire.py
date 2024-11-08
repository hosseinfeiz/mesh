import torch
import trimesh
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import os
import gc

# Use CPU by default to prevent GPU memory issues
device = torch.device('cpu')

def compute_primitive_invariants(vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute primitive invariants based on the vertices of a mesh.
    Invariants are computed using tensor operations for efficiency.

    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) representing mesh vertices.

    Returns:
        torch.Tensor: Tensor containing the computed invariants.
    """
    centroid = vertices.mean(dim=0)
    centered = vertices - centroid
    moment2 = (centered ** 2).sum(dim=0)
    moment3 = (centered ** 3).sum(dim=0)

    invariants = torch.tensor([
        moment2.sum(),
        moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2],
        (moment2[0] * moment2[1] * moment2[2]) - (centered[:, 0] * centered[:, 1] * centered[:, 2]).sum(),
        (moment3[0] + moment3[1]) * (moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2]),
        moment2[0] * moment3[1] - moment2[1] * moment3[2] + moment2[2] * moment3[0],
        (moment3 ** 2).sum(),
        moment2[0] * (moment3[1] + moment3[2]) - moment3[0] * moment2[1],
    ], device=device)

    return invariants

def get_reference_cylinder_primitive_invariants() -> torch.Tensor:
    """
    Create a reference cylinder mesh and compute its primitive invariants.

    Returns:
        torch.Tensor: Tensor containing the reference cylinder invariants.
    """
    cylinder = trimesh.creation.cylinder(radius=2, height=0.5, sections=32)
    vertices = torch.tensor(cylinder.vertices, dtype=torch.float32, device=device)
    return compute_primitive_invariants(vertices)

def load_and_preprocess_mesh(mesh_file: str) -> trimesh.Trimesh:
    """
    Load a mesh from a file and preprocess it by removing unreferenced vertices.

    Args:
        mesh_file (str): Path to the mesh file.

    Returns:
        trimesh.Trimesh: Preprocessed mesh.
    """
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh.remove_unreferenced_vertices()
    return mesh

def extract_submesh_within_cylinder(mesh: trimesh.Trimesh, centers: torch.Tensor, radii: torch.Tensor,
                                    heights: torch.Tensor, rotations: torch.Tensor) -> list:
    """
    Extract submeshes within cylinders defined by centers, radii, heights, and rotations.
    Utilizes tensor operations for batch processing.

    Args:
        mesh (trimesh.Trimesh): The original mesh.
        centers (torch.Tensor): Tensor of shape (B, 3) for cylinder centers.
        radii (torch.Tensor): Tensor of shape (B,) for cylinder radii.
        heights (torch.Tensor): Tensor of shape (B,) for cylinder heights.
        rotations (torch.Tensor): Tensor of shape (B, 3, 3) for rotation matrices.

    Returns:
        list: List of trimesh.Trimesh objects representing submeshes.
    """
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)  # (N, 3)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)  # (M, 3)

    submeshes = []

    batch_size = 100  # Adjust based on available memory
    total = centers.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_centers = centers[start:end]  # (B, 3)
        batch_radii = radii[start:end]      # (B,)
        batch_heights = heights[start:end]  # (B,)
        batch_rotations = rotations[start:end]  # (B, 3, 3)

        # Apply rotation
        rotated_vertices = (vertices.unsqueeze(0) - batch_centers.unsqueeze(1)) @ batch_rotations.transpose(1, 2)  # (B, N, 3)

        # Define cylinder aligned with y-axis after rotation
        radial_dist = torch.norm(rotated_vertices[:, :, [0, 2]], dim=2)  # (B, N)
        radial_mask = radial_dist <= batch_radii.unsqueeze(1)  # (B, N)
        height_mask = torch.abs(rotated_vertices[:, :, 1]) <= (batch_heights / 2).unsqueeze(1)  # (B, N)
        mask = radial_mask & height_mask  # (B, N)

        for i in range(mask.shape[0]):
            current_mask = mask[i]
            selected_indices = torch.nonzero(current_mask, as_tuple=False).squeeze(1)

            if selected_indices.numel() < 100:
                continue  # Skip if not enough vertices

            # Filter faces that are entirely within the selected vertices
            face_mask = current_mask[faces].all(dim=1)  # (M,)
            selected_faces = faces[face_mask]

            if selected_faces.numel() == 0:
                continue  # Skip if no faces

            # Create submesh
            submesh_vertices = vertices[selected_indices].cpu().numpy()
            selected_faces_cpu = selected_faces.cpu().numpy()

            # Remap face indices
            index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_indices)}
            remapped_faces = []
            for face in selected_faces_cpu:
                try:
                    remapped_face = [index_map[idx] for idx in face]
                    remapped_faces.append(remapped_face)
                except KeyError:
                    continue  # Skip faces with vertices outside the mask

            if not remapped_faces:
                continue

            submesh = trimesh.Trimesh(vertices=submesh_vertices, faces=remapped_faces, process=False)
            submeshes.append(submesh)

        # Clean up to free memory
        del batch_centers, batch_radii, batch_heights, batch_rotations, rotated_vertices, radial_dist, radial_mask, height_mask, mask
        gc.collect()

    return submeshes

def compare_invariants(target_invariants: torch.Tensor, submeshes: list) -> torch.Tensor:
    """
    Compare the primitive invariants of submeshes with target invariants.

    Args:
        target_invariants (torch.Tensor): Tensor of target invariants.
        submeshes (list): List of trimesh.Trimesh objects representing submeshes.

    Returns:
        torch.Tensor: Tensor of errors for each submesh.
    """
    if not submeshes:
        return torch.tensor([], device=device)

    errors = []
    for submesh in submeshes:
        vertices = torch.tensor(submesh.vertices, dtype=torch.float32, device=device)
        inv = compute_primitive_invariants(vertices)
        error = torch.abs(inv - target_invariants).sum().item()
        errors.append(error)

    return torch.tensor(errors, device=device)

def generate_grid_points(mesh: trimesh.Trimesh, step_size: float = 2.0, max_points: int = 1000) -> torch.Tensor:
    """
    Generate a grid of points within the bounding box of the mesh.
    Uses tensor operations for efficiency.

    Args:
        mesh (trimesh.Trimesh): The mesh to generate grid points within.
        step_size (float, optional): The step size for the grid. Defaults to 2.0.
        max_points (int, optional): Maximum number of grid points. Defaults to 1000.

    Returns:
        torch.Tensor: Tensor of shape (P, 3) containing grid points.
    """
    bbox_min = torch.tensor(mesh.bounds[0], dtype=torch.float32, device=device)
    bbox_max = torch.tensor(mesh.bounds[1], dtype=torch.float32, device=device)

    grid_ranges = [torch.arange(bmin, bmax, step_size, device=device) for bmin, bmax in zip(bbox_min, bbox_max)]
    grid = torch.stack(torch.meshgrid(*grid_ranges, indexing='ij'), dim=-1).reshape(-1, 3)

    # Optionally limit to max_points by random sampling
    if grid.shape[0] > max_points:
        torch.manual_seed(42)
        indices = torch.randperm(grid.shape[0], device=device)[:max_points]
        grid = grid[indices]

    return grid

def find_best_cylinders(mesh: trimesh.Trimesh, target_invariants: torch.Tensor,
                        grid_points: torch.Tensor, radius_range: tuple, height_range: tuple,
                        angle_range: tuple, error_threshold: float = 500.0, max_cylinders: int = 5) -> list:
    """
    Find the best matching cylinders within the mesh based on primitive invariants.
    Utilizes batch processing with tensor operations.

    Args:
        mesh (trimesh.Trimesh): The mesh to search within.
        target_invariants (torch.Tensor): Tensor of target invariants.
        grid_points (torch.Tensor): Tensor of grid centers.
        radius_range (tuple): (min, max) radius values.
        height_range (tuple): (min, max) height values.
        angle_range (tuple): (min, max) angle values in degrees.
        error_threshold (float, optional): Maximum allowable error. Defaults to 500.0.
        max_cylinders (int, optional): Maximum number of cylinders to detect. Defaults to 5.

    Returns:
        list: List of dictionaries containing detected cylinder information.
    """
    detected = []
    search_mesh = mesh.copy()

    radii = torch.linspace(radius_range[0], radius_range[1], steps=3, device=device)  # (3,)
    heights = torch.linspace(height_range[0], height_range[1], steps=3, device=device)  # (3,)
    angles = torch.linspace(angle_range[0], angle_range[1], steps=3, device=device)  # (3,)

    # Generate all combinations of radii, heights, and angles using cartesian product
    combinations = torch.cartesian_prod(radii, heights, angles)  # (27, 3)

    batch_size = 100  # Adjust based on available memory
    total = grid_points.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_centers = grid_points[start:end]  # (B, 3)
        B = batch_centers.shape[0]

        # Expand combinations
        expanded_centers = batch_centers.unsqueeze(1).repeat(1, combinations.shape[0], 1).reshape(-1, 3)  # (B*27, 3)
        expanded_radii = combinations[:, 0].repeat(B)  # (B*27,)
        expanded_heights = combinations[:, 1].repeat(B)  # (B*27,)
        expanded_angles = combinations[:, 2].repeat(B)  # (B*27,)

        # Convert angles to rotation matrices
        rotations_np = R.from_euler('y', expanded_angles.cpu().numpy(), degrees=True).as_matrix()  # (B*27, 3, 3)
        rotations = torch.tensor(rotations_np, dtype=torch.float32, device=device)

        # Extract submeshes
        submeshes = extract_submesh_within_cylinder(search_mesh, expanded_centers, expanded_radii, expanded_heights, rotations)

        if not submeshes:
            continue

        # Compare invariants
        errors = compare_invariants(target_invariants, submeshes)  # (B_sub,)

        # Filter based on error threshold
        valid_indices = torch.where(errors < error_threshold)[0]
        if valid_indices.numel() == 0:
            continue

        sorted_errors, sorted_order = torch.sort(errors[valid_indices])
        for idx in sorted_order:
            if len(detected) >= max_cylinders:
                break

            submesh = submeshes[valid_indices[idx]]
            error = sorted_errors[idx].item()

            # Retrieve the corresponding parameters
            combination_idx = valid_indices[idx] % combinations.shape[0]
            center_idx = valid_indices[idx] // combinations.shape[0]

            position = expanded_centers[valid_indices[idx]].cpu().numpy()
            radius = expanded_radii[valid_indices[idx]].item()
            height = expanded_heights[valid_indices[idx]].item()
            angle = expanded_angles[valid_indices[idx]].item()

            rotation = R.from_euler('y', angle, degrees=True)

            detected.append({
                'mesh': submesh,
                'position': position,
                'radius': radius,
                'height': height,
                'error': error,
                'rotation': rotation,
                'is_watertight': submesh.is_watertight
            })

            if len(detected) >= max_cylinders:
                break

        # Clean up to free memory
        del batch_centers, expanded_centers, expanded_radii, expanded_heights, expanded_angles, rotations, rotations_np, submeshes, errors, valid_indices, sorted_errors, sorted_order
        gc.collect()

        if len(detected) >= max_cylinders:
            break

    return detected

def combine_detected_cylinders(detected_cylinders: list) -> trimesh.Trimesh:
    """
    Combine all detected cylinder meshes into a single mesh.

    Args:
        detected_cylinders (list): List of detected cylinder dictionaries.

    Returns:
        trimesh.Trimesh: Combined mesh of all detected cylinders.
    """
    watertight_meshes = [cyl['mesh'] for cyl in detected_cylinders if cyl['is_watertight']]
    if not watertight_meshes:
        return None

    combined_mesh = trimesh.util.concatenate(watertight_meshes)

    return combined_mesh

def visualize_cylinders(original: trimesh.Trimesh, detected: list):
    """
    Visualize the original mesh and detected cylinders using PyVista.

    Args:
        original (trimesh.Trimesh): The original vehicle mesh.
        detected (list): List of detected cylinder dictionaries.
    """
    pv_mesh = pv.wrap(original)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color='lightgrey', opacity=0.5, label='Original Mesh')

    watertight_color = 'green'
    non_watertight_color = 'red'

    for idx, cyl in enumerate(detected):
        pv_cyl = pv.wrap(cyl['mesh'])
        color = watertight_color if cyl['is_watertight'] else non_watertight_color
        label = f'{"Watertight" if cyl["is_watertight"] else "Non-Watertight"} Cylinder {idx+1}'
        plotter.add_mesh(pv_cyl, color=color, opacity=0.7, label=label)

    plotter.add_legend()
    plotter.show()

def main():
    """
    Main function to execute the cylinder detection and visualization.
    Utilizes batch processing and optimizations for efficient processing.
    """
    # Load reference invariants
    target_invariants = get_reference_cylinder_primitive_invariants()

    # Load and preprocess mesh
    mesh_file = "vehicle.obj"
    if not os.path.exists(mesh_file):
        print(f"Mesh file '{mesh_file}' does not exist.")
        return

    mesh = load_and_preprocess_mesh(mesh_file)

    # Generate grid points
    step_size = 0.5
    grid_points = generate_grid_points(mesh, step_size=step_size, max_points=4000)
    print(f"Generated {grid_points.shape[0]} grid points for searching.")

    # Define search parameters
    radius_range = (0.2, 2.0)
    height_range = (0.1, 2.0)
    angle_range = (0, 5)
    error_threshold = 100.0
    max_cylinders = 5

    # Find best cylinders
    detected_cylinders = find_best_cylinders(
        mesh,
        target_invariants,
        grid_points,
        radius_range,
        height_range,
        angle_range,
        error_threshold=error_threshold,
        max_cylinders=max_cylinders
    )

    if detected_cylinders:
        watertight_cylinders = [cyl for cyl in detected_cylinders if cyl['is_watertight']]
        non_watertight_cylinders = [cyl for cyl in detected_cylinders if not cyl['is_watertight']]
        print(f"Detected {len(watertight_cylinders)} watertight cylinders and {len(non_watertight_cylinders)} non-watertight cylinders.")

        # Combine and save watertight cylinders
        combined_mesh = combine_detected_cylinders(watertight_cylinders)
        if combined_mesh:
            combined_mesh.export("combined_cylinders.obj")
            print("Combined watertight cylinders mesh saved as 'combined_cylinders.obj'.")

        # Visualize all detected cylinders
        visualize_cylinders(mesh, detected_cylinders)
    else:
        print("No cylinders detected.")

if __name__ == "__main__":
    main()