import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def compute_primitive_invariants(vertices):
    """
    Compute primitive invariants based on the vertices of a mesh.
    """
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    moment2 = np.sum(centered**2, axis=0)
    moment3 = np.sum(centered**3, axis=0)
    
    return {
        "IP1": moment2.sum(),
        "IP2": moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2],
        "IP3": (moment2[0] * moment2[1] * moment2[2]) - np.sum(centered[:, 0] * centered[:, 1] * centered[:, 2]),
        "IP4": (moment3[0] + moment3[1]) * (moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2]),
        "IP5": moment2[0] * moment3[1] - moment2[1] * moment3[2] + moment2[2] * moment3[0],
        "IP6": np.sum(moment3**2),
        "IP7": moment2[0] * (moment3[1] + moment3[2]) - moment3[0] * moment2[1],
    }

def get_reference_cylinder_primitive_invariants():
    """
    Create a reference cylinder mesh and compute its primitive invariants.
    """
    # Create a cylinder with dimensions similar to a typical tire
    cylinder = trimesh.creation.cylinder(radius=0.3, height=0.2, sections=32)
    return compute_primitive_invariants(cylinder.vertices)

def load_and_preprocess_mesh(mesh_file):
    """
    Load a mesh from a file and preprocess it by removing unreferenced vertices.
    """
    mesh = trimesh.load(mesh_file, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a Trimesh. Got type: {type(mesh)}")
    mesh.remove_unreferenced_vertices()
    return mesh

def extract_submesh_within_cylinder(mesh, center, radius, height, rotation):
    """
    Extract a submesh within a cylinder defined by center, radius, height, and rotation.
    """
    rotated_mesh = mesh.copy()
    
    # Create a 4x4 transformation matrix from the rotation
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation.as_matrix()
    rotated_mesh.apply_transform(transformation_matrix)

    # Define cylinder aligned with y-axis after rotation
    radial_mask = np.linalg.norm(rotated_mesh.vertices[:, [0, 2]] - center[[0, 2]], axis=1) <= radius
    height_mask = np.abs(rotated_mesh.vertices[:, 1] - center[1]) <= height / 2
    mask = radial_mask & height_mask

    selected_vertices = np.where(mask)[0]
    if selected_vertices.size < 100:
        return None

    # Create a mapping from old vertex indices to new ones
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_vertices)}

    # Filter faces that are entirely within the selected vertices
    face_mask = np.all(np.isin(rotated_mesh.faces, selected_vertices), axis=1)
    filtered_faces = rotated_mesh.faces[face_mask]

    if filtered_faces.size == 0:
        return None

    # Reindex faces
    reindexed_faces = np.vectorize(vertex_map.get)(filtered_faces)

    submesh = trimesh.Trimesh(
        vertices=rotated_mesh.vertices[selected_vertices],
        faces=reindexed_faces,
        process=False
    )
    return submesh

def compare_invariants(target_invariants, submesh):
    """
    Compare the primitive invariants of a submesh with target invariants.
    """
    sub_invariants = compute_primitive_invariants(submesh.vertices)
    error = sum(abs(sub_invariants[key] - target_invariants[key]) for key in target_invariants)
    return error

def generate_grid_points(mesh, step_size=1.0):
    """
    Generate a grid of points within the bounding box of the mesh.
    """
    bbox_min, bbox_max = mesh.bounds
    x = np.arange(bbox_min[0], bbox_max[0], step_size)
    y = np.arange(bbox_min[1], bbox_max[1], step_size)
    z = np.arange(bbox_min[2], bbox_max[2], step_size)
    grid = np.vstack(np.meshgrid(x, y, z)).reshape(3, -1).T
    return grid

def find_best_cylinders(mesh, target_invariants, grid_points, radius, height, error_threshold=1000.0, max_cylinders=10):
    """
    Find the best matching cylinders within the mesh based on primitive invariants.
    """
    detected = []
    search_mesh = mesh.copy()

    def process_point(center):
        for angle in [0]:
            rotation = R.from_euler('y', angle, degrees=True)
            sub = extract_submesh_within_cylinder(search_mesh, center, radius, height, rotation)
            if sub is not None and len(sub.vertices) >= 100:
                error = compare_invariants(target_invariants, sub)
                if error < error_threshold:
                    return {
                        'mesh': sub,
                        'position': center,
                        'radius': radius,
                        'height': height,
                        'error': error,
                        'rotation': rotation,
                        'is_watertight': sub.is_watertight
                    }
        return None

    # Use joblib's Parallel to process points in parallel
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_point)(pt) for pt in grid_points
    )

    for res in results:
        if res:
            # Check if the detected cylinder is sufficiently far from already detected ones
            if not any(np.linalg.norm(res['position'] - d['position']) < res['radius'] for d in detected):
                detected.append(res)
                if res['is_watertight']:
                    try:
                        # Subtract the detected cylinder from the search mesh to prevent overlapping detections
                        search_mesh = search_mesh.difference(res['mesh'], engine='scad')
                        logging.info(f"Subtracted detected cylinder at {res['position']} from search mesh.")
                    except Exception as e:
                        logging.error(f"Skipping subtraction due to error: {e}")
                else:
                    # Since we are now including non-watertight cylinders, you might want to handle them differently
                    logging.info(f"Detected non-watertight cylinder at {res['position']} with error {res['error']}.")
                
                if search_mesh.is_empty:
                    logging.info("Search mesh is empty. Stopping detection.")
                    break

                # Check if we've reached the maximum number of cylinders
                if len(detected) >= max_cylinders:
                    logging.info(f"Reached maximum limit of {max_cylinders} cylinders.")
                    break

    return detected

def combine_detected_cylinders(detected_cylinders):
    """
    Combine all detected cylinder meshes into a single mesh.
    """
    combined_mesh = None
    watertight_count = 0

    for cyl in detected_cylinders:
        if not cyl['is_watertight']:
            logging.warning(f"Skipping non-watertight cylinder at {cyl['position']}.")
            continue

        if combined_mesh is None:
            combined_mesh = cyl['mesh']
            watertight_count += 1
            logging.debug(f"Initializing combined mesh with cylinder at {cyl['position']}.")
        else:
            try:
                combined_mesh = combined_mesh.union(cyl['mesh'], engine='scad')
                watertight_count += 1
                logging.debug(f"Unioned cylinder at {cyl['position']} with combined mesh.")
            except Exception as e:
                logging.error(f"Failed to union cylinder at {cyl['position']}: {e}")
    
    if watertight_count == 0:
        logging.warning("No watertight cylinders to combine.")
        return None

    logging.info(f"Combined {watertight_count} watertight cylinders into a single mesh.")
    return combined_mesh

def visualize_cylinders(original, detected, include_non_watertight=True):
    """
    Visualize the original mesh and detected cylinders using PyVista.
    """
    pv_mesh = pv.wrap(original)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color='lightgrey', opacity=0.5, label='Original Mesh')
    
    # Define color schemes
    watertight_color = 'green'
    non_watertight_color = 'red'
    
    for idx, cyl in enumerate(detected):
        pv_cyl = pv.wrap(cyl['mesh'])
        if cyl['is_watertight']:
            color = watertight_color
            label = f'Watertight Cylinder {idx+1}'
        else:
            color = non_watertight_color
            label = f'Non-Watertight Cylinder {idx+1}'
        
        # Optionally skip non-watertight cylinders based on the flag
        if not cyl['is_watertight'] and not include_non_watertight:
            continue
        
        plotter.add_mesh(pv_cyl, color=color, opacity=0.7, label=label)
    
    plotter.add_legend()
    plotter.show()

def main():
    """
    Main function to execute the cylinder detection and visualization.
    """
    try:
        logging.info("Starting cylinder detection process.")
        target_invariants = get_reference_cylinder_primitive_invariants()
        mesh = load_and_preprocess_mesh("vehicle.obj")
        logging.info("Loaded and preprocessed the mesh.")
        
        radius = 0.5
        height = 0.2
        step_size = 0.5

        grid_points = generate_grid_points(mesh, step_size=step_size)
        logging.info(f"Generated {len(grid_points)} grid points for searching.")

        # Lowered error_threshold for stricter matching and set max_cylinders
        detected_cylinders = find_best_cylinders(
            mesh,
            target_invariants,
            grid_points,
            radius,
            height,
            error_threshold=100.0,  
            max_cylinders=10
        )

        if detected_cylinders:
            watertight_cylinders = [cyl for cyl in detected_cylinders if cyl['is_watertight']]
            non_watertight_cylinders = [cyl for cyl in detected_cylinders if not cyl['is_watertight']]
            logging.info(f"Detected {len(watertight_cylinders)} watertight cylinders and {len(non_watertight_cylinders)} non-watertight cylinders.")
            
            # Optionally combine only watertight cylinders
            combined_mesh = combine_detected_cylinders(watertight_cylinders)
            if combined_mesh:
                combined_mesh.export("combined_cylinders.obj")
                logging.info("Combined watertight cylinders mesh saved as 'combined_cylinders.obj'.")
            
            # Visualize all detected cylinders, including non-watertight ones
            visualize_cylinders(mesh, detected_cylinders, include_non_watertight=True)
        else:
            logging.info("No cylinders detected.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()