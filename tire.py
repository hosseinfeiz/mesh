import numpy as np
import trimesh
from scipy.optimize import differential_evolution
import pyvista as pv
import warnings

warnings.filterwarnings('ignore')  # Suppress trimesh warnings

def compute_primitive_invariants(vertices):
    """
    Compute rotation, translation, and scale-invariant primitive moments.
    """
    # Center vertices for translation invariance
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid

    # Compute second and third order moments
    m2 = np.sum(centered**2, axis=0)
    m3 = np.sum(centered**3, axis=0)

    # Define invariants
    I1 = m2[0] + m2[1] + m2[2]
    I2 = m2[0]*m2[1] + m2[1]*m2[2] + m2[0]*m2[2]
    I3 = m2[0]*m2[1]*m2[2] - np.sum(centered[:, 0]*centered[:, 1]*centered[:, 2])
    I4 = np.sum(m3)
    I5 = m2[0]*m3[1] - m2[1]*m3[2] + m2[2]*m3[0]
    I6 = np.sum(m3**2)
    I7 = m2[0]*(m3[1] + m3[2]) - m3[0]*m2[1]

    # Normalize invariants to make them scale-invariant
    invariants = {
        'I1': 1.0,  # Normalized to 1
        'I2': I2 / (I1**2),
        'I3': I3 / (I1**3),
        'I4': I4 / (I1**1.5),
        'I5': I5 / (I1**2.0),
        'I6': I6 / (I1**3.0),
        'I7': I7 / (I1**2.5)
    }

    return invariants

def get_reference_invariants():
    """
    Generate reference cylinder invariants scaled to match the normalized mesh.
    """
    # Create a cylinder with radius=0.5 and height=1.0 to match the normalized mesh
    cylinder = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=32)
    invariants = compute_primitive_invariants(cylinder.vertices)

    # Add some tolerance for matching (10% of the invariant values)
    return {k: {'mean': v, 'std': abs(v * 0.1)} for k, v in invariants.items()}

def extract_cylinder_submesh(mesh, params):
    """
    Extract submesh within a cylinder defined by parameters.
    Parameters:
        params: [x, y, z, radius, height]
    """
    x, y, z, radius, height = params
    center = np.array([x, y, z])

    # Transform vertices to cylinder space
    local_verts = mesh.vertices - center

    # Check which vertices are within cylinder bounds
    r_squared = local_verts[:, 0]**2 + local_verts[:, 2]**2  # XZ plane radius
    in_radius = r_squared <= radius**2
    in_height = (local_verts[:, 1] >= -height/2) & (local_verts[:, 1] <= height/2)
    vertex_mask = in_radius & in_height

    if not np.any(vertex_mask):
        return None

    # Get faces where all vertices are inside cylinder
    vertex_indices = np.where(vertex_mask)[0]
    face_mask = np.all(np.isin(mesh.faces, vertex_indices), axis=1)

    if not np.any(face_mask):
        return None

    # Create submesh
    faces = mesh.faces[face_mask]
    vertices = mesh.vertices[vertex_mask]

    # Reindex faces
    index_map = {old: new for new, old in enumerate(vertex_indices)}
    try:
        new_faces = np.array([[index_map[vid] for vid in face] for face in faces])
    except KeyError:
        return None

    try:
        submesh = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=False)
        if not submesh.is_watertight:
            submesh = submesh.convex_hull
        return submesh
    except:
        return None

def optimization_loss(params, mesh, target_invariants):
    """
    Compute loss between submesh and target invariants.
    """
    submesh = extract_cylinder_submesh(mesh, params)
    if submesh is None or len(submesh.vertices) < 20:
        return 1e6

    current = compute_primitive_invariants(submesh.vertices)

    # Compute normalized differences
    loss = sum(
        abs(current[k] - target_invariants[k]['mean']) / target_invariants[k]['std']
        for k in current
    )

    # Penalize very small or very large submeshes
    volume_penalty = abs(1 - len(submesh.vertices) / 500)
    return loss + volume_penalty

def detect_cylinders(mesh, max_cylinders=5, error_threshold=10.0):
    """
    Find multiple cylindrical submeshes in the input mesh.
    """
    # Normalize mesh
    mesh.vertices -= mesh.centroid
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= scale

    # Set up optimization bounds
    bounds_x = (mesh.bounds[0][0], mesh.bounds[1][0])
    bounds_y = (mesh.bounds[0][1], mesh.bounds[1][1])
    bounds_z = (mesh.bounds[0][2], mesh.bounds[1][2])
    radius_range = (0.05, 0.3)
    height_range = (0.1, 0.6)

    bounds = [
        bounds_x,  # x
        bounds_y,  # y
        bounds_z,  # z
        radius_range,  # radius
        height_range   # height
    ]

    target_invariants = get_reference_invariants()
    detected = []
    remaining_mesh = mesh.copy()

    for i in range(max_cylinders):
        print(f"Searching for cylinder {i+1}...")

        result = differential_evolution(
            optimization_loss,
            bounds,
            args=(remaining_mesh, target_invariants),
            maxiter=200,       # Increased iterations for better convergence
            popsize=20,        # Increased population size
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            workers=-1         # Utilize all available CPU cores
        )

        if result.fun > error_threshold:
            print(f"No more cylinders found (error: {result.fun:.2f})")
            break

        submesh = extract_cylinder_submesh(remaining_mesh, result.x)
        if submesh is None:
            print("Optimization found a solution, but failed to extract submesh.")
            continue

        detected.append({
            'mesh': submesh,
            'parameters': result.x,
            'error': result.fun
        })
        print(f"Found cylinder {i+1}:")
        print(f"  Center = ({result.x[0]:.2f}, {result.x[1]:.2f}, {result.x[2]:.2f})")
        print(f"  Radius = {result.x[3]:.2f}")
        print(f"  Height = {result.x[4]:.2f}")
        print(f"  Error  = {result.fun:.2f}\n")

        # Remove detected region from search space
        try:
            remaining_mesh = remaining_mesh.difference(submesh, engine='scad')
            if remaining_mesh.is_empty:
                print("All cylinders have been detected.")
                break
        except Exception as e:
            print(f"Error removing detected cylinder from mesh: {e}")
            break

    return detected

def visualize_results(original_mesh, cylinders):
    """
    Visualize original mesh and detected cylinders.
    """
    plotter = pv.Plotter()

    # Show original mesh semi-transparent
    plotter.add_mesh(
        pv.wrap(original_mesh), 
        color='lightgray',
        opacity=0.3,
        label='Original Mesh'
    )

    # Define a list of distinct colors for cylinders
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange']

    # Show detected cylinders
    for i, cyl in enumerate(cylinders):
        color = colors[i % len(colors)]
        plotter.add_mesh(
            pv.wrap(cyl['mesh']),
            color=color,
            label=f'Cylinder {i+1}',
            opacity=0.8
        )

    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

def main():
    # Load mesh
    mesh_file = "vehicle.obj"  # Replace with your mesh file path
    try:
        mesh = trimesh.load(mesh_file)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()  # Combine multiple meshes if necessary
    except Exception as e:
        print(f"Error loading mesh '{mesh_file}': {e}")
        return

    print(f"Loaded mesh '{mesh_file}' with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.\n")

    # Find cylindrical regions
    cylinders = detect_cylinders(
        mesh,
        max_cylinders=5,
        error_threshold=10.0
    )

    if not cylinders:
        print("No cylindrical regions found.")
        return

    print(f"Detected {len(cylinders)} cylinder(s).")

    # Optionally, print out the invariants for debugging
    print("\nReference Invariants:")
    reference_invariants = get_reference_invariants()
    for k, v in reference_invariants.items():
        print(f"  {k}: mean = {v['mean']:.4f}, std = {v['std']:.4f}")
    print("\nDetected Cylinder Invariants:")
    for i, cyl in enumerate(cylinders, start=1):
        cyl_invariants = compute_primitive_invariants(cyl['mesh'].vertices)
        print(f"  Cylinder {i}:")
        for k, v in cyl_invariants.items():
            print(f"    {k}: {v:.4f}")
        print()

    # Visualize results
    visualize_results(mesh, cylinders)

if __name__ == "__main__":
    main()