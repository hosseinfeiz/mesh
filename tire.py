import numpy as np
import trimesh
import pyvista as pv
from joblib import Parallel, delayed
def compute_primitive_invariants(vertices):
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    moment2 = np.sum(centered**2, axis=0)
    moment3 = np.sum(centered**3, axis=0)
    return {
        "IP1": moment2.sum(),
        "IP2": moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2],
        "IP3": moment2[0] * moment2[1] * moment2[2] - np.sum(centered[:, 0] * centered[:, 1] * centered[:, 2]),
        "IP4": (moment3[0] + moment3[1]) * (moment2[0] * moment2[1] + moment2[1] * moment2[2] + moment2[0] * moment2[2]),
        "IP5": moment2[0] * moment3[1] - moment2[1] * moment3[2] + moment2[2] * moment3[0],
        "IP6": np.sum(moment3**2),
        "IP7": moment2[0] * (moment3[1] + moment3[2]) - moment3[0] * moment2[1],
    }
def get_reference_cylinder_primitive_invariants():
    cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
    return compute_primitive_invariants(cylinder.vertices)
def load_and_preprocess_mesh(mesh_file):
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh.remove_unreferenced_vertices()
    return mesh
def extract_submesh_within_cylinder(mesh, center, radius, height):
    radial_mask = np.linalg.norm(mesh.vertices[:, [0, 2]] - center[[0, 2]], axis=1) <= radius
    height_mask = np.abs(mesh.vertices[:, 1] - center[1]) <= height / 2
    mask = radial_mask & height_mask
    selected_vertices = np.where(mask)[0]
    if selected_vertices.size < 3:
        return None
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_vertices)}
    face_mask = np.all(np.isin(mesh.faces, selected_vertices), axis=1)
    filtered_faces = mesh.faces[face_mask]
    if filtered_faces.size == 0:
        return None
    reindexed_faces = np.vectorize(vertex_map.get)(filtered_faces)
    submesh = trimesh.Trimesh(vertices=mesh.vertices[selected_vertices], faces=reindexed_faces, process=False)
    return submesh
def compare_invariants(target_invariants, submesh):
    sub_invariants = compute_primitive_invariants(submesh.vertices)
    error = sum(abs(sub_invariants[key] - target_invariants[key]) for key in target_invariants)
    return error
def generate_grid_points(mesh, step_size=1.0):
    bbox_min, bbox_max = mesh.bounds
    x = np.arange(bbox_min[0], bbox_max[0], step_size)
    y = np.arange(bbox_min[1], bbox_max[1], step_size)
    z = np.arange(bbox_min[2], bbox_max[2], step_size)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return grid
def find_best_cylinder_grid_search(mesh, target_invariants, grid_points, radius, height, error_threshold=1000.0):
    detected = []
    search_mesh = mesh.copy()
    def process_point(center):
        sub = extract_submesh_within_cylinder(search_mesh, center, radius, height)
        if sub is not None and len(sub.vertices) >= 3:
            error = compare_invariants(target_invariants, sub)
            if error < error_threshold:
                return {
                    'mesh': sub,
                    'position': center,
                    'radius': radius,
                    'height': height,
                    'error': error
                }
        return None
    results = Parallel(n_jobs=-1)(delayed(process_point)(pt) for pt in grid_points)
    for res in results:
        if res and not any(np.linalg.norm(np.array(res['position']) - np.array(d['position'])) < res['radius'] for d in detected):
            detected.append(res)
            # Check if the detected submesh is watertight before subtracting
            if res['mesh'].is_watertight:
                try:
                    search_mesh = search_mesh.difference(res['mesh'], engine='scad')
                except Exception as e:
                    print(f"Skipping subtraction due to error: {e}")
            else:
                print("Skipping non-watertight mesh for subtraction")
            if search_mesh.is_empty:
                break
    return detected[:5]
def visualize_cylinders(original, detected):
    pv_mesh = pv.wrap(original)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color='lightgrey', opacity=0.5, label='Original Mesh')
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for idx, cyl in enumerate(detected):
        pv_cyl = pv.wrap(cyl['mesh'])
        plotter.add_mesh(pv_cyl, color=colors[idx % len(colors)], opacity=0.7, label=f'Cylinder {idx+1}')
    plotter.add_legend()
    plotter.show()
def main():
    target_invariants = get_reference_cylinder_primitive_invariants()
    mesh = load_and_preprocess_mesh("vehicle.obj")
    bbox_min, bbox_max = mesh.bounds
    radius = 0.5
    height = 1.0
    grid_points = generate_grid_points(mesh, step_size=1.0)
    detected_cylinders = find_best_cylinder_grid_search(
        mesh, target_invariants, grid_points, radius, height, error_threshold=65000.0
    )
    if detected_cylinders:
        visualize_cylinders(mesh, detected_cylinders)
    else:
        print("No cylinders detected.")
if __name__ == "__main__":
    main()