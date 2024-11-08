import numpy as np
from scipy.spatial import distance
from scipy.stats import moment
import trimesh
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

class DeformableShapeDescriptor:
    def __init__(self, mesh):
        """
        Initialize shape descriptor for deformable meshes.
        
        Args:
            mesh (trimesh.Trimesh): Input mesh
        """
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.geodesic_matrix = None
        self.local_frames = None
        
    def compute_geodesic_distances(self):
        """
        Compute geodesic distances between vertices using shortest paths.
        """
        # Create adjacency matrix from edges
        edges = self.mesh.edges
        n_vertices = len(self.vertices)
        
        # Create sparse adjacency matrix with edge weights
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        edge_lengths = np.linalg.norm(
            self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], 
            axis=1
        )
        weights = np.concatenate([edge_lengths, edge_lengths])
        
        adj_matrix = csr_matrix(
            (weights, (rows, cols)), 
            shape=(n_vertices, n_vertices)
        )
        
        # Compute all-pairs shortest paths
        self.geodesic_matrix = shortest_path(
            adj_matrix, 
            method='auto', 
            directed=False
        )
        
        return self.geodesic_matrix
    
    def compute_local_reference_frames(self):
        """
        Compute local reference frames for each vertex based on neighborhood structure.
        """
        n_vertices = len(self.vertices)
        self.local_frames = np.zeros((n_vertices, 3, 3))
        
        for i in range(n_vertices):
            # Get 1-ring neighbors
            neighbor_indices = self.mesh.vertex_neighbors[i]
            if len(neighbor_indices) < 3:
                continue
                
            # Get neighbor coordinates relative to current vertex
            neighbors = self.vertices[neighbor_indices] - self.vertices[i]
            
            # Compute PCA of neighborhood
            pca = PCA(n_components=3)
            pca.fit(neighbors)
            
            # Use PCA axes as local frame
            self.local_frames[i] = pca.components_
            
            # Ensure right-handed coordinate system
            self.local_frames[i, 2] = np.cross(
                self.local_frames[i, 0], 
                self.local_frames[i, 1]
            )
            
        return self.local_frames
    
    def compute_bending_invariants(self, n_samples=1000):
        """
        Compute bending-invariant shape descriptors using geodesic distances.
        
        Args:
            n_samples (int): Number of vertex pairs to sample
        
        Returns:
            dict: Bending invariant features
        """
        if self.geodesic_matrix is None:
            self.compute_geodesic_distances()
            
        n_vertices = len(self.vertices)
        
        # Randomly sample vertex pairs
        pairs = np.random.choice(n_vertices, size=(n_samples, 2))
        
        # Compute ratios of Euclidean to geodesic distances
        euclidean_distances = np.array([
            np.linalg.norm(self.vertices[i] - self.vertices[j])
            for i, j in pairs
        ])
        geodesic_distances = np.array([
            self.geodesic_matrix[i, j]
            for i, j in pairs
        ])
        
        distance_ratios = euclidean_distances / geodesic_distances
        
        # Compute statistical moments of the ratios
        invariants = {
            'mean_ratio': np.mean(distance_ratios),
            'std_ratio': np.std(distance_ratios),
            'skew_ratio': moment(distance_ratios, moment=3),
            'kurt_ratio': moment(distance_ratios, moment=4)
        }
        
        return invariants
    
    def compute_twisting_invariants(self, n_rings=2):
        """
        Compute twist-invariant shape descriptors using local reference frames.
        
        Args:
            n_rings (int): Number of rings to consider for local neighborhood
        
        Returns:
            dict: Twisting invariant features
        """
        if self.local_frames is None:
            self.compute_local_reference_frames()
            
        n_vertices = len(self.vertices)
        twist_angles = []
        
        for i in range(n_vertices):
            # Get n-ring neighbors
            neighbors = set([i])
            current_ring = set([i])
            
            for _ in range(n_rings):
                next_ring = set()
                for v in current_ring:
                    next_ring.update(self.mesh.vertex_neighbors[v])
                current_ring = next_ring - neighbors
                neighbors.update(current_ring)
            
            neighbors = list(neighbors - {i})
            if len(neighbors) < 3:
                continue
            
            # Compute relative rotations between local frames
            for j in neighbors:
                # Get rotation between frames
                rotation = np.dot(
                    self.local_frames[j], 
                    self.local_frames[i].T
                )
                
                # Extract twist angle (rotation around normal)
                twist_angle = np.arctan2(rotation[1, 0], rotation[0, 0])
                twist_angles.append(np.abs(twist_angle))
        
        twist_angles = np.array(twist_angles)
        
        # Compute statistical moments of twist angles
        invariants = {
            'mean_twist': np.mean(twist_angles),
            'std_twist': np.std(twist_angles),
            'max_twist': np.max(twist_angles),
            'histogram': np.histogram(twist_angles, bins=10)[0]
        }
        
        return invariants
    
    def compute_curvature_invariants(self):
        """
        Compute curvature-based invariants that are robust to deformations.
        
        Returns:
            dict: Curvature invariant features
        """
        # Compute gaussian and mean curvatures
        gaussian_curvature = self.mesh.gaussian_curvature
        mean_curvature = self.mesh.mean_curvature
        
        # Compute curvature ratios and derivatives
        curvature_ratio = np.abs(gaussian_curvature) / (np.abs(mean_curvature) + 1e-6)
        
        invariants = {
            'mean_gaussian': np.mean(np.abs(gaussian_curvature)),
            'std_gaussian': np.std(gaussian_curvature),
            'mean_ratio': np.mean(curvature_ratio),
            'std_ratio': np.std(curvature_ratio),
            'histogram_gaussian': np.histogram(gaussian_curvature, bins=10)[0],
            'histogram_mean': np.histogram(mean_curvature, bins=10)[0]
        }
        
        return invariants
    
    def compute_all_invariants(self):
        """
        Compute all deformation-invariant descriptors.
        
        Returns:
            dict: Combined invariant features
        """
        bending = self.compute_bending_invariants()
        twisting = self.compute_twisting_invariants()
        curvature = self.compute_curvature_invariants()
        
        # Combine all invariants
        all_invariants = {
            'bending': bending,
            'twisting': twisting,
            'curvature': curvature
        }
        
        return all_invariants

def compare_meshes(mesh1, mesh2, descriptor_class=DeformableShapeDescriptor):
    """
    Compare two meshes using deformation-invariant descriptors.
    
    Args:
        mesh1, mesh2 (trimesh.Trimesh): Meshes to compare
        descriptor_class: Class to compute shape descriptors
        
    Returns:
        float: Similarity score (lower means more similar)
    """
    # Compute descriptors for both meshes
    desc1 = descriptor_class(mesh1)
    desc2 = descriptor_class(mesh2)
    
    inv1 = desc1.compute_all_invariants()
    inv2 = desc2.compute_all_invariants()
    
    # Compare each type of invariant
    differences = []
    
    # Compare bending invariants
    for key in ['mean_ratio', 'std_ratio']:
        diff = abs(inv1['bending'][key] - inv2['bending'][key])
        differences.append(diff)
    
    # Compare twisting invariants
    for key in ['mean_twist', 'std_twist']:
        diff = abs(inv1['twisting'][key] - inv2['twisting'][key])
        differences.append(diff)
    
    # Compare curvature invariants
    for key in ['mean_gaussian', 'std_gaussian', 'mean_ratio']:
        diff = abs(inv1['curvature'][key] - inv2['curvature'][key])
        differences.append(diff)
    
    # Compare histograms using Earth Mover's Distance
    for desc in ['twisting', 'curvature']:
        if 'histogram' in inv1[desc]:
            hist1 = inv1[desc]['histogram'].astype(float)
            hist2 = inv2[desc]['histogram'].astype(float)
            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-6)
            hist2 = hist2 / (np.sum(hist2) + 1e-6)
            diff = np.sum(np.abs(hist1 - hist2))
            differences.append(diff)
    
    # Compute weighted average of differences
    similarity = np.mean(differences)
    
    return similarity

# Example usage
def main():
    # Load a mesh and create a deformed version
    mesh = trimesh.load('model.obj')
    
    # Compute invariants
    descriptor = DeformableShapeDescriptor(mesh)
    invariants = descriptor.compute_all_invariants()
    
    print("Computed invariants:")
    for category, features in invariants.items():
        print(f"\n{category.upper()} invariants:")
        for name, value in features.items():
            if isinstance(value, np.ndarray):
                print(f"  {name}: array of shape {value.shape}")
            else:
                print(f"  {name}: {value}")

if __name__ == "__main__":
    main()