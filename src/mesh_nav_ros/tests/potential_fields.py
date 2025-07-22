import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

# Load mesh and sample points
path = '/home/rui/ds/testsiros2025/'
mesh = o3d.io.read_triangle_mesh(path+"obstacles.ply")
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=10000)
points = np.asarray(pcd.points)

# Build KD-Tree for nearest-neighbor queries
tree = KDTree(points)

# Define voxel grid parameters
grid_size = 50  # Number of grid points per axis
x_range = y_range = z_range = (-10, 10)  # Adjust based on environment size
X, Y, Z = np.mgrid[x_range[0]:x_range[1]:grid_size*1j,
                   y_range[0]:y_range[1]:grid_size*1j,
                   z_range[0]:z_range[1]:grid_size*1j]
grid_positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Compute distance field
distances, indices = tree.query(grid_positions)
max_influence = .5  # Maximum influence radius of the force field
force_magnitudes = np.exp(-distances / max_influence)  # Decay function

# Reshape for volume rendering
force_volume = force_magnitudes.reshape(grid_size, grid_size, grid_size)

# Extract isosurface using Marching Cubes (shows "inflated" effect)
verts, faces, _, _ = marching_cubes(force_volume, level=0.3)  # Adjust threshold

# Convert to Open3D mesh for visualization
inflated_mesh = o3d.geometry.TriangleMesh()
inflated_mesh.vertices = o3d.utility.Vector3dVector(verts / grid_size * 10 - 5)  # Scale to world space
inflated_mesh.triangles = o3d.utility.Vector3iVector(faces)
inflated_mesh.paint_uniform_color([0.5, 0.1, 0.8])  # Purple color for visualization

# Visualize in Open3D
o3d.visualization.draw_geometries([mesh, inflated_mesh])
