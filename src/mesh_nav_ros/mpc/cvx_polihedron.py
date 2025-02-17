import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay
import time

def is_valid_expansion(new_vertices, obstacle_cloud):
    """Check if expanded polyhedron remains convex and obstacle-free."""
    try:
        hull = ConvexHull(new_vertices)
        expanded_polyhedron = new_vertices[hull.vertices]
        
        # Check for collisions
        # obstacle_tree = o3d.geometry.KDTreeFlann(obstacle_cloud)
        # for vertex in expanded_polyhedron:
        #     [_, idx, _] = obstacle_tree.search_knn_vector_3d(vertex, 1)
        #     if np.linalg.norm(np.asarray(obstacle_cloud.points)[idx[0]] - vertex) < 0.05:
        #         return False
        
        # Ensure no obstacle points are inside the polyhedron
        delaunay = Delaunay(expanded_polyhedron)
        for obstacle in np.asarray(obstacle_cloud.points):
            if delaunay.find_simplex(obstacle) >= 0:
                return False
        return True
    except:
        return False  # If ConvexHull fails, it's non-convex

def expand_polyhedron(vertices, obstacle_cloud, step_size=0.1, max_iters=50):
    """Expand a convex polyhedron while maintaining convexity and avoiding obstacles."""
    for _ in range(max_iters):
        new_vertices = vertices.copy()
        for i in range(len(vertices)):
            temp_vertices = vertices.copy()
            temp_vertices[i] += step_size * (vertices[i] - np.mean(vertices, axis=0))
            if is_valid_expansion(temp_vertices, obstacle_cloud):
                new_vertices[i] = temp_vertices[i]
        vertices = new_vertices
    return vertices

def main():
    time_start = time.time()
    # Define robot's starting position
    robot_pose = np.array([0.5, 0.5, 0.5])
    number_of_points = 30
    
    # Create initial polyhedron (icosphere approximation around robot)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(robot_pose)
    vertices = np.asarray(sphere.sample_points_uniformly(number_of_points).points)
    
    # Generate obstacle point cloud (example: random obstacles)
    obstacle_cloud = o3d.io.read_point_cloud('/home/rui/ds/testbed/obstacles_octomap.pcd')
    
    # Expand polyhedron
    expanded_polyhedron = expand_polyhedron(vertices, obstacle_cloud)
    print("time was ", time.time()-time_start)
    # Construct final convex hull
    hull = ConvexHull(expanded_polyhedron)
    final_mesh = o3d.geometry.TriangleMesh()
    final_mesh.vertices = o3d.utility.Vector3dVector(expanded_polyhedron)
    final_mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)
    final_mesh.compute_vertex_normals()
    
    # Visualize result
    o3d.visualization.draw_geometries([final_mesh, obstacle_cloud])

if __name__ == "__main__":
    main()
