#!/usr/bin/env python
import numpy as np
import open3d as o3d
import octomap

def import_octomap(file_name):
    octree = octomap.OcTree(file_name)
    occupied, empty = octree.extractPointCloud()
    print(type(occupied))
    # Step 2: Convert to Open3D point cloud for processing
    # points = np.array(cloud.points)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    # return point_cloud
    # return octree