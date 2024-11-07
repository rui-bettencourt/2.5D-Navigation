#!/usr/bin/env python
import rospy
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

import time
from copy import deepcopy
from mesh_nav_ros.robot_mesh_state import RobotMeshState
from collision_detections import *
# from joints_explorer import JointExploration
# from octomap_functions import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header

class Debug(object):
    def __init__(self):

        rospy.init_node('debug_mesh', anonymous=True)
        self.meshes = RobotMeshState()
        
        self.cfgs_position = np.array([[0.0,0.0,0.0],
                                       [0.5,0.0,0.0],
                                        [1.0,0.0,0.0],
                                        [1.75,0.0,0.0],
                                        [1.75,-0.5,0.0],
                                        [1.75,-1.0,0.0],
                                        [1.75,-1.25,0.0],
                                        [1.0,-1.25,0.0],
                                        [0.5,-1.25,0.0],
                                        [0.1,-1.25,0.0],
                                        [0.1,-1.85,0.0],
                                        [0.1,-1.85,0.0],
                                        [0.1,-1.85,0.0]])

        self.cfgs_orientation = np.array([[0.0,0.0,0.0],
                                        [0.0,0.0,0.0],
                                        [0.0,0.0,0.0],
                                        [0.0,0.0,0.0],
                                        [ 0, 0, -90],
                                        [ 0, 0, -90],
                                        [ 0, 0, -90],
                                        [ 0, 0, -180],
                                        [ 0, 0, -180],
                                        [ 0, 0, -180],
                                        [ 0, 0, -90],
                                        [ 0, 0, -90],
                                        [ 0, 0, -90]])

        self.cfgs_arm = ['arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_open.ply',
                         'arm_half_open.ply',
                         'arm_closed.ply']


    def run_debug(self):
        rate = rospy.Rate(10)
        path = '/home/rui/ds/testbed/'

        # import obstacle mesh
        obstacle_mesh = o3d.io.read_triangle_mesh(path + 'obstacles.ply')
        print("Obstacle mesh: Number of vertices: " + str(len(obstacle_mesh.vertices)) + "; Number of triangles: " +str((obstacle_mesh.triangles)))

        # import octomap
        obstacle_octomap = o3d.io.read_point_cloud(path+'obstacles_octomap.pcd')

        while not rospy.is_shutdown():

            if self.meshes.robot_mesh is not None:
                # # Check intersection with the environment mesh
                # robot_aabb_tree = o3d.geometry.KDTreeFlann(self.meshes.robot_mesh)
                # env_aabb_tree = o3d.geometry.KDTreeFlann(obstacle_mesh)

                
                # intersection = robot_aabb_tree.compute_point_cloud_distance(env_aabb_tree) < 1e-5
                # print("computed")
                # if any(intersection):
                #     rospy.loginfo("Intersection detected!")
                # else:
                #     rospy.loginfo("No intersection detected.")

                # self.meshes.robot_mesh = self.meshes.robot_mesh.simplify_vertex_clustering(0.02)
                # self.meshes.robot_mesh = self.meshes.robot_mesh.simplify_quadric_decimation(2000)

                # obstacle_mesh = obstacle_mesh.remove_unreferenced_vertices()
                # obstacle_mesh = obstacle_mesh.
                for cfg_id in range(len(self.cfgs_position)):
                    # robot_mesh = deepcopy(self.meshes.robot_mesh)
                    robot_mesh = o3d.io.read_triangle_mesh(path + self.cfgs_arm[cfg_id])
                    points = robot_mesh.sample_points_uniformly(number_of_points=1000, use_triangle_normal=False)
                    o3d.visualization.draw_geometries([robot_mesh])
                    o3d.visualization.draw_geometries([points])
                    


                    # translate robot TODO: make a config list with different robot poses and check one by one
                    print("cfg" + str(cfg_id) + " :")
                    robot_mesh.rotate(R.from_rotvec(self.cfgs_orientation[cfg_id],degrees=True).as_matrix())
                    robot_mesh.translate(np.array(self.cfgs_position[cfg_id]))
                    

                    bb = robot_mesh.get_axis_aligned_bounding_box()
                    print(bb.volume())
                    bb.scale(2.0,bb.get_center())
                    print(bb.volume())
                    bb.color = [0,1,0]

                    # simplify obstacle mesh
                    obstacle_mesh_cropped = obstacle_mesh.crop(bb)
                    obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
                    obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)
                    obstacle_mesh_cropped.paint_uniform_color(np.array([0.5, 0.5, 0.0]))
                    print("Obstacle mesh: Number of vertices: " + str(len(obstacle_mesh_cropped.vertices)) + "; Number of triangles: " +str((obstacle_mesh_cropped.triangles)))

                    # simplify octomap
                    print("Obstacle octomap: Number of points: " + str(len(obstacle_octomap.points)))
                    obstacle_octomap_cropped = obstacle_octomap.crop(bb)
                    obstacle_octomap_cropped.paint_uniform_color(np.array([0, 0, 0.5]))
                    print("Obstacle octomap: Number of points after crop: " + str(len(obstacle_octomap_cropped.points)))


                    # simplify robot mesh
                    robot_mesh = robot_mesh.simplify_vertex_clustering(0.02)
                    r_point_cloud = o3d.geometry.PointCloud()
                    r_point_cloud.points = robot_mesh.vertices
                    robot_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=r_point_cloud, alpha=0.4)
                    print("Number of vertices: " + str(len(robot_mesh.vertices)) + "; Number of triangles: " +str((robot_mesh.triangles)))

                    # TODO: import here and check collision with octomap
                    # o3d.io.write_triangle_mesh(path+'arm_home.ply', robot_mesh)
                    # exit()

                    # check intersections
                    is_colliding_o3d(robot_mesh, obstacle_mesh_cropped)
                    if len(obstacle_mesh_cropped.triangles) > 0:
                        is_colliding_fcl(robot_mesh, obstacle_mesh_cropped)
                    else:
                        print('fcl - No colission because no env meshes')
                    is_colliding_fcl_octomap(robot_mesh, obstacle_octomap_cropped)

                    # r_point_cloud = o3d.geometry.PointCloud()
                    # r_point_cloud.points = robot_mesh.vertices
                    # o_point_cloud = o3d.geometry.PointCloud()
                    # o_point_cloud.points = obstacle_mesh_cropped.vertices
                    robot_mesh.compute_vertex_normals()
                    o3d.visualization.draw_geometries([robot_mesh, bb, obstacle_mesh, obstacle_mesh_cropped, obstacle_octomap_cropped])
                # o3d.visualization.draw_geometries([obstacle_mesh])
            rate.sleep()
    def plan_arm(self):
        rate = rospy.Rate(10)
        path = '/home/rui/ds/testbed/'
        self.je = JointExploration(bounds = [(0.0,0.31) for _ in range(1)])
        self.torso_controller = rospy.Publisher('/torso_controller/command',JointTrajectory, queue_size=1)


        while not rospy.is_shutdown():

            if self.meshes.robot_mesh is not None:
                robot_mesh = deepcopy(self.meshes.robot_mesh)
                arm_joints_names = [key for key, value in self.meshes.robot_joint_positions.items() if 'torso' in key]
                arm_joints = [value for key, value in self.meshes.robot_joint_positions.items() if 'torso' in key]
                print(arm_joints)
            # arm_joints = [0.00013214900333036184, -0.07636866473887594, -0.0015072540593443762, 0.05017026552519521, 0.0008770917969416203, 0.0019384540579903131, 0.0016621630988895575]
                plan = self.je.plan(arm_joints, [0.2])#[0.2, -0.1, -0.1, 0.05, 0.0, 0.01, 0.01])
                trajectory = JointTrajectory()
                trajectory.joint_names = arm_joints_names
                trajectory.header.stamp = rospy.Time.now()
                
                for i, pos in enumerate(plan):
                    point = JointTrajectoryPoint()
                    point.positions = pos
                    point.time_from_start = rospy.Duration(i * 0.1)  # 0.1 second intervals
                    trajectory.points.append(point)

                self.torso_controller.publish(trajectory)
                rospy.sleep(10)
                # for pos in plan:
                #     trajectory.points.append(JointTrajectoryPoint(positions=pos))
                # for i in range(3):
                #     trajectory.header.seq = 100+i
                #     self.torso_controller.publish(trajectory)
                # rospy.sleep(10)

            rate.sleep()

if __name__ == '__main__':
    mc = Debug()
    # mc.run_debug()
    mc.plan_arm()