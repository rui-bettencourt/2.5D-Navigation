#!/usr/bin/env python
import rospy
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

import time
import random
from copy import deepcopy
from mesh_nav_ros.robot_mesh_state import RobotMeshState
from collision_detections import *
from joints_explorer import PRM
# from octomap_functions import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from graph_functions import GraphManager
from mesh_nav_ros.robot_mesh_state import RobotMeshState


class Debug(object):
    def __init__(self):
        rospy.init_node('debug_mesh', anonymous=True)
        self.path = '/home/rui/ds/testbed/'
        self.Graph = GraphManager(self.path+"traversablegroundgraph.pkl")
        self.MC = RobotMeshState()
        self.prm = PRM(self.Graph)

    def run_debug(self):
        rate = rospy.Rate(10)

        time_start = time.time()
        self.prm.generate(num_samples=1)
        print("Took {} seconds for PRM".format(time.time() - time_start))
        joint_names = ['base_link','torso_lift_joint','arm_1_joint']

        while not rospy.is_shutdown():
            all_bbs = []
            # import obstacle mesh
            path = '/home/rui/ds/testbed/'
            obstacle_mesh = o3d.io.read_triangle_mesh(path + 'obstacles.ply')
            obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67])
            all_bbs.append(obstacle_mesh)

            # # for debug
            # conf_bb_og = list(self.MC.robot_bbs.values())
            # for bb in conf_bb_og:
            #     bb.color=[0,0,0]
            # all_bbs.extend(conf_bb_og)
            # Looping through all nodes and their attributes
            for node, attr in self.prm.prm.nodes(data=True):
                # print("Node: ", node)
                # print(attr['configuration'])
                # print(attr['configuration'])
                conf_bb = self.MC.simulate_move_joints(self.MC.robot_bbs, joint_names[0], attr['configuration'][:3])  # move base
                conf_bb = self.MC.simulate_move_joints(conf_bb, joint_names[1:], attr['configuration'][3:])             # move joints
                
                mesh= self.MC.convert_bbs_to_mesh(conf_bb)

                bb = mesh.get_axis_aligned_bounding_box()
                bb.scale(2.0,bb.get_center())
                # simplify obstacle mesh
                obstacle_mesh_cropped = obstacle_mesh.crop(bb)
                obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
                obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)
                obstacle_mesh_cropped.remove_duplicated_vertices()
                obstacle_mesh_cropped.remove_duplicated_triangles()
                obstacle_mesh_cropped.remove_degenerate_triangles()
                obstacle_mesh_cropped.remove_non_manifold_edges()

                if is_colliding_o3d(mesh, obstacle_mesh_cropped):
                    rand_color = [1.0,0,0]
                else:
                    rand_color = [0,random.uniform(0,1),random.uniform(0,1)]
                
                conf_bb = list(conf_bb.values())
                # rand_color = [0,random.uniform(0,1),random.uniform(0,1)]
                for bb in conf_bb:
                    bb.color=rand_color

                    vol = self.MC.get_volume_collision(bb,obstacle_mesh_cropped)
                    print("volume is: ", vol)
                    if vol > 0:
                        o3d.visualization.draw_geometries(bb + obstacle_mesh_cropped)
                        o3d.visualization.draw_geometries(bb + obstacle_mesh_cropped.crop(bb))

                all_bbs.extend(conf_bb)



            o3d.visualization.draw_geometries(all_bbs)
            exit()

if __name__ == '__main__':
    mc = Debug()
    mc.run_debug()