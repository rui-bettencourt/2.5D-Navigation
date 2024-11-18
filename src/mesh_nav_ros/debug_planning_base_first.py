#!/usr/bin/env python
# import ros and other libraries
import rospy
import random
import numpy as np

# import other created files
from robot_mesh_state import RobotMeshState
from collision_detections import *
from graph_functions import GraphManager
from elastic_bands import ElasticBandPlanner
import time

###### variables
k_attraction_base=0.3
k_repulsion_base=0.005
k_attraction_joints=0.2
k_repulsion_joints=0.4
k_safety_joints = 0.0
dynamic_safety = False
closest_obstacle_only = True
k_update_joints=0.2
k_orientation=0.02
k_orientation_from_base=0.0
obstacle_threshold=1.5
min_distance_to_obstacle = 0.05
start = np.asarray([-11.0, -8.0, 0.0, 0.0]) # (x,y,z,yaw)
goal = np.asarray([-4.6, -8.5, 0.0, 0.0])
safe_config = {'joint1': 0.0, 'joint2': np.pi}
center_activation_safety = 0.8
################

class MeshNav(object):
    def __init__(self):
        # begin node
        rospy.init_node('mesh_nav', anonymous=True)

        # variables
        self.path = '/home/rui/ds/testbed/'     # path for files, change this to save in the package
        self.rate = rospy.Rate(10)              # TODO: make this an argument of launch file

        # create classes needed for navigation
        # self.Meshes = RobotMeshState()
        self.Graph = GraphManager(self.path+"traversablegroundgraph.pkl")
        self.RM = RobotMeshState()
        self.EBAND = ElasticBandPlanner(k_attraction_base=k_attraction_base, k_repulsion_base=k_repulsion_base, k_attraction_joints=k_attraction_joints,
                            k_repulsion_joints=k_repulsion_joints, k_update_joints=k_update_joints, k_orientation=k_orientation, k_orientation_from_base=k_orientation_from_base,
                            k_safety_joints=k_safety_joints, obstacle_threshold=obstacle_threshold)

    def run(self):
        obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67]) # pink
        execution_times = []
        number_of_attempts = 100

        while not rospy.is_shutdown() and number_of_attempts>0:
            # test path planning with random start and end
            start_coord = start[:4]  # Replace with your start coordinates
            end_coord = goal[:3]     # Replace with your end coordinates

            path = self.Graph.plan(start_coord, end_coord)
            # self.plot_path_3d(path, obstacle_mesh)
            # check for collisions all points in the path
            
            # if there is collisions use some sort of elastic band to move away from the collision
            start_time = time.time()
            new_path = self.EBAND.update_path(path, obstacle_mesh, self.RM, 500, convergence_threshold=1e-2)
            end_time = time.time()
            print("Execution time:", end_time - start_time)

            self.EBAND.animate_path_evolution(50)
            self.plot_path_3d(new_path, obstacle_mesh)
            # print(self.RM.robot_state.get_pose(), self.RM.robot_state.joints)
            exit()
            execution_times.append(end_time-start_time)
            number_of_attempts -= 1
            self.rate.sleep()
        execution_times = np.asarray(execution_times)
        
        print("Avg time: ",np.mean(execution_times)," | std : ", np.std(execution_times), " | min: ", np.min(execution_times), " | max: ", np.max(execution_times))

    def plot_path_3d(self, path, obstacle_mesh):
        all_bbs = []
        all_bbs.append(obstacle_mesh)
        # then use the code of moving the robot to move to each point in the path
        for pose in path:
            conf_bb = self.RM.simulate_move_joints(self.RM.robot_bbs, 'base_link', pose) # this is just moving the base to the pose. arm config is the online one

            mesh= self.RM.convert_bbs_to_mesh(conf_bb)

            bb = mesh.get_axis_aligned_bounding_box()
            bb.scale(2.0,bb.get_center())
            # simplify obstacle mesh
            obstacle_mesh_cropped = obstacle_mesh.crop(bb)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

            if is_colliding_o3d(mesh, obstacle_mesh_cropped):
                rand_color = [1.0,0,0]
            else:
                rand_color = [0,random.uniform(0,1),random.uniform(0,1)]

            conf_bb = list(conf_bb.values())

            for bb in conf_bb:
                bb.color=rand_color
            all_bbs.extend(conf_bb)

        o3d.visualization.draw_geometries(all_bbs)

    # def initialize_full_body_path(self,path):

if __name__ == '__main__':
    mn = MeshNav()
    mn.run()
