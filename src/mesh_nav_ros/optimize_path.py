#!/usr/bin/env python
# import ros and other libraries
import rospy
import random
import numpy as np
import time
import json

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

# import other created files
from robot_mesh_state import RobotMeshState
from collision_detections import *
from graph_functions import GraphManager
from elastic_bands_threads import ElasticBandPlanner as ElasticBandPlannerPy
from robot_kinematics import RobotKinematics
from aux_functions import interpolate_path, sample_path

# NEW: C++ elastic band planner binding (pybind11)
import sys, os
sys.path.append(os.path.join(os.environ["HOME"], "pcl_ws/devel/lib"))  # adjust to where .so is
try:
    from elastic_band_planner_cpp import ElasticBandPlanner as ElasticBandPlannerCPP
    _HAS_CPP_EBAND = True
except Exception as e:
    rospy.logwarn("Could not import C++ ElasticBandPlanner binding: %s", e)
    _HAS_CPP_EBAND = False


###### variables
k_attraction_base = 0.005
k_repulsion_base = 0.00002
k_attraction_joints = 0.2
k_repulsion_joints = 0.01
k_repulsion_robot_joints = 0.005
k_safety_joints = 0.2  # 0.0
k_update_joints = 0.2
k_orientation = 0.02
k_orientation_from_base = 0.0
obstacle_threshold = 1.5
manipulator = True

NUMBER_OF_ATTEMPTS = 100
NUMBER_WAYPOINTS_IN_PATH = 100

# Number of arm joints expected by the binding (q1..q7)
NUM_JOINTS = 7

start = {'x': 0.0, 'y': 0.0, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'q1': 0.0, 'q2': -0.05, 'q3': 0.0, 'q4': 0.02, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}

goal = {'x': 4.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'q1': 0.0, 'q2': -0.05, 'q3': 0.0, 'q4': 0.02, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}

safe_config = {'q1': 0.72, 'q2': -0.9, 'q3': -0.88, 'q4': 1.94, 'q5': -1.2, 'q6': 1.37, 'q7': 0.0}
center_activation_safety = 0.8


class MeshNav(object):
    def __init__(self):
        # begin node
        rospy.init_node('mesh_nav', anonymous=True)

        # variables
        # self.path = '/home/rui/ds/testscilindros/hard/'
        self.path = '/home/rui/ds/testsiros2025/'  # path for files, change this to save in the package
        self.rate = rospy.Rate(10)  # TODO: make this an argument of launch file

        # create classes needed for navigation
        self.robot_kinematics = RobotKinematics(manipulator)
        self.Graph = GraphManager(self.path + "traversablegroundgraph.pkl")
        self.RM = RobotMeshState(robot_kinematics=self.robot_kinematics)

        # Keep the Python planner instance for utilities (e.g., compute_2mesh_distance)
        self.EBAND = ElasticBandPlannerPy(
            robot_kinematics=self.robot_kinematics,
            k_attraction_base=k_attraction_base,
            k_repulsion_base=k_repulsion_base,
            k_attraction_joints=k_attraction_joints,
            k_repulsion_joints=k_repulsion_joints,
            k_repulsion_robot_joints=k_repulsion_robot_joints,
            k_update_joints=k_update_joints,
            k_orientation=k_orientation,
            k_orientation_from_base=k_orientation_from_base,
            k_safety_joints=k_safety_joints,
            safe_config=safe_config,
            obstacle_threshold=obstacle_threshold
        )

        # C++ planner instance (only for update_path)
        self.EBAND_CPP = None
        if _HAS_CPP_EBAND:
            try:
                # Binding signature:
                # ElasticBandPlanner(k_attraction_base, k_repulsion_base,
                #                    k_attraction_joints, k_repulsion_joints,
                #                    k_repulsion_robot_joints, k_update_joints,
                #                    k_orientation, k_orientation_from_base,
                #                    k_position_from_orientation, k_safety_joints,
                #                    obstacle_threshold)
                self.EBAND_CPP = ElasticBandPlannerCPP(
                    k_attraction_base=k_attraction_base,
                    k_repulsion_base=k_repulsion_base,
                    k_attraction_joints=k_attraction_joints,
                    k_repulsion_joints=k_repulsion_joints,
                    k_repulsion_robot_joints=k_repulsion_robot_joints,
                    k_update_joints=k_update_joints,
                    k_orientation=k_orientation,
                    k_orientation_from_base=k_orientation_from_base,
                    k_position_from_orientation=0.0,
                    k_safety_joints=k_safety_joints,
                    obstacle_threshold=obstacle_threshold,
                )

                # Initialize with your SDF/voxel file
                sdf_bin_path = self.path + "sdf.bin"  # <-- adjust as needed
                robot_sdf_bin_path = self.path + "robot_sdf_002.bin"
                self.EBAND_CPP.initialize(sdf_bin_path, robot_sdf_bin_path)
                self.EBAND_CPP.set_safe_config(safe_config, num_joints=NUM_JOINTS)
                self.EBAND_CPP.set_dynamic_safety(True, center_activation_safety=center_activation_safety)
            except Exception as e:
                rospy.logwarn("Failed to initialize C++ ElasticBandPlanner: %s", e)
                self.EBAND_CPP = None

        self.joint_names = [
            'base_link',
            'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'
        ]

    def run(self):
        global NUMBER_OF_ATTEMPTS, NUMBER_WAYPOINTS_IN_PATH
        number_of_attempts = NUMBER_OF_ATTEMPTS
        obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67])  # pink
        execution_times = []
        print("Read obstacles!")

        while not rospy.is_shutdown() and number_of_attempts > 0:
            # test path planning with random start and end
            start_coord = [start['x'], start['y'], start['z'], start['yaw']]
            end_coord = [goal['x'], goal['y'], goal['yaw']]

            start_time = time.time()

            path = self.Graph.plan(start_coord, end_coord)
            path = sample_path(path, NUMBER_WAYPOINTS_IN_PATH)  # 25
            path[-1]['yaw'] = goal['yaw']

            # complete the path initialization with roll, pitch and qs
            path = interpolate_path(path, start, goal)
            end_time = time.time()
            print("ORIGINAL PATH. Time: ", end_time - start_time)
            # self.plot_path_3d(path, obstacle_mesh)

            # Optimize the path: prefer C++ binding, fallback to Python impl
            start_time = time.time()
            try:
                if self.EBAND_CPP is not None:
                    new_path = self.EBAND_CPP.update_path(path, 500, 2e-2)
                else:
                    new_path = self.EBAND.update_path(
                        path, obstacle_mesh, self.RM, 200, convergence_threshold=2e-2
                    )
            except Exception as e:
                rospy.logwarn("update_path failed (%s). Falling back to original path.", e)
                new_path = path
            end_time = time.time()
            print("Execution time:", end_time - start_time)

            # self.plot_path_3d(new_path, obstacle_mesh)

            # with open("/home/rui/pcl_ws/src/mesh_nav/data/testcpp.json", "w") as file:
            #     json.dump(new_path, file, indent=4)

            execution_times.append(end_time - start_time)
            number_of_attempts -= 1
            self.rate.sleep()

        execution_times = np.asarray(execution_times)
        print("Avg time: ", np.mean(execution_times),
              " | std : ", np.std(execution_times),
              " | min: ", np.min(execution_times),
              " | max: ", np.max(execution_times))

    def plot_path_3d(self, path, obstacle_mesh):
        all_bbs = []
        all_bbs.append(obstacle_mesh)
        colormap = plt.get_cmap('jet')
        padding = 0.8

        for pose in path:
            meshes = self.RM.update_robot_arm_bbs(pose, move_base=True, local_frame=False)

            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in meshes.values():
                combined_mesh += mesh

            bb = combined_mesh.get_axis_aligned_bounding_box()
            bb.scale(2.0, bb.get_center())
            # simplify obstacle mesh in a cropped region
            obstacle_mesh_cropped = obstacle_mesh.crop(bb)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

            if is_colliding_o3d(combined_mesh, obstacle_mesh_cropped):
                mesh_color = [1.0, 0, 0]
            else:
                distance, _ = self.EBAND.compute_2mesh_distance(combined_mesh, obstacle_mesh_cropped)
                norm_distance = (1 - min(distance, obstacle_threshold) / (obstacle_threshold)) * padding
                mesh_color = np.asarray(colormap(norm_distance))[:3]

            combined_mesh.paint_uniform_color(mesh_color)
            combined_mesh.compute_vertex_normals()
            all_bbs.append(combined_mesh)

        o3d.visualization.draw_geometries(all_bbs)

    def import_obstacles(self):
        obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67])  # pink
        return obstacle_mesh

    def import_path(self, path_dir):
        with open(path_dir, "r") as file:
            path = json.load(file)
        return path


def generate_colorbar(obstacle_threshold=obstacle_threshold, padding=0.5, cmap='jet'):
    colormap = plt.get_cmap(cmap)

    # Create normalized gradient from 0 to obstacle_threshold
    gradient = np.linspace(0, obstacle_threshold, 256).reshape(-1, 1)  # Make it vertical
    norm_distance = 1 - np.clip(gradient / obstacle_threshold, 0, 1)  # Normalize between 0 and 1

    # Display vertical gradient colorbar
    fig, ax = plt.subplots(figsize=(2, 6))
    ax.imshow(norm_distance, aspect='auto', cmap=colormap, origin='lower')

    # Set labels
    ax.set_ylabel('Distance')
    ax.set_yticks([0, 128, 256])
    ax.set_yticklabels([f"{0}", f"{obstacle_threshold/2:.2f}", f"{obstacle_threshold:.2f}"])
    ax.set_xticks([])
    ax.set_title("Colormap Distance Representation")

    plt.show()


if __name__ == '__main__':
    mn = MeshNav()
    mn.run()
    obstacle_mesh = mn.import_obstacles()
