#!/usr/bin/env python3
import numpy as np
import json
import os, sys

# import other created files
from mesh_nav_ros.graph_functions import GraphManager
from mesh_nav_ros.aux_functions import interpolate_path, sample_path, calculate_path_distance, apply_safety_configuration_to_path

# NEW: C++ elastic band planner binding (pybind11)
ws = os.path.abspath(os.path.dirname(__file__))
print(ws)
for _ in range(10):  # walk up a few levels
    cand = os.path.join(ws, 'build', 'mesh_nav')
    if os.path.isdir(cand):
        if cand not in sys.path:
            sys.path.append(cand)
        break
    parent = os.path.dirname(ws)
    if parent == ws:
        break
    ws = parent
try:
    from elastic_band_planner_cpp import ElasticBandPlanner as ElasticBandPlannerCPP
    _HAS_CPP_EBAND = True
except Exception as e:
    print("Could not import C++ ElasticBandPlanner binding: %s", e)
    _HAS_CPP_EBAND = False

class MeshNav(object):
    def __init__(self, config_path=None):
        if config_path is None:
            print("A config path is required.")
            exit()
        else:
            # load JSON
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # assign keys as attributes dynamically
            for key, value in cfg.items():
                setattr(self, key, value)

        self.Graph = GraphManager(self.path + self.traversablegraphfilename)

        if _HAS_CPP_EBAND:
            try:
                self.EBAND_CPP = ElasticBandPlannerCPP(
                    k_attraction_base=self.k_attraction_base,
                    k_repulsion_base=self.k_repulsion_base,
                    k_attraction_joints=self.k_attraction_joints,
                    k_repulsion_joints=self.k_repulsion_joints,
                    k_repulsion_robot_joints=self.k_repulsion_robot_joints,
                    k_update_joints=self.k_update_joints,
                    k_orientation=self.k_orientation,
                    k_orientation_from_base=self.k_orientation_from_base,
                    k_position_from_orientation=0.0,
                    k_safety_joints=self.k_safety_joints,
                    obstacle_threshold=self.obstacle_threshold,
                )

                # Initialize with your SDF/voxel file
                sdf_bin_path = self.path + self.environmentsdffilename  # <-- adjust as needed
                robot_sdf_bin_path = self.path + self.robotsdffilename
                self.EBAND_CPP.setRobotUrdfPath(self.roboturdffilename)
                self.EBAND_CPP.setRobotBasePoints(self.robot_points_base)
                self.EBAND_CPP.setJointNames(self.joints_names)
                self.EBAND_CPP.initialize(sdf_bin_path, robot_sdf_bin_path)
                self.EBAND_CPP.setRepulsiveJointIndices(self.repulsive_joints)
                self.EBAND_CPP.set_safe_config(self.safe_config, num_joints=self.dof)
                self.EBAND_CPP.set_dynamic_safety(True, center_activation_safety=self.center_activation_safety)
            except Exception as e:
                print("Failed to initialize C++ ElasticBandPlanner: %s", e)
                exit()
        else:
            print("CPP Library not found")
            exit()

    def plan(self, start, goal):
        start_z = start['z'] if 'z' in start else 0.0
        start_roll = start['roll'] if 'roll' in start else 0.0
        start_pitch = start['pitch'] if 'pitch' in start else 0.0

        start_coord = [start['x'], start['y'], start_z, start['yaw']]
        end_coord = [goal['x'], goal['y'], goal['yaw']]
        
        path = self.Graph.plan(start_coord, end_coord)
        number_waypoints = int(np.ceil(self.metersperwaypoint * calculate_path_distance(path)))
        path = sample_path(path, number_waypoints)
        path[-1]['yaw'] = goal['yaw']

        path = interpolate_path(path, start, goal, self.dof)

        try:
            return  self.EBAND_CPP.update_path(path, self.max_iterations, self.convergence_distance, num_joints=self.dof)
        except Exception as e:
            print("update_path failed (%s). Falling back to original path.", e)
            return apply_safety_configuration_to_path(path, self.safe_config, self.dof)

if __name__ == '__main__':
    mn = MeshNav('/home/rui/mesh_nav_ws/src/mesh_nav/config/tiago.json')
    print(mn.safe_config)