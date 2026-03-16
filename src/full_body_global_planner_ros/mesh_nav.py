#!/usr/bin/env python3
import numpy as np
import json
import os, sys

# import other created files
try:
    from full_body_global_planner_ros.graph_functions import GraphManager
    from full_body_global_planner_ros.aux_functions import (
        interpolate_path,
        sample_path,
        calculate_path_distance,
        apply_safety_configuration_to_path,
    )
except ImportError:
    try:
        from mesh_nav_ros.graph_functions import GraphManager
        from mesh_nav_ros.aux_functions import (
            interpolate_path,
            sample_path,
            calculate_path_distance,
            apply_safety_configuration_to_path,
        )
    except ImportError:
        from graph_functions import GraphManager
        from aux_functions import (
            interpolate_path,
            sample_path,
            calculate_path_distance,
            apply_safety_configuration_to_path,
        )

# C++ elastic band planner binding (pybind11)
module_dir = os.path.abspath(os.path.dirname(__file__))
ws = module_dir
for _ in range(10):  # walk up a few levels
    candidates = [
        os.path.join(ws, 'build', 'full_body_global_planner'),
        os.path.join(ws, 'build', 'mesh_nav'),
        os.path.join(ws, 'install', 'full_body_global_planner', 'lib', 'full_body_global_planner'),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.append(cand)

    parent = os.path.dirname(ws)
    if parent == ws:
        break
    ws = parent
try:
    from elastic_band_planner_cpp import ElasticBandPlanner as ElasticBandPlannerCPP
    _HAS_CPP_EBAND = True
except Exception as e:
    print(f"Could not import C++ ElasticBandPlanner binding: {e}")
    _HAS_CPP_EBAND = False

class MeshNav(object):
    def __init__(self, config_path=None):
        if config_path is None:
            raise ValueError("A config path is required.")
        else:
            # load JSON
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # assign keys as attributes dynamically
            for key, value in cfg.items():
                setattr(self, key, value)

        self.path = os.path.normpath(self.path)

        def _p(*parts):
            return os.path.join(*parts)

        self.Graph = GraphManager(_p(self.path, self.traversablegraphfilename))

        if not _HAS_CPP_EBAND:
            raise RuntimeError(
                "C++ ElasticBandPlanner binding not found (elastic_band_planner_cpp). "
                "Build full_body_global_planner and ensure its build/install binding path is on PYTHONPATH."
            )

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
        sdf_bin_path = _p(self.path, self.environmentsdffilename)
        robot_sdf_bin_path = _p(self.path, self.robotsdffilename)
        self.EBAND_CPP.setRobotUrdfPath(self.roboturdffilename)
        self.EBAND_CPP.setRobotBasePoints(self.robot_points_base)
        self.EBAND_CPP.setJointNames(self.joints_names, _p(self.path, self.joint_limits_path))
        self.EBAND_CPP.initialize(sdf_bin_path, robot_sdf_bin_path)
        self.EBAND_CPP.setRepulsiveJointIndices(self.repulsive_joints)
        self.EBAND_CPP.set_safe_config(self.safe_config, num_joints=self.dof)
        self.EBAND_CPP.set_dynamic_safety(True, center_activation_safety=self.center_activation_safety)

    def plan(self, start, goal):
        start_z = start['z'] if 'z' in start else 0.0
        start_roll = start['roll'] if 'roll' in start else 0.0
        start_pitch = start['pitch'] if 'pitch' in start else 0.0

        start_coord = [start['x'], start['y'], start_z, start['yaw']]
        end_coord = [goal['x'], goal['y'], goal['yaw']]
        
        path = self.Graph.plan(start_coord, end_coord)
        number_waypoints = int(np.ceil(calculate_path_distance(path)/ self.metersperwaypoint))
        path = sample_path(path, number_waypoints)
        path[-1]['yaw'] = goal['yaw']

        path = interpolate_path(path, start, goal, self.dof)

        return self.EBAND_CPP.update_path(path, self.max_iterations, self.convergence_distance, num_joints=self.dof)

if __name__ == '__main__':
    mn = MeshNav('/home/rui/mesh_nav_ws/src/mesh_nav/config/tiago.json')
    print(mn.safe_config)
    start = {'x': 0.0, 'y': 0.0, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'q1': 0.0, 'q2': -0.05, 'q3': 0.0, 'q4': 0.02, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}

    goal = {'x': 4.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'q1': 0.0, 'q2': -0.05, 'q3': 0.0, 'q4': 0.02, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}
    
    path = mn.plan(start,goal)
    print(path)