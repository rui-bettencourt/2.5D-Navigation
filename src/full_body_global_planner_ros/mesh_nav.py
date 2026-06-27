#!/usr/bin/env python3
import csv
import numpy as np
import json
import os, sys
import time

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
        self.EBAND_CPP.useManipulator(self.use_manipulator)
        # Debug: print the safe configuration that was set on the C++ EBAND planner
        try:
            print(f"Safe config set (num_joints={self.dof}): {self.safe_config}", flush=True)
        except Exception:
            print("Safe config set (could not stringify safe_config)", flush=True)
        if self.k_safety_joints > 0.0:
            self.EBAND_CPP.set_dynamic_safety(True, center_activation_safety=self.center_activation_safety)

    def plan(self, start, goal):
        start_z = start['z'] if 'z' in start else 0.0
        goal_z = goal['z'] if 'z' in goal else start_z
        start_roll = start['roll'] if 'roll' in start else 0.0
        start_pitch = start['pitch'] if 'pitch' in start else 0.0

        start_coord = [start['x'], start['y'], start_z, start['yaw']]
        end_coord = [goal['x'], goal['y'], goal_z]
        
        # Debug: print the start/end coordinates passed to the graph planner
        print(f"MeshNav.plan called with start_coord={start_coord} end_coord={end_coord}", flush=True)
        path = self.Graph.plan(start_coord, end_coord)
        print(f"Graph.plan returned path with {len(path) if path is not None else 0} nodes", flush=True)
        if path is None or len(path) == 0:
            raise RuntimeError("Graph planner returned an empty path")

        # Keep at least two waypoints so downstream interpolation and terminal-yaw
        # assignment remain valid, even when start and goal map to the same graph node.
        path_distance = calculate_path_distance(path)
        number_waypoints = max(2, int(np.ceil(path_distance / self.metersperwaypoint)) + 1)
        path = sample_path(path, number_waypoints)

        if len(path) == 0:
            raise RuntimeError("Path sampling produced no waypoints")

        path[-1]['yaw'] = goal['yaw']

        path = interpolate_path(path, start, goal, self.dof)

        eb_path = self.EBAND_CPP.update_path(path, self.max_iterations, self.convergence_distance, num_joints=self.dof)
        # Debug: log EBAND-updated path summary
        try:
            if eb_path and len(eb_path) > 0:
                print(f"EBAND.update_path returned {len(eb_path)} waypoints; first5={[ {k: wp.get(k) for k in ('x','y','z','yaw')} for wp in eb_path[:5]]}", flush=True)
            else:
                print("EBAND.update_path returned empty path", flush=True)
        except Exception:
            print("EBAND update_path: could not stringify path", flush=True)
        return eb_path

    def save_path(self, path, filepath):
        """Save the planned path to a CSV file for visualization with plot_path.py."""
        if not path:
            raise ValueError("Cannot save an empty path.")

        preferred_order = ["x", "y", "z", "roll", "pitch", "yaw"]
        fieldnames = [name for name in preferred_order if any(name in pose for pose in path)]
        extra_keys = sorted({key for pose in path for key in pose.keys()} - set(fieldnames))
        fieldnames.extend(extra_keys)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(path)
        print(f"Path saved to: {filepath}")

if __name__ == '__main__':
    mn = MeshNav('/home/dolores/tiago_ws/src/full_body_nav_mpc/submodules/2.5D-Navigation/config/tiago_8floor.json')

    # start = {'x': 1.421, 'y': -3.10, 'z': 0.0,
    #      'roll': 0.0, 'pitch': 0.0, 'yaw': 1.99,
    #      'q1': 0.49997891452238863, 'q2': -1.3399987452510393, 'q3': -0.48000630350043716, 'q4': 1.939944987749467, 'q5': -1.489913471550803, 'q6': 1.3700503991114767, 'q7': 0.0}

    # start = {'x': -9.55, 'y': -6.42, 'z': 0.0,
    #      'roll': 0.0, 'pitch': 0.0, 'yaw': 1.99,
    #      'q1': 0.49997891452238863, 'q2': -1.3399987452510393, 'q3': -0.48000630350043716, 'q4': 1.939944987749467, 'q5': -1.489913471550803, 'q6': 1.3700503991114767, 'q7': 0.0}
    #start manip open 'q1': 0.24536540610839241, 'q2': 0.08151408666845633, 'q3': -0.056760050298209824, 'q4': 0.12295132317415515, 'q5': -2.074212027237278, 'q6': 0.18358833735206717, 'q7': 0.0}
    #start person avoidance
    #[-0.16452785718282203, -1.2666020972563956, 1.6379030207586271]
    # start = {'x': -0.16452785718282203, 'y': -1.2666020972563956, 'z': 0.0,
    start= {'x': -11.78, 'y': -7.21, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 2.66,
         'q1': 2.56, 'q2': -0.27, 'q3': -3.11, 'q4': 0.60, 'q5': -1.58, 'q6': -0.01, 'q7': -1.57}

    
    #start = {'x': -0.99, 'y': -0.92, 'z': 0.0,
        #  'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         #'q1': 0.49997891452238863, 'q2': -1.3399987452510393, 'q3': -0.48000630350043716, 'q4': 1.939944987749467, 'q5': -1.489913471550803, 'q6': 1.3700503991114767, 'q7': 0.0}
        # arm to the side for the door
        #'q1': 0.24536540610839241, 'q2': 0.08151408666845633, 'q3': -0.056760050298209824, 'q4': 0.12295132317415515, 'q5': -2.074212027237278, 'q6': 0.18358833735206717, 'q7': 0.0}     
        # arm up for shelf
        # 'q1': 1.6, 'q2': 0.02, 'q3': -3.20, 'q4': 0.99, 'q5': -1.7, 'q6': -0.11, 'q7': 0.0}     
        # arm lock ness
        #'q1': 1.6100128159474758,  'q2': -0.9300216018679532, 'q3': -3.14011767198285, 'q4': 2.2860616638758073, 'q5': -1.4611383590959146, 'q6': 0.08223813763811383, 'q7': 0.0}
    #goal [-0.13158763131980655, 2.6294367929957643, 3.109611631010219]
    # goal = {'x': -0.13158763131980655, 'y': 2.629, 'z': 0.0,
    #         'roll': 0.0, 'pitch': 0.0, 'yaw': 3.109,
    goal = {'x': -16.2, 'y': -3.6, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # for joints 0.4799,0.4201,-1.6100,1.4400,0.4784,-0.2190,-0.0009
            'q1': 0.25, 'q2': 0.45, 'q3': -2.0, 'q4': 1.75, 'q5': -0.56, 'q6': 0.22, 'q7': 0.0}

    
    # goal = {'x': -15.3, 'y': -6.5, 'z': 0.0,
    #         'roll': 0.0, 'pitch': 0.0, 'yaw': 1.57,
    #         'q1': 0.15, 'q2': 0.95, 'q3': -2.89, 'q4': 0.96, 'q5': 1.18, 'q6': -0.89, 'q7': -0.62}
    
    time_start = time.time()
    path = mn.plan(start, goal)
    print("Time to plan is ", time.time()-time_start)
    
    # Save the path to CSV file for visualization with plot_path.py
    # output_path = '/home/dolores/tiago_ws/src/full_body_nav_mpc/submodules/2.5D-Navigation/data/planned_path.csv'
    output_path = '/home/dolores/tiago_ws/src/full_body_nav_mpc/submodules/2.5D-Navigation/data/door.csv'

    mn.save_path(path, output_path)
    
    print(path)