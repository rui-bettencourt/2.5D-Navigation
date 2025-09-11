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
k_attraction_base = 0.1
k_repulsion_base = 0.02
k_attraction_joints = 0.1
k_repulsion_joints = 0.05 #0.009
k_repulsion_robot_joints = 0.05 #0.005
k_safety_joints = 0.08 #2  # 0.0
k_update_joints = 0.2
k_orientation = 0.05
k_orientation_from_base = 0.0
obstacle_threshold = 1.5
manipulator = True

NUMBER_OF_ATTEMPTS = 1
NUMBER_WAYPOINTS_IN_PATH = 15

# Number of arm joints expected by the binding (q1..q7)
NUM_JOINTS = 6

# start = {'x': 0.0, 'y': 0.0, 'z': 0.0,
#          'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
#          'q1': 0.0, 'q2': -0.05, 'q3': 0.0, 'q4': 0.02, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}
start = {'x': -9.0, 'y': -3.8, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': -0.93,
         'q1': -1.57, 'q2': 0.0, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0}

# goal = {'x': 4.0, 'y': 0.0, 'z': 0.0,
#         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
#         'q1': 0.0, 'q2': -0.05, 'q3': 0.0, 'q4': 0.02, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}
goal = {'x': 2.8, 'y': -3.8, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': -1.51,
        'q1': 0.0, 'q2': -1.57, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0}

safe_config = {'q1': 0.0, 'q2': -2.242, 'q3': 2.228, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0}
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
                robot_sdf_bin_path = self.path + "robot_mm_sdf.bin"
                self.EBAND_CPP.initialize(sdf_bin_path, robot_sdf_bin_path)
                self.EBAND_CPP.set_safe_config(safe_config, num_joints=NUM_JOINTS)
                self.EBAND_CPP.set_dynamic_safety(True, center_activation_safety=center_activation_safety)
            except Exception as e:
                rospy.logwarn("Failed to initialize C++ ElasticBandPlanner: %s", e)
                self.EBAND_CPP = None

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

            self.plot_path_3d(new_path, obstacle_mesh)

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

    def run_batch(self):
        """
        Batch-mode path planning aligned with REMANI outputs.
        - Reads planner_tests.csv
        - Per test: plan + (optionally) optimize path
        - Writes per-test trajectory CSV: out_dir/test_<id>.csv
            * Header matches REMANI: t,x,y,yaw,q1..q6
            * t is the integer waypoint index (as time)
        - Appends planning time to out_dir/planning_times.csv
        - Appends trajectory size to out_dir/planning_size.csv
        - Waypoint count per test:
            * If present in REMANI planning_size.csv and not FAIL, use it
            * Else fallback to ceil(alpha_wp_per_meter * euclidean distance(start, goal))
        ROS params (all optional):
        ~batch_csv_path              (str) path to planner_tests.csv (default: self.path + 'planner_tests.csv')
        ~batch_out_dir               (str) output folder for CSVs    (default: self.path + 'tests_eband')
        ~batch_start_id              (int) first test_id to run      (default: 0)
        ~batch_resume                (bool) skip tests already in planning_times.csv (default: True)
        ~batch_skip_if_traj_exists   (bool) skip if test_<id>.csv exists            (default: True)
        ~batch_wp_per_meter          (float) alpha for fallback waypoint count      (default: 4.0)
        ~remani_dir                  (str) folder with REMANI planning_size.csv     (default: batch_out_dir)
        """
        import os, csv, math, time
        import numpy as np
        import rospy

        # --------- helpers ----------
        def ensure_dir(d):
            os.makedirs(d, exist_ok=True)
            return d

        def file_exists(p):
            return os.path.isfile(p)

        def ensure_csv_with_header(path, header_line):
            if not file_exists(path):
                with open(path, 'w') as f:
                    f.write(header_line + '\n')

        def append_csv_line(path, line):
            with open(path, 'a') as f:
                f.write(line + '\n')
                f.flush()

        def trim_cell(s):
            if s is None:
                return ''
            s = str(s).strip().strip('\r')
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                s = s[1:-1]
            return s

        def read_csv_rows(path):
            rows = []
            with open(path, 'r') as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    parts = [trim_cell(x) for x in raw.split(',')]
                    rows.append(parts)
            return rows

        def header_index_map(header, names):
            idx = {name: -1 for name in names}
            for i, h in enumerate(header):
                if h in idx:
                    idx[h] = i
            return idx

        def load_done_ids(times_csv):
            done = set()
            if not file_exists(times_csv):
                return done
            rows = read_csv_rows(times_csv)
            if not rows:
                return done
            for parts in rows[1:]:
                if not parts:
                    continue
                try:
                    test_id = int(float(parts[0]))
                    done.add(test_id)
                except Exception:
                    pass
            return done

        def load_remani_sizes(remani_size_csv):
            """Return {test_id -> int or 'FAIL'} from REMANI planning_size.csv (header: test_id,traj_npts)"""
            sizes = {}
            if not file_exists(remani_size_csv):
                return sizes
            rows = read_csv_rows(remani_size_csv)
            if not rows:
                return sizes
            header = rows[0]
            idx = header_index_map(header, ['test_id', 'traj_npts'])
            if idx['test_id'] < 0 or idx['traj_npts'] < 0:
                return sizes
            for parts in rows[1:]:
                if len(parts) <= max(idx.values()):
                    continue
                try:
                    tid = int(float(parts[idx['test_id']]))
                except Exception:
                    continue
                val = parts[idx['traj_npts']]
                if val.upper() == 'FAIL':
                    sizes[tid] = 'FAIL'
                else:
                    try:
                        sizes[tid] = int(float(val))
                    except Exception:
                        sizes[tid] = 'FAIL'
            return sizes

        def write_traj_csv_aligned(filename, path_list):
            """
            Write trajectory aligned with REMANI header:
            t,x,y,yaw,q1..q6
            t is the integer index (as time), formatted to 9 decimals.
            """
            with open(filename, 'w', newline='') as f:
                w = csv.writer(f)
                header = ['t', 'x', 'y', 'yaw'] + [f'q{i}' for i in range(1, NUM_JOINTS + 1)]
                w.writerow(header)
                for i, p in enumerate(path_list):
                    # format with 9 decimals for consistency
                    row = [f"{float(i):.9f}",
                        f"{p.get('x', 0.0):.9f}",
                        f"{p.get('y', 0.0):.9f}",
                        f"{p.get('yaw', 0.0):.9f}"]
                    for j in range(1, NUM_JOINTS + 1):
                        row.append(f"{p.get(f'q{j}', 0.0):.9f}")
                    w.writerow(row)

        # --------- params / IO ----------
        csv_in        = rospy.get_param('~batch_csv_path', os.path.join(self.path, 'planner_tests.csv'))
        out_dir       = rospy.get_param('~batch_out_dir',  os.path.join(self.path, 'testsmeshnav'))
        start_id      = rospy.get_param('~batch_start_id', 0)
        resume        = rospy.get_param('~batch_resume', True)
        skip_if_exist = rospy.get_param('~batch_skip_if_traj_exists', True)
        alpha_wp_m    = rospy.get_param('~batch_wp_per_meter', 4.0)
        remani_dir    = rospy.get_param('~remani_dir', out_dir)

        ensure_dir(out_dir)
        times_csv = os.path.join(out_dir, 'planning_times.csv')
        size_csv  = os.path.join(out_dir, 'planning_size.csv')
        ensure_csv_with_header(times_csv, 'test_id,plan_time_sec')
        ensure_csv_with_header(size_csv,  'test_id,traj_npts')

        done_ids = load_done_ids(times_csv) if resume else set()
        remani_sizes = load_remani_sizes(os.path.join(remani_dir, 'planning_size.csv'))

        # Preload obstacle mesh for Python fallback
        obstacle_mesh = o3d.io.read_triangle_mesh(os.path.join(self.path, 'obstacles.ply'))
        if not obstacle_mesh.is_empty():
            obstacle_mesh.compute_vertex_normals(False)

        # --------- read tests ----------
        rows = read_csv_rows(csv_in)
        if len(rows) < 2:
            rospy.logerr("planner_tests.csv has no data rows: %s", csv_in)
            return

        header = rows[0]
        needed = (
            ['test_id',
            'start_x', 'start_y', 'start_z', 'start_roll', 'start_pitch', 'start_yaw',
            'goal_x',  'goal_y',  'goal_z',  'goal_roll',  'goal_pitch',  'goal_yaw'] +
            [f'start_q{i}' for i in range(1, NUM_JOINTS + 1)] +
            [f'goal_q{i}'  for i in range(1, NUM_JOINTS + 1)]
        )
        idx = header_index_map(header, needed)
        missing = [k for k, v in idx.items() if v < 0]
        if missing:
            rospy.logerr("Missing columns in planner_tests.csv: %s", ','.join(missing))
            return

        # --------- main ----------
        for r in rows[1:]:
            try:
                test_id = int(float(r[idx['test_id']]))
            except Exception:
                rospy.logwarn("Skipping row with invalid test_id: %s", r)
                continue

            if test_id < start_id:
                continue
            if resume and test_id in done_ids:
                rospy.loginfo("[Batch] skip test %d (already in planning_times.csv)", test_id)
                continue

            traj_file = os.path.join(out_dir, f"test_{test_id}.csv")
            if skip_if_exist and file_exists(traj_file):
                rospy.loginfo("[Batch] skip test %d (trajectory exists)", test_id)
                continue

            # Build start/goal dicts
            try:
                s = {
                    'x': float(r[idx['start_x']]), 'y': float(r[idx['start_y']]), 'z': float(r[idx['start_z']]),
                    'roll': float(r[idx['start_roll']]), 'pitch': float(r[idx['start_pitch']]), 'yaw': float(r[idx['start_yaw']])
                }
                g = {
                    'x': float(r[idx['goal_x']]),  'y': float(r[idx['goal_y']]),  'z': float(r[idx['goal_z']]),
                    'roll': float(r[idx['goal_roll']]), 'pitch': float(r[idx['goal_pitch']]), 'yaw': float(r[idx['goal_yaw']])
                }
                for j in range(1, NUM_JOINTS + 1):
                    s[f'q{j}'] = float(r[idx[f'start_q{j}']])
                    g[f'q{j}'] = float(r[idx[f'goal_q{j}']])
            except Exception as e:
                rospy.logwarn("Test %d: parse error: %s", test_id, e)
                append_csv_line(times_csv, f"{test_id},FAIL")
                append_csv_line(size_csv,  f"{test_id},FAIL")
                continue

            # Determine waypoint count
            rs = remani_sizes.get(test_id, None)
            if isinstance(rs, int):
                n_wp = rs
            else:
                dist = math.hypot(g['x'] - s['x'], g['y'] - s['y'])
                n_wp = max(3, int(math.ceil(alpha_wp_m * dist)))

            t0 = time.time()
            try:
                coarse = self.Graph.plan([s['x'], s['y'], s['z'], s['yaw']], [g['x'], g['y'], g['yaw']])
                if coarse is None or len(coarse) < 2:
                    raise RuntimeError("Graph.plan returned empty path")

                coarse = sample_path(coarse, n_wp)
                if coarse:
                    coarse[-1]['yaw'] = g['yaw']

                path = interpolate_path(coarse, s, g)

                # Optimize path
                failed = False
                if self.EBAND_CPP is not None:
                    try:
                        new_path = self.EBAND_CPP.update_path(path, 500, 2e-2)
                    except Exception as e:
                        rospy.logwarn("CPP EBAND failed (%s). Falling back to Python.", e)
                        # new_path = self.EBAND.update_path(path, obstacle_mesh, self.RM, 200, convergence_threshold=2e-2)
                        failed = True
                else:
                    # new_path = self.EBAND.update_path(path, obstacle_mesh, self.RM, 200, convergence_threshold=2e-2)
                    failed = True
                    print("CPP code not working")
                    exit()

                if not new_path or len(new_path) < 2:
                    new_path = path  # keep something reasonable
                    failed = True

                plan_time = time.time() - t0

                # Save outputs incrementally
                if not failed:
                    write_traj_csv_aligned(traj_file, new_path)
                    append_csv_line(times_csv, f"{test_id},{plan_time:.6f}")
                    append_csv_line(size_csv,  f"{test_id},{len(new_path)}")
                    rospy.loginfo("[Batch] test %d OK: n=%d  time=%.3fs  -> %s",
                                test_id, len(new_path), plan_time, traj_file)
                else:
                    rospy.logwarn("[Batch] test %d FAILED (%.3fs): %s", test_id, plan_time, e)
                    try:
                        with open(traj_file, 'w') as f:
                            f.write("status\nFAILED\n")
                    except Exception:
                        pass
                    append_csv_line(times_csv, f"{test_id},FAIL")
                    append_csv_line(size_csv,  f"{test_id},FAIL")

            except Exception as e:
                plan_time = time.time() - t0
                rospy.logwarn("[Batch] test %d FAILED (%.3fs): %s", test_id, plan_time, e)
                try:
                    with open(traj_file, 'w') as f:
                        f.write("status\nFAILED\n")
                except Exception:
                    pass
                append_csv_line(times_csv, f"{test_id},FAIL")
                append_csv_line(size_csv,  f"{test_id},FAIL")

            # mark as done for in-loop resume
            done_ids.add(test_id)
            if rospy.is_shutdown():
                break


if __name__ == '__main__':
    mn = MeshNav()
    # mn.run()
    # obstacle_mesh = mn.import_obstacles()
    mn.run_batch()   # <-- use batch
