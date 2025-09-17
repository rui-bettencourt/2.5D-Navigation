#!/usr/bin/env python
# import ros and other libraries
import rospy
import random
import numpy as np
import csv
import math

# import other created files
from robot_mesh_state import RobotMeshState
from collision_detections import *
from graph_functions import GraphManager
from elastic_bands_threads import ElasticBandPlanner
from robot_kinematics import RobotKinematics
# from robot_kinematics_rob_toolbox import RobotKinematics
import time
from aux_functions import interpolate_path, sample_path
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
import json

###### variables
k_attraction_base=0.005
k_repulsion_base=0.0002
k_attraction_joints=0.2
k_repulsion_joints=0.2#09
k_repulsion_robot_joints=0.0
k_safety_joints = 0.3 #0.0
closest_obstacle_only = True
k_update_joints=0.2
k_orientation=0.02
k_orientation_from_base=0.0
obstacle_threshold=1.5
manipulator = True
#min_distance_to_obstacle = 0.05

start = {'x': 0.0, 'y': 0.0, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'q1': 1.57, 'q2': -0.0, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0}

goal = {'x': 4.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'q1': 0.0, 'q2': -0.0, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0, 'q6': 0.0, 'q7': 0.0}
# safe_config = {'q1': 0.2, 'q2': -1.34, 'q3': -0.2, 'q4': 1.94, 'q5': -1.57, 'q6': 1.37, 'q7': 0.0}
safe_config = {'q1': 0.72, 'q2': -0.9, 'q3': -0.88, 'q4': 1.94, 'q5': -1.2, 'q6': 1.37}
center_activation_safety = 0.8
################

class MeshNav(object):
    def __init__(self):
        # begin node
        rospy.init_node('mesh_nav', anonymous=True)

        # variables
        # self.path = '/home/rui/ds/testscilindros/hard/'     # path for files, change this to save in the package
        self.path = '/home/rui/ds/testsiros2025/'     # path for files, change this to save in the package
        self.rate = rospy.Rate(10)              # TODO: make this an argument of launch file

        # create classes needed for navigation
        # self.Meshes = RobotMeshState()
        self.robot_kinematics = RobotKinematics(manipulator)
        self.Graph = GraphManager(self.path+"traversablegroundgraph.pkl")
        self.RM = RobotMeshState(robot_kinematics=self.robot_kinematics)
        self.EBAND = ElasticBandPlanner(robot_kinematics=self.robot_kinematics, k_attraction_base=k_attraction_base, 
                                        k_repulsion_base=k_repulsion_base, k_attraction_joints=k_attraction_joints,
                                        k_repulsion_joints=k_repulsion_joints, k_repulsion_robot_joints=k_repulsion_robot_joints, k_update_joints=k_update_joints,
                                        k_orientation=k_orientation, k_orientation_from_base=k_orientation_from_base,
                                        k_safety_joints=k_safety_joints, safe_config=safe_config,
                                        obstacle_threshold=obstacle_threshold)
        # self.joint_names = ['base_link','joint1','joint2','joint3','joint4','joint5','joint6']

    def run(self):
        # obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        # obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'retail_visuals.ply')
        # obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67]) # pink
        obstacle_mesh = self.import_obstacles()
        execution_times = []
        number_of_attempts = 1
        print("Read obstacles!")

        while not rospy.is_shutdown() and number_of_attempts>0:
            # test path planning with random start and end
            start_coord = [start['x'], start['y'], start['z'], start['yaw']]
            end_coord = [goal['x'], goal['y'], goal['yaw']]

            start_time = time.time()

            path = self.Graph.plan(start_coord, end_coord)
            path = sample_path(path, 15) #25
            path[-1]['yaw'] = goal['yaw']
            # self.plot_path_3d(path, obstacle_mesh)
            # check for collisions all points in the path
            
            # complete the path initialization with roll, pitch and qs
            path = interpolate_path(path,start,goal)
            end_time = time.time()
            print("ORIGINAL PATH. Time: ", end_time - start_time)
            self.plot_path_3d(path, obstacle_mesh)
            # if rospy.is_shutdown():
            #     exit()

            # if there is collisions use some sort of elastic band to move away from the collision
            start_time = time.time()
            new_path = self.EBAND.update_path(path, obstacle_mesh, self.RM, 200, convergence_threshold=2e-2)
            # new_path = path
            end_time = time.time()
            print("Execution time:", end_time - start_time)
            # save execution time to a CSV file
            # with open("/home/rui/pcl_ws/src/mesh_nav/data/testscilindros/planning_times.csv", "a") as file:
            #     file.write(str(end_time - start_time) + "\n")

            # self.EBAND.animate_path_evolution(50)
            self.plot_path_3d(new_path, obstacle_mesh)
            with open("/home/rui/pcl_ws/src/mesh_nav/data//testscilindros/test_hard_path2.json", "w") as file:
                json.dump(new_path, file, indent=4)

            # print(self.RM.robot_state.get_pose(), self.RM.robot_state.joints)
            # exit()
            execution_times.append(end_time-start_time)
            number_of_attempts -= 1
            self.rate.sleep()
        execution_times = np.asarray(execution_times)
        
        print("Avg time: ",np.mean(execution_times)," | std : ", np.std(execution_times), " | min: ", np.min(execution_times), " | max: ", np.max(execution_times))

    def plot_path_3d(self, path, obstacle_mesh):
        all_bbs = []
        all_bbs.append(obstacle_mesh)
        colormap= plt.get_cmap('jet')
        # colormap = cm.get_cmap('jet') #('RdYlGn')
        padding = 0.8
        # then use the code of moving the robot to move to each point in the path
        for pose in path:
            # pose_array = np.array([pose['x'], pose['y'], pose['z'], pose['roll'], pose['pitch'], pose['yaw']])
            # conf_bb = self.RM.simulate_move_joints(self.RM.robot_bbs, self.joint_names[0], pose_array) # this is just moving the base to the pose. arm config is the online one
            # joint_poses = []
            # for key in pose.keys():
            #     if key.startswith('q'):
            #         joint_poses.append(pose[key])
            # conf_bb = self.RM.simulate_move_joints(conf_bb, self.joint_names[1:], joint_poses) # move the joints
            meshes = self.RM.update_robot_arm_bbs(pose, move_base = True, local_frame = False)

            # mesh= self.RM.convert_bbs_to_mesh(conf_bb)
            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in meshes.values():
                combined_mesh += mesh

            bb = combined_mesh.get_axis_aligned_bounding_box()
            bb.scale(2.0,bb.get_center())
            # simplify obstacle mesh
            obstacle_mesh_cropped = obstacle_mesh.crop(bb)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
            obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

            if is_colliding_o3d(combined_mesh, obstacle_mesh_cropped):
                mesh_color = [1.0,0,0]
            else:
                distance, _ = self.EBAND.compute_2mesh_distance(combined_mesh, obstacle_mesh_cropped)
                # print(distance)
                norm_distance = (1-min(distance, obstacle_threshold)/(obstacle_threshold))*padding
                # print(norm_distance)
                # print('----------------')
                mesh_color = np.asarray(colormap(norm_distance))[:3]


            # conf_bb = list(meshes.values())

            # for bb in conf_bb:
            #     bb.color=mesh_color
            # all_bbs.extend(conf_bb)
            combined_mesh.paint_uniform_color(mesh_color)
            combined_mesh.compute_vertex_normals()
            all_bbs.append(combined_mesh)

        o3d.visualization.draw_geometries(all_bbs)

    # def import_obstacles(self):
    #     # obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
    #     obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'retail_visuals.ply')
    #     obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67]) # pink
    #     return obstacle_mesh
    def import_obstacles(self):
        # mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        mesh = o3d.io.read_triangle_mesh(self.path + 'retail_visuals.ply')
        if mesh.is_empty():
            return mesh

        V = np.asarray(mesh.vertices)
        z = V[:, 2]
        zmin, zmax = float(z.min()), float(z.max())

        # Normalize z -> [0,1]
        if zmax == zmin:
            s = np.ones_like(z)
        else:
            s = (z - zmin) / (zmax - zmin)

        # Shades of pink: scale brightness between 0.3 and 1.0
        base = np.array([1.0, 0.72, 0.67], dtype=float)  # pink
        brightness = 0.3 + 0.7 * s                       # 0.3 (dark) .. 1.0 (full)
        colors = np.clip(brightness[:, None] * base[None, :], 0.0, 1.0)

        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def import_path(self, path_dir):
        with open(path_dir, "r") as file:
            path = json.load(file)
        return path

    

    def import_traj_csv(self, csv_path,
                        num_joints=6,
                        angles_in_degrees=False,
                        default_z=0.0,
                        default_roll=0.0,
                        default_pitch=0.0):
        """
        Read a trajectory CSV and return a list[dict] with keys:
        x, y, z, roll, pitch, yaw, q1..qN
        Expected headers: at least x,y,yaw and (optionally) t, q1..qN.
        Missing z/roll/pitch are filled with provided defaults.
        Missing q's are filled with 0.0.
        """
        def _to_float(s):
            try:
                return float(s)
            except Exception:
                return 0.0

        path = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            hdr = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            have_x   = any(h.lower() == "x"   for h in hdr)
            have_y   = any(h.lower() == "y"   for h in hdr)
            have_yaw = any(h.lower() == "yaw" for h in hdr)
            if not (have_x and have_y and have_yaw):
                raise ValueError(f"CSV must contain columns x,y,yaw. Found: {hdr}")

            # normalize key lookup to be case-insensitive
            def g(row, key, default="0"):
                # exact first, then case-insensitive
                if key in row:
                    return row[key]
                for k in row.keys():
                    if k.lower() == key.lower():
                        return row[k]
                return default

            for row in reader:
                pose = {}
                pose["x"] = _to_float(g(row, "x"))
                pose["y"] = _to_float(g(row, "y"))
                pose["z"] = _to_float(g(row, "z", default=str(default_z)))

                pose["roll"]  = _to_float(g(row, "roll",  default=str(default_roll)))
                pose["pitch"] = _to_float(g(row, "pitch", default=str(default_pitch)))
                pose["yaw"]   = _to_float(g(row, "yaw"))

                # joints q1..qN
                for j in range(1, num_joints + 1):
                    key = f"q{j}"
                    pose[key] = _to_float(g(row, key, default="0"))

                # angle unit conversion (if CSV in degrees)
                if angles_in_degrees:
                    pose["roll"]  = math.radians(pose["roll"])
                    pose["pitch"] = math.radians(pose["pitch"])
                    pose["yaw"]   = math.radians(pose["yaw"])
                    for j in range(1, num_joints + 1):
                        key = f"q{j}"
                        pose[key] = math.radians(pose[key])

                path.append(pose)

        rospy.loginfo(f"[import_traj_csv] Loaded {len(path)} poses from {csv_path}")
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
    # mn.run()
    obstacle_mesh = mn.import_obstacles()
    # o3d.visualization.draw_geometries([obstacle_mesh])
    
    # path = mn.import_path("/home/rui/pcl_ws/src/mesh_nav/data/irosusingresult1.json")
    # path = mn.import_path("/home/rui/pcl_ws/src/mesh_nav/data/test_path_goal2.json")
    csv_path = "/home/rui/ds/testsiros2025/testsremani_newdatasetvis/test_166.csv"  # <-- change to your CSV
    # csv_path = "/home/rui/ds/testsiros2025/testsmeshnav_newdataset/test_166.csv"  # <-- change to your CSV
    path = mn.import_traj_csv(
        csv_path,
        num_joints=6,               # UR5
        angles_in_degrees=False,    # set True if your CSV has degrees
        default_z=0.0,
        default_roll=0.0,
        default_pitch=0.0
    )
    
    mn.plot_path_3d(path, obstacle_mesh)
    # generate_colorbar()