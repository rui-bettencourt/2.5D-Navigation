import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tests.construct_ikpy_chains import create_ikpy_chain_from_urdf

# import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import rclpy
from sensor_msgs.msg import JointState

###### variables
k_attraction_base=0.3
k_repulsion_base=0.1
k_attraction_joints=0.2
k_repulsion_joints=0.4
k_safety_joints = 0.1
dynamic_safety = True
closest_obstacle_only = True
k_update_joints=0.2
k_orientation=0.02
k_orientation_from_base=0.0
obstacle_threshold=3.0
min_distance_to_obstacle = 0.05
num_samples = 30
start = [2, 1, 0, 0, 180]
goal = [9,8,90,60,70]
safe_config = {'joint1': 0.0, 'joint2': np.pi}
center_activation_safety = 0.8
number_points_robot = 200
################

def wrap_angle(angle):
    """
    Wraps an angle in radians to the range [-pi, pi].
    
    Args:
        angle (float): Angle in radians.
    
    Returns:
        float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

#####
# ROBOT
class RobotKinematics:
    def __init__(self, radius, x=0, y=0, z=0, roll=0.0,pitch=0.0,yaw=0.0, joints_angles=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
        self.radius = radius
        self.x = x
        self.y = y
        self.z = z
        self.joints_angles = np.asarray(joints_angles)  # Angle of the first joint
        self.yaw = yaw        # Yaw angle (robot's orientation)
        self.roll = roll
        self.pitch = pitch
        self.joints_limits = {}
        ############## CONFIG #################################################################
        urdf_file = '/home/dolores/tiago_ws/src/full_body_nav_rl/urdf/tiago.urdf'
        root_link = ['base_link']
        ee_link = "arm_7_joint"
        self.dof = 7
        ######################################################################################

        # Define the kinematic chain using ikpy
        self.joint7_chain = create_ikpy_chain_from_urdf(urdf_file,root_link,ee_link)
        self.num_links = len(self.joint7_chain.links)
        for link in self.joint7_chain .links:
            if 'arm' in link.name:
                tag = 'joint' + link.name.split('_')[1]
                self.joints_limits[tag] = link.bounds

        self.joint6_chain = Chain(name='Sub-chain to Joint6', links=self.joint7_chain.links[:(self.num_links-1)])
        self.joint5_chain = Chain(name='Sub-chain to Joint5', links=self.joint7_chain.links[:(self.num_links-2)])
        self.joint4_chain = Chain(name='Sub-chain to Joint4', links=self.joint7_chain.links[:(self.num_links-3)])
        self.joint3_chain = Chain(name='Sub-chain to Joint3', links=self.joint7_chain.links[:(self.num_links-4)])
        self.joint2_chain = Chain(name='Sub-chain to Joint2', links=self.joint7_chain.links[:(self.num_links-5)])
        self.joint1_chain = Chain(name='Sub-chain to Joint1', links=self.joint7_chain.links[:(self.num_links-6)])

    def get_arm_endpoints(self, local_frame=False):
        # Forward kinematics using the chain
        angles = np.append((self.num_links-7) * [0.0],self.joints_angles)
        fk_results = self.joint7_chain.forward_kinematics(angles, full_kinematics=True)
        # Extract the (x, y) positions of the arm's first joint and the end effector
        # The position of all joints
        x1, y1, z1 = fk_results[self.num_links-7][0:3, 3]  # Position of the joints
        x2, y2, z2 = fk_results[self.num_links-6][0:3, 3]
        x3, y3, z3 = fk_results[self.num_links-5][0:3, 3]
        x4, y4, z4 = fk_results[self.num_links-4][0:3, 3]
        x5, y5, z5 = fk_results[self.num_links-3][0:3, 3]
        x6, y6, z6 = fk_results[self.num_links-2][0:3, 3]

        # The position of the end effector is in the final link
        x7, y7, z7 = fk_results[self.num_links-1][0:3, 3]  # Position of the end effector (arm2 endpoint) in the robot frame

        if local_frame:
            joints = {'joint1': np.asarray([x1,y1,z1]), 'joint2': np.asarray([x2,y2,z2]), 'joint3': np.asarray([x3,y3,z3]), 'joint4': np.asarray([x4,y4,z4]), 'joint5': np.asarray([x5, y5,z5]), 'joint6': np.asarray([x6,y6,z6]), 'joint7': np.asarray([x7,y7,z7])}  # Local frame coordinates of the joints
            
        else:# make this use z as well
            joints = {'joint1': self.local_to_world([x1,y1,z1]), 'joint2': self.local_to_world([x2, y2, z2]), 'joint3': self.local_to_world([x3, y3, z3]), 'joint4': self.local_to_world([x4, y4, z4]), 'joint5': self.local_to_world([x5, y5, z5]), 'joint6': self.local_to_world([x6, y6, z6]), 'joint7': self.local_to_world([x7, y7, z7])}
        return joints

    def compute_inverse_kinematics(self, target_x, target_y, target_z, chain='joint7'):
        # Target position in 2D space
        target_position = [target_x, target_y, target_z]
        
        # Solve inverse kinematics
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if chain=='joint7':
            joint_angles = self.joint7_chain.inverse_kinematics(target_position)
            self.joint7_chain.plot(joint_angles,ax)
        elif chain == 'joint6':
            joint_angles = self.joint6_chain.inverse_kinematics(target_position)
        elif chain == 'joint5':
            joint_angles = self.joint5_chain.inverse_kinematics(target_position)
        elif chain == 'joint4':
            joint_angles = self.joint4_chain.inverse_kinematics(target_position)
        elif chain == 'joint3':
            joint_angles = self.joint3_chain.inverse_kinematics(target_position)
        elif chain == 'joint2':
            joint_angles = self.joint2_chain.inverse_kinematics(target_position)
            self.joint2_chain.plot(joint_angles,ax)
        elif chain == 'joint1':
            joint_angles = self.joint1_chain.inverse_kinematics(target_position)
        else:
            print("Wrong chain specified. Use 'joint1' or 'joint2'.")

        # Show the plot
        plt.show()
        return joint_angles

    def set_robot_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_robot_pose(self, x, y, z, yaw, roll=0.0, pitch=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch

    def set_joint_angles(self, angle, joint=-1):
        # Ensure angles are within bounds
        if isinstance(angle, (list, np.ndarray)):
            for i, a in enumerate(angle):
                self.joints_angles[i] = np.clip(a, self.joints_limits['joint'+str(i+1)][0], self.joints_limits['joint'+str(i+1)][1])
        else:
            self.joints_angles[joint-1] = np.clip(angle, self.joints_limits['joint'+str(joint)][0], self.joints_limits['joint'+str(joint)][1])

    def set_rotations_angle(self, roll,pitch,yaw):
        #     Set the robot's yaw angle (orientation)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def create_transformation_matrix(self, x, y, z, roll, pitch, yaw):
        """
        Creates a 4x4 homogeneous transformation matrix from position (x, y, z) and
        orientation (roll, pitch, yaw) in radians.
        
        Args:
        - x, y, z: Position coordinates.
        - roll, pitch, yaw: Orientation angles in radians.
        
        Returns:
        - np.array: 4x4 homogeneous transformation matrix.
        """
        # Calculate individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (Rz * Ry * Rx)
        R = Rz @ Ry @ Rx
        
        # Create the 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R  # Top-left 3x3 block is the rotation matrix
        T[:3, 3] = [x, y, z]  # Top-right 3x1 column is the translation vector
        
        return T

    def local_to_world(self, vector, force=False):
        """
        Converts a 3D force vector from the local frame to the world frame based on the robot's pose.
        
        Args:
        - vector (np.array): vector in the world frame, [x, y, z].
        - force (bool): whether it is a force vector or a position vector
        
        Returns:
        - np.array: Transformed force vector in the local frame.
        """
        # Generate the homogeneous transformation matrix
        T = self.create_transformation_matrix(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        
        if force:
            # Extract the rotation part (top-left 3x3 submatrix)
            R = T[:3, :3]
            
            # Apply the rotation to the force vector
            world_force = R @ vector

            return world_force
        else:
            # Convert position to homogeneous coordinates (4x1 vector)
            position_homogeneous = np.array([vector[0], vector[1], vector[2], 1.0])
            
            # Apply the inverse transformation to the position vector
            world_position_homogeneous = T @ position_homogeneous
            
            # Return the first three elements (x, y, z) of the transformed position
            return world_position_homogeneous[:3]

    def world_to_local(self, vector, force=False):
        """
        Converts a 3D vector from the world frame to the local frame based on the robot's pose.
        
        Args:
        - vector (np.array): vector in the world frame, [x, y, z].
        - force (bool): whether it is a force vector or a position vector
        Returns:
        - np.array: Transformed force vector in the local frame.
        """
        # Generate the homogeneous transformation matrix
        T = self.create_transformation_matrix(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        
        if force:
            # Extract the rotation part (top-left 3x3 submatrix)
            R = T[:3, :3]
            
            # Invert the rotation matrix to go from world frame to local frame
            R_inv = np.linalg.inv(R)
            
            # Apply the rotation to the force vector
            local_force = R_inv @ vector

            return local_force
        else:
            # Invert the transformation matrix to go from world frame to local frame
            T_inv = np.linalg.inv(T)
            
            # Convert position to homogeneous coordinates (4x1 vector)
            position_homogeneous = np.array([vector[0], vector[1], vector[2], 1.0])
            
            # Apply the inverse transformation to the position vector
            local_position_homogeneous = T_inv @ position_homogeneous
            
            # Return the first three elements (x, y, z) of the transformed position
            return local_position_homogeneous[:3]

    def calculate_jacobians(self, joints_angles=None):
        if joints_angles is None:
            theta1 = self.joint1_angle
            theta2 = self.joint2_angle
        else:
            theta1, theta2 = joints_angles

        l1 = self.arm1_length
        l2 = self.arm2_length

        # Jacobian for the joint1
        J1 = np.array([[-l1 * np.sin(theta1), 0],
                       [l1 * np.cos(theta1), 0]])

        # Jacobian for the last joint (end effector)
        J2 = np.array([
            [-l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2), -l2 * np.sin(theta1 + theta2)],
            [l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2), l2 * np.cos(theta1 + theta2)]
        ])

        J = {1: J1, 2: J2}

        return J

    def calculate_joint_torques(self, cartesian_forces, joint, joints_angles=None):
        # Calculate the Jacobian matrix
        # J = self.calculate_jacobians(joints_angles)
        J = self.calculate_jacobians(joints_angles)

        # Transpose of the Jacobian
        J_T = np.transpose(J[joint])

        # Calculate joint torques using τ = J^T * F
        joint_torques = np.dot(J_T, cartesian_forces)

        return joint_torques

    def interpolate_path(self, n, start_pose=None, goal_pose=None):
        """
        Interpolates the robot's base position, yaw angle, and joint configurations between a start and goal pose.

        Args:
            start_pose (tuple): Start pose of the robot in the format (x, y, yaw, joint1_angle, joint2_angle).
            goal_pose (tuple): Goal pose of the robot in the format (x, y, yaw, joint1_angle, joint2_angle).
            n (int): Number of intermediate configurations (including the start and goal).
        
        Returns:
            np.ndarray: Array of interpolated poses, where each pose is of the format (x, y, yaw, joint1_angle, joint2_angle).
        """
        if start_pose is None:
            x_start, y_start, yaw_start, joint1_start, joint2_start = self.generate_random_pose()
        else:
            # Split start and goal poses into components
            x_start, y_start, yaw_start, joint1_start, joint2_start = start_pose
        if goal_pose is None:
            x_goal, y_goal, yaw_goal, joint1_goal, joint2_goal = self.generate_random_pose()
        else:
            x_goal, y_goal, yaw_goal, joint1_goal, joint2_goal = goal_pose

        print("start_pose: ", x_start, y_start, yaw_start, joint1_start, joint2_start)
        print("goal_pose: ", x_goal, y_goal, yaw_goal, joint1_goal, joint2_goal)

        # Linearly interpolate for x, y, yaw, joint1_angle, joint2_angle
        x_interp = np.linspace(x_start, x_goal, n)
        y_interp = np.linspace(y_start, y_goal, n)
        yaw_interp = np.linspace(yaw_start, yaw_goal, n)
        joint1_interp = np.linspace(joint1_start, joint1_goal, n)
        joint2_interp = np.linspace(joint2_start, joint2_goal, n)

        # Combine the interpolated values into an array of poses
        interpolated_path = np.vstack([x_interp, y_interp, yaw_interp, joint1_interp, joint2_interp]).T
        
        # Convert to numpy array and force all elements to be float
        interpolated_path= np.array(interpolated_path, dtype=float)
        interpolated_path[:, 2:] = np.deg2rad(interpolated_path[:, 2:])  # Convert joint angles to radians

        return interpolated_path

class ElasticBandPlanner:
    def __init__(self, k_attraction_base=0.2, k_repulsion_base=0.2, k_attraction_joints=0.1, k_repulsion_joints=0.1, k_update_joints=0.1, k_orientation=0.1, k_orientation_from_base=0.1, k_position_from_orientation=0.0, k_safety_joints=0.05, obstacle_threshold=1.5):
        self.robot = RobotKinematics(radius=0.32, x=0, y=0, z=0, joints_angles=[0.0,0.0,0.0,0.0,0.0,0.0,0.0], yaw=np.radians(0))
        # Create the environment with obstacles

        self.k_attraction_base = k_attraction_base
        self.k_repulsion_base = k_repulsion_base
        self.k_repulsion_joints = k_repulsion_joints
        self.k_attraction_joints = k_attraction_joints
        self.k_update_joints = k_update_joints
        self.k_orientation = k_orientation
        self.k_orientation_from_base = k_orientation_from_base
        self.k_position_from_orientation = k_position_from_orientation
        self.k_safety_joints = k_safety_joints
        self.obstacle_threshold = obstacle_threshold
        self.weak_torque_threshold = 0.1
        self.min_distance_to_obstacle = min_distance_to_obstacle
        self.safe_config = safe_config
        self.obstacles_points = None
        self.obs_kdtree = None
        self.history = []  # To store the path evolution

    def compute_attractive_force(self, prev_pos, current_pos, next_pos):
        """
        Compute the attractive force between two configurations.
        Args:
        - current_config: The current position (x, y) or joint position.
        - next_config: The target or next position (x, y) or joint position.
        """
        # Convert the tuples to numpy arrays to allow element-wise operations
        current_pos = np.array(current_pos)
        next_pos = np.array(next_pos)
        direction_previous = prev_pos - current_pos
        direction_next = next_pos - current_pos
        magnitude_previous = np.linalg.norm(direction_previous)
        magnitude_next = np.linalg.norm(direction_next)
        direction_previous = direction_previous/magnitude_previous
        direction_next = next_pos/magnitude_next
        
        if magnitude_previous < self.obstacle_threshold and magnitude_next < self.obstacle_threshold:
            return (prev_pos - current_pos)+(next_pos - current_pos)
        elif magnitude_previous >= self.obstacle_threshold and magnitude_next < self.obstacle_threshold:
            return self.obstacle_threshold*direction_previous + (next_pos - current_pos)
        elif magnitude_next >= self.obstacle_threshold:
            return (prev_pos - current_pos)+direction_next*self.obstacle_threshold
        else:
            return direction_previous*self.obstacle_threshold + direction_next*self.obstacle_threshold

    def plot_path_3d(self, pcl=None, bbs=None, obstacle_mesh=None):
        all_bbs = []
        all_bbs.append(obstacle_mesh)
        point_cloud_obs = o3d.geometry.PointCloud()

        # vertices = np.asarray(obstacle_mesh.vertices)
        point_cloud_obs.points.extend(obstacle_mesh.vertices)
        all_bbs.extend([point_cloud_obs])
        # then use the code of moving the robot to move to each point in the path


        # mesh= self.RM.convert_bbs_to_mesh(bbs)

        # bb = mesh.get_axis_aligned_bounding_box()
        # bb.scale(2.0,bb.get_center())
        # # simplify obstacle mesh
        # obstacle_mesh_cropped = obstacle_mesh.crop(bb)
        # obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
        # obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

        if bbs is not None:
            rand_color = [0,random.uniform(0,1),random.uniform(0,1)]

            bbs = list(bbs.values())

            # for i_bb in bbs:
            #     i_bb.color=rand_color
            # all_bbs.extend(bbs)
            point_cloud = o3d.geometry.PointCloud()
            for bb in bbs:
                vertices = np.asarray(bb.get_box_points())
                point_cloud.points.extend(vertices)
            all_bbs.extend([point_cloud])
        elif pcl is not None:
            all_bbs.extend([pcl])

        o3d.visualization.draw_geometries(all_bbs)

    def compute_repulsive_force_o3d(self, pose, bbs):
        #convert bbs to pose
        global_bbs = self.RM.simulate_move_joints(bbs, 'base_link', pose)
        # print(pose)
        # self.plot_path_3d(global_bbs,self.obs_mesh)
        # Collect all bounding box points into a single array
        # all_bb_points = np.vstack([np.asarray(bb.get_box_points()) for bb in global_bbs.values()])
        
        min_distance = float('inf')
        direction_vector = None
        repulsive_force = np.array([0.0, 0.0, 0.0])

        # crop obstacle mesh and kdtree it
        robot_mesh= self.RM.convert_bbs_to_mesh(global_bbs)
        robot_pcl = robot_mesh.sample_points_uniformly(number_points_robot)
        # all_bb_points = np.asarray(robot_pcl.points)
        # self.plot_path_3d(pcl=robot_pcl,obstacle_mesh=self.obs_mesh)
        bb = robot_mesh.get_axis_aligned_bounding_box()
        bb.scale(2.0,bb.get_center())
        # simplify obstacle mesh
        obstacle_mesh_cropped = self.obs_mesh.crop(bb)
        obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
        obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)
        # obstacles_points = np.asarray(obstacle_mesh_cropped.vertices)
        # obs_kdtree = KDTree(obstacles_points)

        # # Ensure KDTree is initialized
        # if obs_kdtree is None or obstacles_points is None:
        #     raise RuntimeError("KDTree or obstacle points not initialized.")
        # elif len(obstacles_points) == 0:
        #     repulsive_force = np.array([0.0, 0.0, 0.0])
        # else:
        if len(obstacle_mesh_cropped.vertices)>0:
        #TODO: Check if its faster like this, creating kdtree in the beginning or cropping obstacle mesh and converting to kdtree at each pose
            # Perform batch nearest neighbor search
            # for point in all_bb_points:
            #     distance, idx = obs_kdtree.query(point, k=1)
            #     closest_point = obstacles_points[idx]
                
            #     # Update the minimum distance and direction vector found
            #     if distance < min_distance:
            #         min_distance = distance
            #         direction_vector = (point - closest_point) / distance  # Normalized direction vector
            # print("Min distance: ", min_distance)
            
            point_cloud_obs = o3d.geometry.PointCloud()
            point_cloud_obs.points.extend(obstacle_mesh_cropped.vertices)
            pcl_distances = robot_pcl.compute_point_cloud_distance(point_cloud_obs)
            min_distance = min(pcl_distances)

            min_index_robot = np.argmin(pcl_distances)
            # Get the point in robot_pcl that has the minimum distance
            closest_point_robot = np.asarray(robot_pcl.points)[min_index_robot]

            # Now find the corresponding closest point in point_cloud_obs
            # By computing the distances from closest_point_robot to all points in point_cloud_obs
            distances_to_obs = np.linalg.norm(np.asarray(point_cloud_obs.points) - closest_point_robot, axis=1)
            min_index_obs = np.argmin(distances_to_obs)
            closest_point_obs = np.asarray(point_cloud_obs.points)[min_index_obs]
            direction_vector = (closest_point_robot - closest_point_obs) / min_distance 
            # print("Min distance o3d: ",min(o3d_dist))
            # print("------------------")
            
            if min_distance < self.obstacle_threshold:
                force_magnitude = 1 / (max(self.min_distance_to_obstacle,min_distance) ** 2)
                repulsive_force = force_magnitude * direction_vector
            else:
                repulsive_force = np.array([0.0, 0.0, 0.0])

        return repulsive_force

    def update_path(self, path, obs_mesh, RM, iterations=100, convergence_threshold=1e-3):
        self.path = np.array(path)
        self.RM = RM
        self.obs_mesh = obs_mesh

        for iteration in range(iterations):
            new_path = self.path.copy()
            max_change = 0  # Track the maximum change in the path for convergence
            path_len = len(self.path)

            for i in range(1, path_len - 1):  # Don't modify start and goal
                current_pos = self.path[i, :4]
                next_pos = self.path[i + 1, :4]
                prev_pos = self.path[i - 1, :4]

                # Get arm joint positions for current configuration
                self.robot.set_robot_pose(current_pos[0], current_pos[1], current_pos[2], current_pos[3])

                # # Get the previous positions of the arm joints
                # self.robot.set_joint_angles(self.path[i-1, 4:])
                # prev_joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # # # Get the next positions of the arm joints
                # self.robot.set_joint_angles(self.path[i+1, 4:])
                # next_joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # # # Get the current positions of the arm joints
                # self.robot.set_joint_angles(self.path[i, 4:])
                # joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # # calculate dynamic safety value
                # # if dynamic_safety:
                # #     # self.k_safety_joints = self.k_attraction_joints * (float(path_len)-i)/float(path_len)
                # #     self.k_safety_joints = self.k_attraction_joints * ((1-np.tanh(i-center_activation_safety*path_len))/2)

                # # # Compute repulsive forces on the arm joints
                # joints_torques = np.asarray(self.dof * [0.0])
                # for joint in joint_positions.keys():
                #     if joint not in []:
                #         joint_id = int(joint.replace('joint', ''))
                #         joint_index = joint_id + 3 # 2 from the base position

                #         # Compute attractive and repulsive forces on the arm joints
                #         attractive_joint_force = self.compute_attractive_force(prev_joint_positions[joint],joint_positions[joint], next_joint_positions[joint])
                #         # Convert attractive force to the joint's local frame
                #         local_attractive_joint_force = self.robot.world_to_local(attractive_joint_force, force=True)

                #         attractive_torques = self.robot.calculate_joint_torques(local_attractive_joint_force, joint_id, joints_angles=self.path[i, 4:])
                #         joints_torques += self.k_attraction_joints * attractive_torques

                #         repulsive_joint_force = self.compute_repulsive_force(joint_positions[joint],frame='global', closest_obstacle_only=closest_obstacle_only)
                #         # local_repulsive_joint_force = self.robot.force_to_local(repulsive_joint_force, joint_positions[joint])
                #         local_repulsive_joint_force = self.robot.world_to_joint_frame(repulsive_joint_force, 0, current_pos[2], joint_angles=[self.path[i, 3], self.path[i, 4]])


                #         # Update joint angles using the jacobian
                #         repulsive_torques = self.robot.calculate_joint_torques(local_repulsive_joint_force, joint_id, joints_angles=[self.path[i, 3], self.path[i, 4]])
                #         # update the joint torques
                #         joints_torques += self.k_repulsion_joints * repulsive_torques

                #         # if i ==6 and joint=='joint2':
                #         # print('i: ',iteration,'| rep: f[', repulsive_joint_force,' t[',repulsive_torques, ']| atract: f[', attractive_joint_force,'] t[',attractive_torques,']')
                #         # Compute elastic forces to a safe config
                #         safe_joint_force = (self.safe_config[joint] - self.path[i, joint_index])
                #         # # add the attractive force to the torques
                #         joints_torques[joint_id-1] += self.k_safety_joints * safe_joint_force

                # new_path[i, 3:] = self.path[i, 3:] + self.k_update_joints * joints_torques

                # Compute forces on the robot base
                attractive_force = self.compute_attractive_force(prev_pos[:3], current_pos[:3], next_pos[:3])
                repulsive_force = self.compute_repulsive_force_o3d(current_pos,self.RM.body_bbs) # this repulsive force is calculated in the world frame
                repulsive_force = self.robot.world_to_local(repulsive_force,force=True)

                # Compute the orientation attraction force
                orientation_correction = (wrap_angle(self.path[i - 1, 3] - self.path[i, 3])+wrap_angle(self.path[i + 1, 3] - self.path[i, 3]))

                # A larger orientation correction might suggest the robot needs to move differently to achieve this orientation.
                # position_adjustment_force = orientation_correction * np.array([np.cos(current_pos[3]), np.sin(current_pos[3])])


                # Total force on the robot base is a sum of attractive and repulsive forces
                total_force = self.k_attraction_base * attractive_force + self.k_repulsion_base * repulsive_force #+ self.k_position_from_orientation * position_adjustment_force

                # Project the force onto the robot's heading direction
                #heading_vector = np.array([np.cos(current_pos[2]), np.sin(current_pos[2])])

                # Update the position with the total force
                new_pos = current_pos[:3] + total_force

                # Compute the torque on the robot base from the total force
                base_torque_from_total_force = np.arctan2(total_force[1], total_force[0])

                # Update the orientation with the corrective force
                new_theta = current_pos[3] + self.k_orientation * orientation_correction + self.k_orientation_from_base * base_torque_from_total_force

                # Ensure new_theta is within -pi to pi
                new_theta = wrap_angle(new_theta)

                # Update the path
                new_path[i, :3] = new_pos
                new_path[i, 3] = new_theta

                # Calculate the maximum change in this iteration
                max_change = max(max_change, np.linalg.norm(new_path[i, :] - self.path[i, :]))


            self.path = new_path
            self.history.append(new_path.copy())  # Store the path at each iteration for animation

            # If the maximum change is smaller than the threshold, we stop early
            if max_change < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations with max change {max_change}")
                break

        return self.path

    def animate_path_evolution(self, time_interval=200):
        """
        Visualize the path evolution with arrows for forces (attractive and repulsive) on the base,
        and arcs for angular forces at the joints.
        Show each robot position as a blue circle and draw the arm at each step.
        Display the iteration number for better tracking.
        Also visualize arm joints and configuration for every base position in every iteration.
        """
        fig, ax = plt.subplots()
        # Find the min and max for x and y
        self.history = np.asarray(self.history)
        x_min = np.min(self.history[:,:,0])
        x_max = np.max(self.history[:,:,0])
        y_min = np.min(self.history[:,:,1])
        y_max = np.max(self.history[:,:,1])
        ax.set_xlim(x_min-1, x_max+1)
        ax.set_ylim(y_min-1, y_max+1)

        # Create a colormap that fades from a color to transparent
        cmap = plt.cm.Greys_r  # You can use any color map you like
        norm = Normalize(vmin=0, vmax=1)
        radius = 0.5

        # Plot obstacles
        # obstacles = self.environment.get_obstacles()
        # for obs in obstacles:
        #     rect = plt.Rectangle(obs, .1, .1, color="gray")
        #     ax.add_patch(rect)
        # Plot each obstacle as a circle with a gradient
        # for obs in obstacles:
        #     # Create a radial gradient with decreasing intensity
        #     N = 100  # Number of points for the radial gradient
        #     x, y = np.linspace(obs[0] - radius, obs[0] + radius, N), np.linspace(obs[1] - radius, obs[1] + radius, N)
        #     X, Y = np.meshgrid(x, y)
        #     Z = np.sqrt((X - obs[0])**2 + (Y - obs[1])**2)
        #     Z = np.clip(Z / radius, 0, 1)  # Normalize to range [0, 1]

        #     # Create a circular gradient plot
        #     ax.pcolormesh(X, Y, Z, shading='gouraud', cmap=cmap, norm=norm)


        # Initialize robot circles for each position in the path history
        robot_circles = []
        yaw_arrows = []
        for _ in range(len(self.history[0])):
            circle = plt.Circle((0, 0), self.robot.radius, color='blue', fill=False, lw=2)
            ax.add_patch(circle)
            robot_circles.append(circle)

            # Initialize yaw arrow
            arrow = ax.arrow(0, 0, 0, 0, head_width=0.1, color='blue')
            yaw_arrows.append(arrow)

        # List to store lines representing the arm for every position
        # arm_lines = []
        # joint1_markers = []
        # joint2_markers = []

        # Create empty lines and markers for each arm and joint position in the path
        # for _ in range(len(self.history[0])):
        #     # Arm line from base to joint1 to joint2
        #     arm_line, = ax.plot([], [], 'm-', lw=2)
        #     arm_lines.append(arm_line)

        #     # Joint 1 and Joint 2 markers
        #     joint1_marker, = ax.plot([], [], 'yo', markersize=8)
        #     joint2_marker, = ax.plot([], [], 'co', markersize=8)
        #     joint1_markers.append(joint1_marker)
        #     joint2_markers.append(joint2_marker)

        # Text for iteration number
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            current_path = self.history[frame]

            # Update robot circles and arm configurations for each base position
            for i, circle in enumerate(robot_circles):
                current_base_pos = current_path[i, :4]
                circle.center = (current_base_pos[0], current_base_pos[1])

                # Update yaw arrow
                yaw_angle = current_path[i, 3] 
                arrow_length = 0.5  # Length of the yaw arrow
                yaw_arrows[i].remove()  # Remove the previous arrow
                yaw_arrows[i] = ax.arrow(current_base_pos[0], current_base_pos[1], 
                                        arrow_length * np.cos(yaw_angle), 
                                        arrow_length * np.sin(yaw_angle), 
                                        head_width=0.1, color='blue')

                self.robot.set_robot_pose(current_base_pos[0], current_base_pos[1], current_base_pos[2], current_base_pos[3])
                # self.robot.set_joint_angles(current_path[i, 3], current_path[i, 4])

                # Get the arm's joint positions
                # joint_positions = self.robot.get_arm_endpoints()

                # Update arm line (from base to joint1 to joint2)
                # arm_lines[i].set_data([current_base_pos[0], joint_positions['joint1'][0], joint_positions['joint2'][0]],
                                    # [current_base_pos[1], joint_positions['joint1'][1], joint_positions['joint2'][1]])

                # Update joint marker positions
                # joint1_markers[i].set_data([joint_positions['joint1'][0]], [joint_positions['joint1'][1]])
                # joint2_markers[i].set_data([joint_positions['joint2'][0]], [joint_positions['joint2'][1]])


            # Update the iteration number
            iteration_text.set_text(f"Iteration: {frame + 1}")

            # Return all objects that are drawn in each frame
            return (*robot_circles,*yaw_arrows, iteration_text)

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.history), interval=time_interval, repeat=False)
        plt.show()


####

# Example usage of ElasticBandPlanner with a robot and environment
if __name__ == '__main__':
    rclpy.init(args=None)
    node = rclpy.create_node('elastic_band_planner_node')

    def joint_states_callback(msg):
            # Extract the first 7 joint positions
        joint_positions = np.asarray(msg.position[:7])
        # Update the robot joints
        ep.robot.set_joint_angles(joint_positions)

    ep = ElasticBandPlanner(k_attraction_base=k_attraction_base, k_repulsion_base=k_repulsion_base, k_attraction_joints=k_attraction_joints,
                            k_repulsion_joints=k_repulsion_joints, k_update_joints=k_update_joints, k_orientation=k_orientation, k_orientation_from_base=k_orientation_from_base,
                            k_safety_joints=k_safety_joints, obstacle_threshold=obstacle_threshold)

    # initial_path = ep.robot.interpolate_path(n=num_samples, start_pose=start, goal_pose=goal)
    # initial_path = robot.interpolate_path(n=10)
    # ep.update_path(initial_path, iterations=500,convergence_threshold=5e-3)

    # Animate the path evolution
    # ep.animate_path_evolution(time_interval=100)

    # angles = ep.robot.compute_inverse_kinematics(-0.15,0.1,0.63, chain='joint2')
    # print(angles)
    # Subscribe to the /joint_states topic
    node.create_subscription(JointState, '/joint_states', joint_states_callback, 10)
    end_wait_time = time.time() + 2.0
    while rclpy.ok() and time.time() < end_wait_time:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(ep.robot.joints_angles)
    start_time = time.time()
    positions = ep.robot.get_arm_endpoints(local_frame=True)
    end_time = time.time()
    print("Forward kinematics calculation time:", end_time - start_time)
    print(positions)
    
    # Spin the ROS node
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
