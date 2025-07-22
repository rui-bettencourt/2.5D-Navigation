import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import ikpy
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

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
class CircularRobot2D:
    def __init__(self, radius, x=0, y=0, joint1_angle=0, joint2_angle=0, yaw=0):
        self.radius = radius
        self.x = x
        self.y = y
        self.joint1_angle = joint1_angle  # Angle of the first joint
        self.joint2_angle = joint2_angle  # Angle of the second joint
        self.yaw = yaw        # Yaw angle (robot's orientation)

        # Arm lengths, assuming the lengths are equal for simplicity
        self.arm1_length = radius * 1.5  # First arm is 1.5 times the radius
        self.arm2_length = radius * 1.0  # Second arm is the same length as the radius

        # limits
        self.joints_limits = ({'joint1':[-180, 180], 'joint2':[-180, 180]})
        self.joints_limits['joint1'] = np.deg2rad(self.joints_limits['joint1'])
        self.joints_limits['joint2'] = np.deg2rad(self.joints_limits['joint2'])

        # Define the kinematic chain using ikpy
        self.joint2_chain = Chain(name='Full 2D Arm', links=[
            URDFLink(
                name="base_link",
                origin_translation=[0, 0, 0],  # Translation of joint 1
                origin_orientation=[0, 0, 0],                 # Orientation of joint 1
                rotation=[0, 0, 1],                            # Joint 1 rotates around Z-axis (2D plane)
                bounds=self.joints_limits['joint1']
            ),  # Base link without any translation or rotation
            URDFLink(
                name="joint1",
                origin_translation=[self.arm1_length, 0, 0],  # Translation of joint 1
                origin_orientation=[0, 0, 0],                 # Orientation of joint 1
                rotation=[0, 0, 1],                            # Joint 1 rotates around Z-axis (2D plane)
                bounds=self.joints_limits['joint2']
            ),
            URDFLink(
                name="joint2",
                origin_translation=[self.arm2_length, 0, 0],  # Translation of joint 2
                origin_orientation=[0, 0, 0],                 # Orientation of joint 2
                rotation=[0, 0, 1]                            # Joint 2 rotates around Z-axis (2D plane)
            )],
            active_links_mask=[True, True, True])  # Only joints 1 and 2 are active)

        self.joint1_chain = Chain(name='Sub-chain to Joint1', links=self.joint2_chain.links[:2], active_links_mask=[True, True])

    def get_arm_endpoints(self, local_frame=False):
        # Forward kinematics using the chain
        fk_results = self.joint2_chain.forward_kinematics([self.joint1_angle, self.joint2_angle, 0], full_kinematics=True)
        # Extract the (x, y) positions of the arm's first joint and the end effector
        # The position of the first joint is in the second link
        x1, y1 = fk_results[1][0:2, 3]  # Position of the first joint (arm1 endpoint) in the robot frame

        # The position of the end effector is in the final link
        x2, y2 = fk_results[2][0:2, 3]  # Position of the end effector (arm2 endpoint) in the robot frame

        if local_frame:
            joints = {'joint1': np.asarray([x1,y1]), 'joint2': np.asarray([x2, y2])}  # Local frame coordinates of the joints
            
        else:
            joints = {'joint1': self.local_to_world(x1, y1)[0:2], 'joint2': self.local_to_world(x2, y2)[0:2]}
        return joints

    def compute_inverse_kinematics(self, target_x, target_y, chain='joint2'):
        # Target position in 2D space
        target_position = [target_x, target_y, 0]
        
        # Solve inverse kinematics
        import matplotlib.pyplot as plt
        if chain=='joint2':
            joint_angles = self.joint2_chain.inverse_kinematics(target_position)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.joint2_chain.plot(joint_angles,ax)
            plt.show()
        elif chain == 'joint1':
            joint_angles = self.joint1_chain.inverse_kinematics(target_position)
        else:
            print("Wrong chain specified. Use 'joint1' or 'joint2'.")

        return joint_angles

    def set_robot_position(self, x, y):
        self.x = x
        self.y = y
    
    def set_robot_pose(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    def set_joint_angles(self, joint1_angle, joint2_angle):
        # Ensure angles are within bounds
        self.joint1_angle = np.clip(joint1_angle, self.joints_limits['joint1'][0], self.joints_limits['joint1'][1])
        self.joint2_angle = np.clip(joint2_angle, self.joints_limits['joint2'][0], self.joints_limits['joint2'][1])

    def set_yaw_angle(self, yaw_angle):
        # Set the robot's yaw angle (orientation)
        self.yaw = yaw_angle

    def local_to_world(self, x_local, y_local, yaw_local=None):
        """
        Transforms a point from the robot's local frame to the world frame.
        Args:
            x_local (float): x coordinate in the robot frame
            y_local (float): y coordinate in the robot frame
        
        Returns:
            (float, float): The transformed (x, y) coordinates in the world frame
        """
        # Calculate the cosine and sine of the yaw angle
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        # Apply the rotation and translation to transform to the world frame
        x_world = self.x + cos_yaw * x_local - sin_yaw * y_local
        y_world = self.y + sin_yaw * x_local + cos_yaw * y_local

        yaw_world = yaw_local + self.yaw if yaw_local is not None else self.yaw

        # Normalize yaw_world to be within -pi and pi
        yaw_world = (yaw_world + np.pi) % (2 * np.pi) - np.pi

        return np.asarray([x_world, y_world, yaw_world])

    def world_to_local(self, x_world, y_world, yaw_world=None):
        """
        Transforms a point from the world frame to the robot's local frame.
        Args:
            x_world (float): x coordinate in the world frame
            y_world (float): y coordinate in the world frame
        
        Returns:
            (float, float): The transformed (x, y) coordinates in the robot's local frame
        """
        # Calculate the cosine and sine of the yaw angle
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        # Apply the inverse rotation and translation to transform to the robot's local frame
        x_local = cos_yaw * (x_world - self.x) + sin_yaw * (y_world - self.y)
        y_local = -sin_yaw * (x_world - self.x) + cos_yaw * (y_world - self.y)

        yaw_local = yaw_world - self.yaw if yaw_world is not None else self.yaw

        # Normalize yaw_local to be within -pi and pi
        yaw_local = (yaw_local + np.pi) % (2 * np.pi) - np.pi

        return np.asarray([x_local, y_local, yaw_local])

    def world_to_joint_frame(self, force_world, joint_id, yaw, joint_angles):
        """
        Converts a force vector from the world frame to the local frame of a specific joint.

        Args:
            force_world (np.array): Force vector in the world frame (e.g., [Fx, Fy]).
            joint_id (int): The ID of the joint for which we are converting the frame.

        Returns:
            np.array: The force vector in the joint's local frame.
        """
        # Calculate the cumulative rotation angle up to the joint
        rotation_angle = yaw + sum(joint_angles[:joint_id])  # Sum of angles up to this joint
        
        # Rotation matrix from world to joint local frame
        cos_angle = np.cos(-rotation_angle)  # Negative sign to convert world to local
        sin_angle = np.sin(-rotation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], 
                                    [sin_angle, cos_angle]])
        
        # Transform the world frame force to the joint's local frame
        force_local = np.dot(rotation_matrix, force_world)
        return force_local

    def force_to_local(self, force, position): # force and position in world frame
        x_world_frame = force[0] + position[0]  # x-coordinate in world frame
        y_world_frame = force[1] + position[1]   # y-coordinate in world frame
        return self.world_to_local(x_world_frame, y_world_frame)[:2] - self.world_to_local(position[0],position[1])[:2]

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

    def calculate_jacobian_in_joint_frame(self, joint_id, joint_angles):
        """
        Calculate the Jacobian matrix at a specific joint, transformed into that joint's local frame.
        
        Args:
            joint_id (int): ID of the joint (1 for the first joint, 2 for the second, etc.)
            joint_angles (list of floats): List of joint angles [theta1, theta2, ...]

        Returns:
            np.array: The Jacobian matrix in the joint's local frame.
        """
        # Calculate the cumulative rotation angle up to the joint
        rotation_angle = sum(joint_angles[:joint_id])
        
        # Get the Jacobian in the world frame
        J_world = self.calculate_jacobians(joint_angles)[joint_id]
        
        # Rotate Jacobian to align with the joint's local frame
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle], 
                                    [sin_angle, cos_angle]])
        
        # Transform Jacobian from world frame to the joint's local frame
        J_local = np.dot(rotation_matrix.T, J_world)  # Apply transpose to align with local frame
        
        return J_local

    def calculate_joint_torques(self, cartesian_forces, joint, joints_angles=None):
        # Calculate the Jacobian matrix
        # J = self.calculate_jacobians(joints_angles)
        J = self.calculate_jacobian_in_joint_frame(joint,joints_angles)

        # Transpose of the Jacobian
        # J_T = np.transpose(J[joint])
        J_T = np.transpose(J)

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

    def generate_random_pose(self):
        """
        Generates a random pose within specified limits.

        Args:
            x_range (tuple): Min and max values for x position (e.g., (x_min, x_max)).
            y_range (tuple): Min and max values for y position (e.g., (y_min, y_max)).
            joint1_range (tuple): Min and max values for joint1 angle in radians (e.g., (joint1_min, joint1_max)).
            joint2_range (tuple): Min and max values for joint2 angle in radians (e.g., (joint2_min, joint2_max)).

        Returns:
            tuple: Randomly generated pose (x, y, yaw, joint1_angle, joint2_angle).
        """
        x = np.random.uniform(0,10)
        y = np.random.uniform(0,10)
        yaw = np.random.uniform(0,360)
        joint1_angle = np.random.uniform(self.joints_limits['joint1'][0], self.joints_limits['joint1'][1])
        joint2_angle = np.random.uniform(self.joints_limits['joint2'][0], self.joints_limits['joint2'][1])
        
        return x, y, yaw, joint1_angle, joint2_angle

    def visualize(self, obstacles=None):
        fig, ax = plt.subplots()

        # Plot the robot's body
        robot_circle = plt.Circle((self.x, self.y), self.radius, fill=True, color='blue', alpha=0.5)
        ax.add_patch(robot_circle)

        # Plot the robot's yaw as a line indicating direction
        yaw_x = self.x + self.radius * 1.2 * np.cos(self.yaw)
        yaw_y = self.y + self.radius * 1.2 * np.sin(self.yaw)
        ax.plot([self.x, yaw_x], [self.y, yaw_y], 'b-', lw=2, label="Yaw")

        # Plot the robot's arm
        joint_positions = self.get_arm_endpoints()

        ax.plot([self.x, joint_positions['joint1'][0]], [self.y, joint_positions['joint1'][1]], 'r-', lw=4)  # Arm 1
        ax.plot([joint_positions['joint1'][0], joint_positions['joint2'][0]], [joint_positions['joint1'][1], joint_positions['joint2'][1]], 'g-', lw=4)          # Arm 2

        # Plot obstacles if provided
        if obstacles is not None:
            for obstacle in obstacles:
                ax.add_patch(plt.Rectangle(obstacle[:2], 1, 1, color='gray'))

        # Set limits and grid
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.grid(True)
        plt.show()

####

# Environment

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0 for free space, 1 for obstacles

    def add_obstacle(self, x, y):
        """ Adds an obstacle at position (x, y) on the grid """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
        else:
            raise ValueError("Obstacle position out of bounds")

    def remove_obstacle(self, x, y):
        """ Removes an obstacle at position (x, y) on the grid """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 0
        else:
            raise ValueError("Obstacle position out of bounds")

    def get_obstacles(self):
        """ Return a list of obstacle coordinates """
        return [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y, x] == 1]

    def visualize(self):
        fig, ax = plt.subplots()
        for y in range(self.height):
            for x in range(self.width):
                color = 'gray' if self.grid[y, x] == 1 else 'white'
                rect = plt.Rectangle((x, y), 1, 1, color=color, edgecolor='black')
                ax.add_patch(rect)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()

####

class ElasticBandPlanner:
    def __init__(self, robot, environment, path, k_attraction_base=0.2, k_repulsion_base=0.2, k_attraction_joints=0.1, k_repulsion_joints=0.1, k_update_joints=0.1, k_orientation=0.1, k_orientation_from_base=0.1, k_position_from_orientation=0.0, k_safety_joints=0.05, obstacle_threshold=1.5):
        self.robot = robot
        self.environment = environment
        self.path = np.array(path)
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
    
    # def compute_attractive_force_joint(self, current_config, next_config):
    #     """
    #     Compute the attractive force between two configurations.
    #     Args:
    #     - current_config: The current position (x, y) or joint position.
    #     - next_config: The target or next position (x, y) or joint position.
    #     """
    #     # Convert the tuples to numpy arrays to allow element-wise operations
    #     current_config = np.array(current_config)
    #     next_config = np.array(next_config)
        
    #     return (next_config - current_config)


    # def update_repulsive_force(self, direction, position, obstacle_center):
    #     """
    #     Update the repulsive force based on the direction of the force and the distance to the obstacle.
    #     Args:
    #     - direction: The direction of the repulsive force.
    #     - position: The current position of the robot.
    #     - obstacle_center: The center of the obstacle.
    #     Returns:
    #     - The updated repulsive force vector.
    #     """
    #     dist = np.linalg.norm(position - obstacle_center)
    #     if dist < self.obstacle_threshold:
    #         force_magnitude = 1 / (max(self.min_distance_to_obstacle, dist) ** 2)
    #         repulsive_force = force_magnitude * direction
    #         # force_magnitude = self.obstacle_threshold + dist
    #         # repulsive_force = force_magnitude * direction
    #     else:
    #         repulsive_force = np.array([0.0, 0.0])
    #     return repulsive_force

    def compute_repulsive_force_from_obstacle(self, position, obstacle_center):
        repulsive_force = np.array([0.0, 0.0])
        direction = np.array([0.0, 0.0])
        dist = np.linalg.norm(position - obstacle_center)
        if dist < self.obstacle_threshold:
            force_magnitude = 1 / (max(self.min_distance_to_obstacle,dist) ** 2)
            direction = (position - obstacle_center)/dist
            repulsive_force += force_magnitude * direction
        else:
            repulsive_force = np.array([0.0, 0.0])
        return repulsive_force, direction

    def compute_repulsive_force(self, position, frame='global', closest_obstacle_only=True):
        repulsive_force = np.array([0.0, 0.0])
        obstacles = self.environment.get_obstacles()
        if closest_obstacle_only:
            if frame == 'local':
                print('falta este caso')
                exit()
            else:
                # Calculate distances to each obstacle
                dists = np.linalg.norm(obstacles - position, axis=1)
                # Find the closest obstacle
                min_dist_index = np.argmin(dists)
                closest_obstacle = obstacles[min_dist_index]
                obstacle_center = np.array(closest_obstacle) + 0.5  # Center of the obstacle
                force_from_obstacle, _ = self.compute_repulsive_force_from_obstacle(position, obstacle_center)
                repulsive_force += force_from_obstacle
        else:
            for obstacle in obstacles:
                obstacle_center = np.array(obstacle) + 0.5  # Center of the obstacle
                if frame == 'local':
                    obstacle_center = self.robot.world_to_local(obstacle_center[0], obstacle_center[1])[:2]
                force_from_obstacle, _ = self.compute_repulsive_force_from_obstacle(position, obstacle_center)
                repulsive_force += force_from_obstacle
        return repulsive_force

    def update_path(self, iterations=100, convergence_threshold=1e-3):
        repulsive_force_magnitudes = {1:[],2:[]}  # List of lists for each joint's repulsive force magnitudes
        attractive_force_magnitudes = {1:[],2:[]}  # List of lists for each joint's attractive force magnitudes

        for iteration in range(iterations):
            new_path = self.path.copy()
            max_change = 0  # Track the maximum change in the path for convergence
            path_len = len(self.path)

            # during debug plot
            # self.history.append(new_path.copy())
            # self.animate_path_evolution(time_interval=1)

            for i in range(1, path_len - 1):  # Don't modify start and goal
                current_pos = self.path[i, :3]
                next_pos = self.path[i + 1, :3]
                prev_pos = self.path[i - 1, :3]

                # Get arm joint positions for current configuration
                self.robot.set_robot_pose(current_pos[0], current_pos[1], current_pos[2])

                # Get the previous positions of the arm joints
                self.robot.set_joint_angles(self.path[i-1, 3], self.path[i-1, 4])
                prev_joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # Get the next positions of the arm joints
                self.robot.set_joint_angles(self.path[i+1, 3], self.path[i+1, 4])
                next_joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # Get the current positions of the arm joints
                self.robot.set_joint_angles(self.path[i, 3], self.path[i, 4])
                joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # calculate dynamic safety value
                if dynamic_safety:
                    # self.k_safety_joints = self.k_attraction_joints * (float(path_len)-i)/float(path_len)
                    self.k_safety_joints = self.k_attraction_joints * ((1-np.tanh(i-center_activation_safety*path_len))/2)

                # Compute repulsive forces on the arm joints
                joints_torques = np.asarray([0.0,0.0])
                for joint in joint_positions.keys():
                    if joint not in []:
                        joint_id = int(joint.replace('joint', ''))
                        joint_index = joint_id + 2 # 2 from the base position

                        # Compute attractive and repulsive forces on the arm joints
                        # attractive_joint_force = (self.path[i - 1, joint_index] - self.path[i, joint_index])+(self.path[i + 1, joint_index] - self.path[i, joint_index])
                        # # add the attractive force to the torques
                        # joints_torques[joint_id-1] = self.k_attraction_joints * attractive_joint_force
                        attractive_joint_force = self.compute_attractive_force(prev_joint_positions[joint],joint_positions[joint], next_joint_positions[joint])
                        # Convert attractive force to the joint's local frame
                        # local_attractive_joint_force = self.robot.force_to_local(attractive_joint_force, joint_positions[joint])
                        local_attractive_joint_force = self.robot.world_to_joint_frame(attractive_joint_force, joint_id, current_pos[2], joint_angles=[self.path[i, 3], self.path[i, 4]])

                        attractive_torques = self.robot.calculate_joint_torques(local_attractive_joint_force, joint_id, joints_angles=[self.path[i, 3], self.path[i, 4]])
                        joints_torques += self.k_attraction_joints * attractive_torques

                        repulsive_joint_force = self.compute_repulsive_force(joint_positions[joint],frame='global', closest_obstacle_only=closest_obstacle_only)
                        # local_repulsive_joint_force = self.robot.force_to_local(repulsive_joint_force, joint_positions[joint])
                        local_repulsive_joint_force = self.robot.world_to_joint_frame(repulsive_joint_force, joint_id, current_pos[2], joint_angles=[self.path[i, 3], self.path[i, 4]])
                        # Store the magnitudes of the attractive and repulsive forces
                        attractive_force_magnitudes[joint_id].append(np.linalg.norm(local_attractive_joint_force))
                        repulsive_force_magnitudes[joint_id].append(np.linalg.norm(local_repulsive_joint_force))

                        # Update joint angles using the jacobian
                        repulsive_torques = self.robot.calculate_joint_torques(local_repulsive_joint_force, joint_id, joints_angles=[self.path[i, 3], self.path[i, 4]])
                        # update the joint torques
                        joints_torques += self.k_repulsion_joints * repulsive_torques

                        # if i ==6 and joint=='joint2':
                        # print('i: ',iteration,'| rep: f[', repulsive_joint_force,' t[',repulsive_torques, ']| atract: f[', attractive_joint_force,'] t[',attractive_torques,']')
                        # Compute elastic forces to a safe config
                        safe_joint_force = (self.safe_config[joint] - self.path[i, joint_index])
                        # # add the attractive force to the torques
                        joints_torques[joint_id-1] += self.k_safety_joints * safe_joint_force

                new_path[i, 3:] = self.path[i, 3:] + self.k_update_joints * joints_torques

                # Compute forces on the robot base
                attractive_force = self.compute_attractive_force(prev_pos[:2], current_pos[:2], next_pos[:2])
                repulsive_force = self.compute_repulsive_force(current_pos[:2],closest_obstacle_only=False)

                # Compute the orientation attraction force
                orientation_correction = (wrap_angle(self.path[i - 1, 2] - self.path[i, 2])+wrap_angle(self.path[i + 1, 2] - self.path[i, 2]))

                # A larger orientation correction might suggest the robot needs to move differently to achieve this orientation.
                position_adjustment_force = orientation_correction * np.array([np.cos(current_pos[2]), np.sin(current_pos[2])])


                # Total force on the robot base is a sum of attractive and repulsive forces
                total_force = self.k_attraction_base * attractive_force + self.k_repulsion_base * repulsive_force + self.k_position_from_orientation * position_adjustment_force

                # Project the force onto the robot's heading direction
                #heading_vector = np.array([np.cos(current_pos[2]), np.sin(current_pos[2])])

                # Update the position with the total force
                new_pos = current_pos[:2] + total_force

                # Compute the torque on the robot base from the total force
                base_torque_from_total_force = np.arctan2(total_force[1], total_force[0])

                # Update the orientation with the corrective force
                new_theta = current_pos[2] + self.k_orientation * orientation_correction + self.k_orientation_from_base * base_torque_from_total_force

                # Ensure new_theta is within -pi to pi
                new_theta = wrap_angle(new_theta)

                # Update the path
                new_path[i, :2] = new_pos
                new_path[i, 2] = new_theta

                # Calculate the maximum change in this iteration
                max_change = max(max_change, np.linalg.norm(new_path[i, :5] - self.path[i, :5]))


            self.path = new_path
            self.history.append(new_path.copy())  # Store the path at each iteration for animation

            # If the maximum change is smaller than the threshold, we stop early
            if max_change < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations with max change {max_change}")
                break
        # in the end plot magnitudes
        # for joint_id in range(2):
        #     plt.figure()
        #     plt.title(f'Force Magnitudes for Joint {joint_id + 1}')
        #     plt.xlabel('Iteration')
        #     plt.ylabel('Force Magnitude')

        #     # Extract and plot repulsive force magnitudes for the joint across all iterations
        #     repulsive_magnitudes_joint = [repulsive_force_magnitudes[joint_id+1][it] for it in range(len(repulsive_force_magnitudes))]
        #     attractive_magnitudes_joint = [attractive_force_magnitudes[joint_id+1][it] for it in range(len(attractive_force_magnitudes))]

        #     plt.plot(repulsive_magnitudes_joint, label='Repulsive Force Magnitude', color='red')
        #     plt.plot(attractive_magnitudes_joint, label='Attractive Force Magnitude', color='blue')
        #     plt.legend()
        #     plt.show()
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
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Create a colormap that fades from a color to transparent
        cmap = plt.cm.Greys_r  # You can use any color map you like
        norm = Normalize(vmin=0, vmax=1)
        radius = 0.5

        # Plot obstacles
        obstacles = self.environment.get_obstacles()
        # for obs in obstacles:
        #     rect = plt.Rectangle(obs, .1, .1, color="gray")
        #     ax.add_patch(rect)
        # Plot each obstacle as a circle with a gradient
        for obs in obstacles:
            # Create a radial gradient with decreasing intensity
            N = 100  # Number of points for the radial gradient
            x, y = np.linspace(obs[0] - radius, obs[0] + radius, N), np.linspace(obs[1] - radius, obs[1] + radius, N)
            X, Y = np.meshgrid(x, y)
            Z = np.sqrt((X - obs[0])**2 + (Y - obs[1])**2)
            Z = np.clip(Z / radius, 0, 1)  # Normalize to range [0, 1]

            # Create a circular gradient plot
            ax.pcolormesh(X, Y, Z, shading='gouraud', cmap=cmap, norm=norm)


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
        arm_lines = []
        joint1_markers = []
        joint2_markers = []

        # Create empty lines and markers for each arm and joint position in the path
        for _ in range(len(self.history[0])):
            # Arm line from base to joint1 to joint2
            arm_line, = ax.plot([], [], 'm-', lw=2)
            arm_lines.append(arm_line)

            # Joint 1 and Joint 2 markers
            joint1_marker, = ax.plot([], [], 'yo', markersize=8)
            joint2_marker, = ax.plot([], [], 'co', markersize=8)
            joint1_markers.append(joint1_marker)
            joint2_markers.append(joint2_marker)

        # Text for iteration number
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            current_path = self.history[frame]

            # Update robot circles and arm configurations for each base position
            for i, circle in enumerate(robot_circles):
                current_base_pos = current_path[i, :3]
                circle.center = (current_base_pos[0], current_base_pos[1])

                # Update yaw arrow
                yaw_angle = current_path[i, 2] 
                arrow_length = 0.5  # Length of the yaw arrow
                yaw_arrows[i].remove()  # Remove the previous arrow
                yaw_arrows[i] = ax.arrow(current_base_pos[0], current_base_pos[1], 
                                        arrow_length * np.cos(yaw_angle), 
                                        arrow_length * np.sin(yaw_angle), 
                                        head_width=0.1, color='blue')

                self.robot.set_robot_pose(current_base_pos[0], current_base_pos[1], current_base_pos[2])
                self.robot.set_joint_angles(current_path[i, 3], current_path[i, 4])

                # Get the arm's joint positions
                joint_positions = self.robot.get_arm_endpoints()

                # Update arm line (from base to joint1 to joint2)
                arm_lines[i].set_data([current_base_pos[0], joint_positions['joint1'][0], joint_positions['joint2'][0]],
                                    [current_base_pos[1], joint_positions['joint1'][1], joint_positions['joint2'][1]])

                # Update joint marker positions
                joint1_markers[i].set_data([joint_positions['joint1'][0]], [joint_positions['joint1'][1]])
                joint2_markers[i].set_data([joint_positions['joint2'][0]], [joint_positions['joint2'][1]])

                # Compute attractive and repulsive forces for the base
                if i > 0 and i < len(current_path) - 1:
                    next_pos = current_path[i + 1, :2]
                    prev_pos = current_path[i - 1, :2]

            # Update the iteration number
            iteration_text.set_text(f"Iteration: {frame + 1}")

            # Return all objects that are drawn in each frame
            return (*robot_circles, *arm_lines, *joint1_markers, *joint2_markers,
                    *yaw_arrows, iteration_text)

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(self.history), interval=time_interval, repeat=False)
        plt.show()


####

# Example usage of ElasticBandPlanner with a robot and environment

# Create the robot
robot = CircularRobot2D(radius=1, x=2, y=2, joint1_angle=np.radians(0), joint2_angle=np.radians(0), yaw=np.radians(0))

# Create the environment with obstacles
env = Environment(width=10, height=10)
env.add_obstacle(7, 4)
env.add_obstacle(6, 7)
env.add_obstacle(2, 3)
# env.add_obstacle(4, 5)

# # call function to interpolate
# initial_path = robot.interpolate_path(n=num_samples, start_pose=start, goal_pose=goal)
# # initial_path = robot.interpolate_path(n=10)

# # Initialize the elastic band planner
# planner = ElasticBandPlanner(robot, env, initial_path, k_attraction_base=k_attraction_base, k_repulsion_base=k_repulsion_base, k_attraction_joints=k_attraction_joints,
#                              k_repulsion_joints=k_repulsion_joints, k_update_joints=k_update_joints, k_orientation=k_orientation, k_orientation_from_base=k_orientation_from_base,
#                              k_safety_joints=k_safety_joints, obstacle_threshold=obstacle_threshold)

# # Update the path using the elastic band algorithm
# smoothed_path = planner.update_path(iterations=500,convergence_threshold=5e-3)

# # Animate the path evolution
# planner.animate_path_evolution(time_interval=100)
angles = robot.compute_inverse_kinematics(0.5,0.5, chain='joint2')
print(angles)
