import numpy as np
import open3d as o3d
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from aux_functions import wrap_angle, create_bounding_box_corners

# Check if Open3D is compiled with CUDA support
if o3d.core.cuda.is_available():
    print("Open3D is using GPU (CUDA is available).")
else:
    print("Open3D is not using GPU (CUDA is not available).")

###### variables
obstacle_threshold=1.5
min_distance_to_obstacle = 0.0 #5
# num_samples = 30
center_activation_safety = 0.8
number_points_robot = 200
radius_joint = 0.01
################


class ElasticBandPlanner:
    def __init__(self, robot_kinematics, k_attraction_base=0.2, k_repulsion_base=0.2, k_attraction_joints=0.1, k_repulsion_joints=0.1, k_repulsion_robot_joints=0.1, k_update_joints=0.1, k_orientation=0.1, k_orientation_from_base=0.1, k_position_from_orientation=0.0, k_safety_joints=0.05, safe_config={}, obstacle_threshold=1.5):
        self.robot = robot_kinematics
        # Create the environment with obstacles

        self.k_attraction_base = k_attraction_base
        self.k_repulsion_base = k_repulsion_base
        self.k_repulsion_joints = k_repulsion_joints
        self.k_repulsion_robot_joints = k_repulsion_robot_joints
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
        if len(safe_config.keys()) == 0:
            self.k_safety_joints = 0.0
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
        if magnitude_previous > 0.0:
            direction_previous = direction_previous/magnitude_previous
        if magnitude_next > 0.0:
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


    def compute_2mesh_distance(self, mesh1, mesh2, num_points=50, crop=True):
        min_distance = float('inf')
        direction_vector = None

        pcl1 = mesh1.sample_points_uniformly(num_points)

        if crop:
            bb = mesh1.get_axis_aligned_bounding_box()
            bb.scale(2.0,bb.get_center())
            # simplify obstacle mesh
            mesh2_cropped = mesh2.crop(bb)
            mesh2_cropped = mesh2_cropped.simplify_vertex_clustering(0.02)
            mesh2_cropped = mesh2_cropped.simplify_quadric_decimation(2000)
        else:
            mesh2_cropped = mesh2

        if len(mesh2_cropped.vertices)>0:
            point_cloud_2 = o3d.geometry.PointCloud()
            point_cloud_2.points.extend(mesh2_cropped.vertices)
            pcl_distances = pcl1.compute_point_cloud_distance(point_cloud_2)
            min_distance = min(pcl_distances)

            min_index_robot = np.argmin(pcl_distances)
            # Get the point in pcl1 that has the minimum distance
            closest_point_1 = np.asarray(pcl1.points)[min_index_robot]

            # Now find the corresponding closest point in point_cloud_2
            # By computing the distances from closest_point_1 to all points in point_cloud_2
            distances_to_mesh2 = np.linalg.norm(np.asarray(point_cloud_2.points) - closest_point_1, axis=1)
            min_index_obs = np.argmin(distances_to_mesh2)
            closest_point_2 = np.asarray(point_cloud_2.points)[min_index_obs]
            direction_vector = (closest_point_1 - closest_point_2) / min_distance 

        return min_distance, direction_vector

    def compute_force_from_dist(self, min_distance, direction_vector, max_threshold=None):
        if max_threshold is None:
            max_threshold = self.obstacle_threshold
        if min_distance < max_threshold:
            force_magnitude = 1 / (max(self.min_distance_to_obstacle,min_distance) ** 2)
            force = force_magnitude * direction_vector
        else:
            force = np.array([0.0, 0.0, 0.0])
        return force

    def compute_repulsive_force_o3d(self, robot_mesh): #TODO_ REPLACE IN THIS FUNCTION THE BIG CHUNK BY THE COMPUTE DISTANCE FUNCTION
        bb = robot_mesh.get_axis_aligned_bounding_box()
        bb.scale(2.0,bb.get_center())
        # simplify obstacle mesh
        obstacle_mesh_cropped = self.obs_mesh.crop(bb)
        obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
        obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

        if len(obstacle_mesh_cropped.vertices)>0:
            min_distance, direction_vector = self.compute_2mesh_distance(robot_mesh,obstacle_mesh_cropped, crop=False)

            return self.compute_force_from_dist(min_distance, direction_vector)
        else: 
            return np.array([0.0, 0.0, 0.0])
        #TODO: Check if its faster like this, creating kdtree in the beginning or cropping obstacle mesh and converting to kdtree at each pose

    def compute_repulsive_force_joints_o3d(self, point, obstacle_mesh, max_threshold=None):
        joint_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius_joint)

        # Translate the sphere to the desired point (in this case, it's already at the point)
        joint_mesh.translate(point)

        # create the env mesh around the joint
        min_corner, max_corner = create_bounding_box_corners(point,self.obstacle_threshold)
        bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corner, max_bound=max_corner)

        # simplify obstacle mesh
        obstacle_mesh_cropped = obstacle_mesh.crop(bb)
        obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
        obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)
        if len(obstacle_mesh_cropped.vertices)>0:
            min_distance, direction_vector = self.compute_2mesh_distance(joint_mesh,obstacle_mesh_cropped,num_points=1,crop=False)
            # obstacle_mesh.paint_uniform_color([1.0, 0.0, 0.0])
            # obstacle_mesh_cropped.paint_uniform_color([0.0, 0.0,1.0])
            # o3d.visualization.draw_geometries([joint_mesh,obstacle_mesh, obstacle_mesh_cropped])
            # print()(min_distance,direction_vector)
            # exit()
            # print(self.compute_force_from_dist(min_distance, direction_vector))
            return self.compute_force_from_dist(min_distance, direction_vector, max_threshold=max_threshold)
        else: 
            return np.array([0.0, 0.0, 0.0])

    def update_path(self, path, obs_mesh, RM, iterations=100, convergence_threshold=1e-3):
        self.path = self.path_from_dict(path)
        self.RM = RM
        self.obs_mesh = obs_mesh

        for iteration in range(iterations):
            new_path = self.path.copy()
            max_change = 0  # Track the maximum change in the path for convergence
            path_len = len(self.path)

            for i in range(1, path_len - 1):  # Don't modify start and goal
                current_pos = self.path[i, :6]
                next_pos = self.path[i + 1, :6]
                prev_pos = self.path[i - 1, :6]

                start_time = time.time()

                # Get arm joint positions for current configuration
                self.robot.set_robot_pose(current_pos[0], current_pos[1], current_pos[2], roll=current_pos[3], pitch=current_pos[4], yaw=current_pos[5])

                # Get the previous positions of the arm joints
                # TODO: get the three forward kinematics at the same time to use GPU
                self.robot.set_joint_angles(self.path[i-1, 6:])
                prev_joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # # Get the next positions of the arm joints
                self.robot.set_joint_angles(self.path[i+1, 6:])
                next_joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # # Get the current positions of the arm joints
                self.robot.set_joint_angles(self.path[i, 6:])
                joint_positions = self.robot.get_arm_endpoints(local_frame=False)

                # Get body mesh
                global_bbs = self.RM.simulate_move_joints(self.RM.body_bbs, 'base_link', current_pos)
                # crop obstacle mesh and kdtree it
                robot_mesh= self.RM.convert_bbs_to_mesh(global_bbs)

                # # calculate dynamic safety value
                # # if dynamic_safety:
                # #     # self.k_safety_joints = self.k_attraction_joints * (float(path_len)-i)/float(path_len)
                # #     self.k_safety_joints = self.k_attraction_joints * ((1-np.tanh(i-center_activation_safety*path_len))/2)

                # end_time = time.time()
                # print("Forward kinematics times:", end_time - start_time)
                
                # # Compute repulsive forces on the arm joints
                joints_torques = np.asarray(self.robot.dof * [0.0])
                for joint in joint_positions.keys():
                    if joint not in []:
                        start_time = time.time()
                        joint_id = int(joint.replace('q', ''))
                        joint_index = joint_id + 5 # 2 from the base position x,y,z and rpy

                        ###### ATTRACTIVE JOINT FORCES ##########
                        # Compute attractive and repulsive forces on the arm joints
                        attractive_joint_force = self.compute_attractive_force(prev_joint_positions[joint],joint_positions[joint], next_joint_positions[joint])
                        # Convert attractive force to the joint's local frame
                        local_attractive_joint_force = self.robot.world_to_local(attractive_joint_force, force=True)

                        attractive_torques, J_joint = self.robot.calculate_joint_torques(local_attractive_joint_force, joint_id, joints_angles=self.path[i, 6:])
                        joints_torques += self.k_attraction_joints * attractive_torques
                        # end_time = time.time()
                        # print("joint " + str(joint_id) + " time:", end_time - start_time)
                        # print("atractive: ", attractive_torques)

                        # ###### REPULSIVE ENVIRONMENT JOINT FORCES #############
                        if joint_id>1:
                            repulsive_joint_force = self.compute_repulsive_force_joints_o3d(joint_positions[joint], self.obs_mesh+robot_mesh)
                        else:
                            repulsive_joint_force = self.compute_repulsive_force_joints_o3d(joint_positions[joint], self.obs_mesh)
                        local_repulsive_joint_force = self.robot.world_to_local(repulsive_joint_force, force=True)

                        # Update joint angles using the jacobian
                        repulsive_torques, _ = self.robot.calculate_joint_torques(local_repulsive_joint_force, joint_id, joints_angles=self.path[i, 6:], J=J_joint)
                        # update the joint torques
                        joints_torques += self.k_repulsion_joints * repulsive_torques
                        # print("repulsive obstacles: ", repulsive_torques)

                        ###### REPULSIVE SELF-COLLISION JOINT FORCES #############
                        # if joint_id > 1: ############### Change this to better suit it
                        #     repulsive_joint_force = self.compute_repulsive_force_joints_o3d(joint_positions[joint], robot_mesh, max_threshold=0.3)
                        #     local_repulsive_joint_force = self.robot.world_to_local(repulsive_joint_force, force=True)

                        #     # Update joint angles using the jacobian
                        #     repulsive_torques, _ = self.robot.calculate_joint_torques(local_repulsive_joint_force, joint_id, joints_angles=self.path[i, 6:], J=J_joint)
                        #     # update the joint torques
                        #     joints_torques += self.k_repulsion_robot_joints * repulsive_torques
                        #     # print("repulsive robot: ", repulsive_torques)
                        #     # print("----------------------")

                        end_time = time.time()
                        # print("joint " + str(joint_id) + " time:", end_time - start_time)

                        # if i ==6 and joint=='joint2':
                        # print('i: ',iteration,'| rep: f[', repulsive_joint_force,' t[',repulsive_torques, ']| atract: f[', attractive_joint_force,'] t[',attractive_torques,']')
                        # Compute elastic forces to a safe config
                        if len(self.safe_config.keys()) > 0:
                            safe_joint_force = (self.safe_config[joint] - self.path[i, joint_index])
                            # # add the attractive force to the torques
                            joints_torques[joint_id-1] += self.k_safety_joints * safe_joint_force

                new_joints_config = self.path[i, 6:] + self.k_update_joints * joints_torques
                new_path[i, 6:] = self.bound_joints(new_joints_config)

                # Compute forces on the robot base
                attractive_force = self.compute_attractive_force(prev_pos[:3], current_pos[:3], next_pos[:3])
                repulsive_force = self.compute_repulsive_force_o3d(robot_mesh) # this repulsive force is calculated in the world frame
                repulsive_force = self.robot.world_to_local(repulsive_force,force=True)

                # Compute the orientation attraction force
                orientation_correction_roll = (wrap_angle(self.path[i - 1, 3] - self.path[i, 3])+wrap_angle(self.path[i + 1, 3] - self.path[i, 3]))
                orientation_correction_pitch = (wrap_angle(self.path[i - 1, 4] - self.path[i, 4])+wrap_angle(self.path[i + 1, 4] - self.path[i, 4]))
                orientation_correction_yaw = (wrap_angle(self.path[i - 1, 5] - self.path[i, 5])+wrap_angle(self.path[i + 1, 5] - self.path[i, 5]))

                # A larger orientation correction might suggest the robot needs to move differently to achieve this orientation.
                # position_adjustment_force = orientation_correction_yaw * np.array([np.cos(current_pos[3]), np.sin(current_pos[3])])


                # Total force on the robot base is a sum of attractive and repulsive forces
                total_force = self.k_attraction_base * attractive_force + self.k_repulsion_base * repulsive_force #+ self.k_position_from_orientation * position_adjustment_force

                # Project the force onto the robot's heading direction
                #heading_vector = np.array([np.cos(current_pos[2]), np.sin(current_pos[2])])

                # Update the position with the total force
                new_pos = current_pos[:3] + total_force

                # Compute the torque on the robot base from the total force
                base_torque_from_total_force = np.arctan2(total_force[1], total_force[0])

                # Update the orientation with the corrective force
                new_roll = current_pos[3] + self.k_orientation * orientation_correction_roll
                new_pitch = current_pos[4] + self.k_orientation * orientation_correction_pitch
                new_theta = current_pos[5] + self.k_orientation * orientation_correction_yaw + self.k_orientation_from_base * base_torque_from_total_force

                # Ensure new_theta is within -pi to pi
                new_roll= wrap_angle(new_roll)
                new_pitch = wrap_angle(new_pitch)
                new_theta = wrap_angle(new_theta)

                # Update the path
                new_path[i, :3] = new_pos
                new_path[i, 3] = new_roll
                new_path[i, 4] = new_pitch
                new_path[i, 5] = new_theta

                # Calculate the maximum change in this iteration
                max_change = max(max_change, np.linalg.norm(new_path[i, :] - self.path[i, :]))


            self.path = new_path
            self.history.append(new_path.copy())  # Store the path at each iteration for animation

            # If the maximum change is smaller than the threshold, we stop early
            if max_change < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations with max change {max_change}")
                break
            print("Iteration " + str(iteration) + " | maximum change: " + str(max_change))

        return self.path_to_dict(self.path)

    def bound_joints(self, values):
        for i, q in enumerate(values):
            if q < self.robot.joints_limits['q'+str(i+1)][0]:
                values[i] = self.robot.joints_limits['q'+str(i+1)][0]
            elif q > self.robot.joints_limits['q'+str(i+1)][1]:
                values[i] = self.robot.joints_limits['q'+str(i+1)][1]
        return values

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
        radius = 0.32
        for _ in range(len(self.history[0])):
            circle = plt.Circle((0, 0), radius, color='blue', fill=False, lw=2)
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

    def path_from_dict(self, path):
        """
        Converts a path given as an array of dictionaries to a list of lists.
        Each dictionary should contain x, y, z, roll, pitch, yaw, and joint angles q1 to qN.
        
        Args:
            path (list of dict): The input path as a list of dictionaries.
        
        Returns:
            list of list: The converted path as a list of lists.
        """
        # Initialize the resulting path as a list of lists
        converted_path = []
        
        # Determine the joint angle keys by analyzing the first dictionary
        if not path:
            raise ValueError("The path is empty.")

        for waypoint in path:
            # Extract x, y, z, roll, pitch, yaw, and joint angles in the correct order
            point = [
                waypoint['x'],
                waypoint['y'],
                waypoint['z'],
                waypoint['roll'],
                waypoint['pitch'],
                waypoint['yaw'],
            ]
            joint_keys = sorted([key for key in waypoint.keys() if key.startswith('q')])
            point.extend(waypoint[joint] for joint in joint_keys)

            # Append the converted point to the path
            converted_path.append(point)

        return np.array(converted_path)

    def path_to_dict(self, path):
        """
        Converts a path given as a list of lists back into an array of dictionaries.
        Each list should contain x, y, z, roll, pitch, yaw, and joint angles q1 to qN.
        
        Args:
            path (numpy.ndarray or list of list): The input path as a list of lists.
            num_joints (int): The number of joint angles (q1 to qN) in the path.
        
        Returns:
            list of dict: The converted path as an array of dictionaries.
        """
        # Define the fixed keys for the first six elements
        base_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        # Generate the joint keys dynamically
        num_joints = len(path[0])-len(base_keys)
        joint_keys = [f'q{i + 1}' for i in range(num_joints)]
        all_keys = base_keys + joint_keys

        converted_path = []
        for waypoint in path:
            # Create a dictionary by mapping keys to values
            waypoint_dict = {key: value for key, value in zip(all_keys, waypoint)}
            converted_path.append(waypoint_dict)

        return converted_path

####

# Example usage of ElasticBandP    lanner with a robot and environment
if __name__ == '__main__':
    # rospy.init_node('elastic_band_planner_node')

    # def joint_states_callback(msg):
    #         # Extract the first 7 joint positions
    #     joint_positions = np.asarray(msg.position[:7])
    #     # Update the robot joints
    #     ep.robot.set_joint_angles(joint_positions)

    # ep = ElasticBandPlanner(k_attraction_base=k_attraction_base, k_repulsion_base=k_repulsion_base, k_attraction_joints=k_attraction_joints,
    #                         k_repulsion_joints=k_repulsion_joints, k_update_joints=k_update_joints, k_orientation=k_orientation, k_orientation_from_base=k_orientation_from_base,
    #                         k_safety_joints=k_safety_joints, obstacle_threshold=obstacle_threshold)

    # # initial_path = ep.robot.interpolate_path(n=num_samples, start_pose=start, goal_pose=goal)
    # # initial_path = robot.interpolate_path(n=10)
    # # ep.update_path(initial_path, iterations=500,convergence_threshold=5e-3)

    # # Animate the path evolution
    # # ep.animate_path_evolution(time_interval=100)

    # # angles = ep.robot.compute_inverse_kinematics(-0.15,0.1,0.63, chain='joint2')
    # # print(angles)
    # # Subscribe to the /joint_states topic
    # rospy.Subscriber('/joint_states', JointState, joint_states_callback)
    # rospy.sleep(2.0)
    # print(ep.robot.joints_angles)
    # start_time = time.time()
    # positions = ep.robot.get_arm_endpoints(local_frame=True)
    # end_time = time.time()
    # print("Forward kinematics calculation time:", end_time - start_time)
    # print(positions)
    
    # # Spin the ROS node
    # rospy.spin()
    point = np.array([1.0, 0.0, 0.0])
        # Create a sphere mesh with a tiny radius (e.g., 0.1)
    start_time = time.time()

    min_corner, max_corner = create_bounding_box_corners(point,1.5)
    bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corner, max_bound=max_corner)
    end_time = time.time()

    print("Time taken:", end_time - start_time)
    bb.color=np.array([0.0,0.0,0.0])
    print(bb.volume())