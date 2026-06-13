
import torch
import pytorch_kinematics as pk
import numpy as np
import json
import time


# Check if CUDA is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("Using GPU for computation.")
    print("Device type:", torch.cuda.get_device_name(device))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    # Set the device to CPU
    device = torch.device("cpu")
    print("GPU not available. Using CPU for computation.")


#####
# ROBOT
class RobotKinematics:
    def __init__(self, x=0, y=0, z=0, roll=0.0,pitch=0.0,yaw=0.0, joints_angles=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
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
        ee_link = "arm_7_link"
        self.dof = 7
        dtype = torch.float64
        ######################################################################################

        # Define the kinematic chain using ikpy
        self.joint7_chain = pk.build_serial_chain_from_urdf(open(urdf_file).read(), ee_link)
        # self.joint7_chain.print_tree()
        ubs = self.joint7_chain.high.numpy()
        lbs = self.joint7_chain.low.numpy()
        self.num_links = len(ubs)
        for i, link in enumerate(self.joint7_chain.frame_to_idx.keys()):
            # for a different robot maybe change these rules
            if 'arm' in link:
                tag = 'q' + link.split('_')[1]
                ub = ubs[self.joint7_chain.joint_indices[i]]
                lb = lbs[self.joint7_chain.joint_indices[i]]
                self.joints_limits[tag] = [lb, ub]
        # save joints limits to json file
        # Convert all NumPy arrays and floats to Python lists and floats
        json_joints_limits = {key: list(map(float, value)) for key, value in self.joints_limits.items()}
        with open("/home/dolores/tiago_ws/src/full_body_nav_mpc/submodules/2.5D-Navigation/data/joints_limits.json", "w") as file:
                json.dump(json_joints_limits, file, indent=4)

        # Create all joints chains
        self.joint6_chain = pk.SerialChain(self.joint7_chain, "arm_6_link", "base_footprint")
        self.joint5_chain = pk.SerialChain(self.joint7_chain, "arm_5_link", "base_footprint")
        self.joint4_chain = pk.SerialChain(self.joint7_chain, "arm_4_link", "base_footprint")
        self.joint3_chain = pk.SerialChain(self.joint7_chain, "arm_3_link", "base_footprint")
        self.joint2_chain = pk.SerialChain(self.joint7_chain, "arm_2_link", "base_footprint")
        self.joint1_chain = pk.SerialChain(self.joint7_chain, "arm_1_link", "base_footprint")
        # convert all chains to the gpu if one is available
        self.joint7_chain = self.joint7_chain.to(dtype=dtype, device=device)
        self.joint6_chain = self.joint6_chain.to(dtype=dtype, device=device)
        self.joint5_chain = self.joint5_chain.to(dtype=dtype, device=device)
        self.joint4_chain = self.joint4_chain.to(dtype=dtype, device=device)
        self.joint3_chain = self.joint3_chain.to(dtype=dtype, device=device)
        self.joint2_chain = self.joint2_chain.to(dtype=dtype, device=device)
        self.joint1_chain = self.joint1_chain.to(dtype=dtype, device=device)




    def forward_kinematics(self, joints_angles):
        angles = np.append((self.num_links-7) * [0.0], joints_angles)
        angles = torch.tensor(angles, dtype=torch.float64).to(device)
        fk_results = self.joint7_chain.forward_kinematics(angles, end_only=False)
        return fk_results

    # def get_arm_endpoints(self, local_frame=False):
    #     # Forward kinematics using the chain
    #     angles = np.append((self.num_links-7) * [0.0],self.joints_angles)
    #     angles = torch.tensor(angles, dtype=torch.float64).to(device)
    #     fk_results = self.joint7_chain.forward_kinematics(angles, end_only=False)
    #     # Extract the (x, y) positions of the arm's first joint and the end effector
    #     # The position of all joints
    #     p1 = np.squeeze(fk_results['arm_1_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
    #     p2 = np.squeeze(fk_results['arm_2_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
    #     p3 = np.squeeze(fk_results['arm_3_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
    #     p4 = np.squeeze(fk_results['arm_4_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
    #     p5 = np.squeeze(fk_results['arm_5_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
    #     p6 = np.squeeze(fk_results['arm_6_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
    #     p7 = np.squeeze(fk_results['arm_7_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)

    #     if local_frame:
    #         joints = {'q1': p1, 'q2': p2, 'q3': p3, 'q4': p4, 'q5': p5, 'q6': p6, 'q7': p7}  # Local frame coordinates of the joints
            
    #     else:# make this use z as well
    #         joints = {'q1': self.local_to_world(p1), 'q2': self.local_to_world(p2), 'q3': self.local_to_world(p3), 'q4': self.local_to_world(p4), 'q5': self.local_to_world(p5), 'q6': self.local_to_world(p6), 'q7': self.local_to_world(p7)}
    #     return joints

    def get_arm_endpoints(self, local_frame=False, pose=None, config=None):
        # Forward kinematics using the chain
        if config is None:
            angles = np.append((self.num_links-7) * [0.0],self.joints_angles)
        else:
            angles = np.append((self.num_links-7) * [0.0],config)
        angles = torch.tensor(angles, dtype=torch.float64).to(device)
        fk_results = self.joint7_chain.forward_kinematics(angles, end_only=False)
        # Extract the (x, y) positions of the arm's first joint and the end effector
        # The position of all joints
        p1 = np.squeeze(fk_results['arm_1_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p2 = np.squeeze(fk_results['arm_2_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p3 = np.squeeze(fk_results['arm_3_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p4 = np.squeeze(fk_results['arm_4_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p5 = np.squeeze(fk_results['arm_5_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p6 = np.squeeze(fk_results['arm_6_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p7 = np.squeeze(fk_results['arm_7_link'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)

        if local_frame:
            joints = {'q1': p1, 'q2': p2, 'q3': p3, 'q4': p4, 'q5': p5, 'q6': p6, 'q7': p7}  # Local frame coordinates of the joints
        elif pose is None:# make this use z as well
            joints = {'q1': self.local_to_world(p1), 'q2': self.local_to_world(p2), 'q3': self.local_to_world(p3), 'q4': self.local_to_world(p4), 'q5': self.local_to_world(p5), 'q6': self.local_to_world(p6), 'q7': self.local_to_world(p7)}
        else:
            joints = {'q1': self.local_to_world(p1,pose=pose), 'q2': self.local_to_world(p2,pose=pose), 'q3': self.local_to_world(p3,pose=pose), 'q4': self.local_to_world(p4,pose=pose), 'q5': self.local_to_world(p5,pose=pose), 'q6': self.local_to_world(p6,pose=pose), 'q7': self.local_to_world(p7,pose=pose)}
        return joints

    # def compute_inverse_kinematics(self, target_x, target_y, target_z, chain='joint7'):
    #     # Target position in 2D space
    #     target_position = [target_x, target_y, target_z]
        
    #     # Solve inverse kinematics
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     if chain=='joint7':
    #         joint_angles = self.joint7_chain.inverse_kinematics(target_position)
    #         self.joint7_chain.plot(joint_angles,ax)
    #     elif chain == 'joint6':
    #         joint_angles = self.joint6_chain.inverse_kinematics(target_position)
    #     elif chain == 'joint5':
    #         joint_angles = self.joint5_chain.inverse_kinematics(target_position)
    #     elif chain == 'joint4':
    #         joint_angles = self.joint4_chain.inverse_kinematics(target_position)
    #     elif chain == 'joint3':
    #         joint_angles = self.joint3_chain.inverse_kinematics(target_position)
    #     elif chain == 'joint2':
    #         joint_angles = self.joint2_chain.inverse_kinematics(target_position)
    #         self.joint2_chain.plot(joint_angles,ax)
    #     elif chain == 'joint1':
    #         joint_angles = self.joint1_chain.inverse_kinematics(target_position)
    #     else:
    #         print("Wrong chain specified. Use 'joint1' or 'joint2'.")

    #     # Show the plot
    #     plt.show()
    #     return joint_angles

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
                self.joints_angles[i] = np.clip(a, self.joints_limits['q'+str(i+1)][0], self.joints_limits['q'+str(i+1)][1])
        else:
            self.joints_angles[joint-1] = np.clip(angle, self.joints_limits['q'+str(joint)][0], self.joints_limits['q'+str(joint)][1])

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

    def local_to_world(self, vector, force=False, pose=None):
        """
        Converts a 3D force vector from the local frame to the world frame based on the robot's pose.
        
        Args:
        - vector (np.array): vector in the world frame, [x, y, z].
        - force (bool): whether it is a force vector or a position vector
        
        Returns:
        - np.array: Transformed force vector in the local frame.
        """
        # Generate the homogeneous transformation matrix
        if pose is None:
            T = self.create_transformation_matrix(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        else:
            T = self.create_transformation_matrix(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
        
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

    def calculate_jacobians(self, joint, joints_angles=None):
        if joints_angles is None:
            joints_angles = np.append((self.num_links-7) * [0.0],self.joints_angles)
        else:
            joints_angles = np.append((self.num_links-7) * [0.0],joints_angles)
        size_ja = len(joints_angles)
        if joint==7:
            joints_angles7 = torch.tensor(joints_angles, dtype=torch.float64).to(device)
            J = np.squeeze(self.joint7_chain.jacobian(joints_angles7).cpu().numpy(), axis=0)
        elif joint==6:
            joints_angles6 = torch.tensor(joints_angles[:size_ja-1], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint6_chain.jacobian(joints_angles6).cpu().numpy(), axis=0)
        elif joint==5:
            joints_angles5 = torch.tensor(joints_angles[:size_ja-2], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint5_chain.jacobian(joints_angles5).cpu().numpy(), axis=0)
        elif joint==4:
            joints_angles4 = torch.tensor(joints_angles[:size_ja-3], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint4_chain.jacobian(joints_angles4).cpu().numpy(), axis=0)
        elif joint==3:
            joints_angles3 = torch.tensor(joints_angles[:size_ja-4], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint3_chain.jacobian(joints_angles3).cpu().numpy(), axis=0)
        elif joint==2:
            joints_angles2 = torch.tensor(joints_angles[:size_ja-5], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint2_chain.jacobian(joints_angles2).cpu().numpy(), axis=0)
        elif joint==1:
            joints_angles1 = torch.tensor(joints_angles[:size_ja-6], dtype=torch.float64).to(device)
            # Jacobian for the joints
            J = np.squeeze(self.joint1_chain.jacobian(joints_angles1).cpu().numpy(), axis=0)
        else:
            print('Joint not defined')
        return J

    def calculate_joint_torques(self, cartesian_forces, joint, joints_angles=None, J=None):
        # to save time, there is the possibility to receive the jacobian so it does not need to be calculated
        if J is None:
            J = self.calculate_jacobians(joint, joints_angles)

        # Transpose of the Jacobian
        J_T = np.transpose(J)


        # For now the cartesian forces only have x, y and z but the jacobian requires angular as well. we just add 0s
        cartesian_forces = np.concatenate([cartesian_forces, np.zeros(3)])

        # Calculate joint torques using τ = J^T * F
        joint_torques = np.dot(J_T, cartesian_forces)
        joints_not_used = (self.num_links-len(joint_torques)) * [0.0]

        complete_torques = np.append(joint_torques[1:],joints_not_used)
        return complete_torques, J

if __name__ == '__main__':
    a = RobotKinematics()
    t1 = time.time()
    c = a.get_arm_endpoints(local_frame=True, config=[0.0]*7)
    print("Python forward kinematics time: " + str((time.time()-t1)*1000) + " ms")
    print(c)
    t2 = time.time()
    # b = a.calculate_jacobians(joint=1,joints_angles=[0.0]*7)
    # b = a.calculate_jacobians(joint=2,joints_angles=[0.0]*7)
    # b = a.calculate_jacobians(joint=3,joints_angles=[0.0]*7)
    # b = a.calculate_jacobians(joint=4,joints_angles=[0.0]*7)
    # b = a.calculate_jacobians(joint=5,joints_angles=[0.0]*7)
    # b = a.calculate_jacobians(joint=6,joints_angles=[0.0]*7)
    b = a.calculate_jacobians(joint=1,joints_angles=[0.0]*7)
    print("Python jacobians time: " + str((time.time()-t2)*1000) + " ms")
    print(b)