import numpy as np
import json
import time
from roboticstoolbox import ERobot

# ROBOT
class RobotKinematics:
    def __init__(self, x=0, y=0, z=0, roll=0.0, pitch=0.0, yaw=0.0, joints_angles=None):
        if joints_angles is None:
            joints_angles = [0.0] * 7

        self.x = x
        self.y = y
        self.z = z
        self.joints_angles = np.asarray(joints_angles)  # Joint angles
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch
        self.joints_limits = {}

        # Load the robot model from URDF
        urdf_file = '/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster_pybullet.urdf'
        self.robot = ERobot.URDF(urdf_file)

        blacklisted_joints = ['torso_lift_link']
        # Assuming self.robot is an instance of ERobot
        self.num_links = self.robot.n
        self.joints_names = {}
        for i, link in enumerate(self.robot.links):
            if link.isjoint and link.name not in blacklisted_joints:
                self.joints_names[i] = link.name
            # print(f"Joint Name: {link.name}")


        # Define the end-effector link
        self.ee_link = "arm_7_link"
        self.dof = 7

        # Extract joint limits
        for i, lname in enumerate(self.joints_names.values()):
            link=self.robot.link_dict[lname]
            tag = f'q{i+1}'
            self.joints_limits[tag] = [link.qlim[0], link.qlim[1]]

    def forward_kinematics(self, joints_angles):
        angles = np.append((self.num_links-7) * [0.0], joints_angles)

        fks = self.robot.fkine_all(angles)

        ps = {}

        for i,fk in enumerate(fks):
            if i in self.joints_names.keys():
                ps[self.joints_names[i]]=fk.A
        return ps

    def get_arm_endpoints(self, local_frame=False, pose=None, config=None):
        # Forward kinematics using the chain
        if config is None:
            angles = np.append((self.num_links-7) * [0.0], self.joints_angles)
        else:
            angles = np.append((self.num_links-7) * [0.0], config)
        fks = self.robot.fkine_all(angles)
        ps = []

        for i,fk in enumerate(fks):
            if i in self.joints_names.keys():
                ps.append(fk.A[:3,3])

        if local_frame:
            joints = {'q1': ps[0], 'q2': ps[1], 'q3': ps[2], 'q4': ps[3], 'q5': ps[4], 'q6': ps[5], 'q7': ps[6]}  # Local frame coordinates of the joints
        elif pose is None:# make this use z as well
            joints = {'q1': self.local_to_world(ps[0]), 'q2': self.local_to_world(ps[1]), 'q3': self.local_to_world(ps[2]), 'q4': self.local_to_world(ps[3]), 'q5': self.local_to_world(ps[4]), 'q6': self.local_to_world(ps[5]), 'q7': self.local_to_world(ps[6])}
        else:
            joints = {'q1': self.local_to_world(ps[0],pose=pose), 'q2': self.local_to_world(ps[1],pose=pose), 'q3': self.local_to_world(ps[2],pose=pose), 'q4': self.local_to_world(ps[3],pose=pose), 'q5': self.local_to_world(ps[4],pose=pose), 'q6': self.local_to_world(ps[5],pose=pose), 'q7': self.local_to_world(ps[6],pose=pose)}
        return joints


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

        J = self.robot.jacob0(joints_angles, end='arm_'+str(joint)+'_link')
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
    c = a.get_arm_endpoints(config=[0.0]*7)
    print(c)
    b = a.calculate_jacobians(joint=7,joints_angles=[0.0]*7)
    print(b)
    print(a.joints_limits)
    print(a.forward_kinematics([0.0]*7))