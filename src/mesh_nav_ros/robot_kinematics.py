
import torch
import pytorch_kinematics as pk
import numpy as np
import json
import time
import xml.etree.ElementTree as ET


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
        self._urdf_file = "/home/rui/mesh_nav_ws/src/REMANI-Planner/remani_planner/mm_config/meshes/ur5/ur5.urdf"
        ee_link = "mani_6"
        self.dof = 6
        dtype = torch.float64
        ######################################################################################

        # Define the kinematic chain using ikpy
        with open(self._urdf_file, "rb") as f:
            xml_bytes = f.read()
        self.joint6_chain = pk.build_serial_chain_from_urdf(xml_bytes, ee_link)
        # self.joint7_chain.print_tree()
        ubs = self.joint6_chain.high.numpy()
        lbs = self.joint6_chain.low.numpy()
        self.num_links = len(ubs)
        for i, link in enumerate(self.joint6_chain.frame_to_idx.keys()):
            # for a different robot maybe change these rules
            if 'mani' in link:
                tag = 'q' + link.split('_')[1]
                ub = ubs[self.joint6_chain.joint_indices[i]]
                lb = lbs[self.joint6_chain.joint_indices[i]]
                self.joints_limits[tag] = [lb, ub]
        # save joints limits to json file
        # Convert all NumPy arrays and floats to Python lists and floats
        json_joints_limits = {key: list(map(float, value)) for key, value in self.joints_limits.items()}
        # with open("/home/rui/pcl_ws/src/mesh_nav/data/joints_limits.json", "w") as file:
        #         json.dump(json_joints_limits, file, indent=4)

        # Create all joints chains
        self.joint5_chain = pk.SerialChain(self.joint6_chain, "mani_5", "mm_base")
        self.joint4_chain = pk.SerialChain(self.joint6_chain, "mani_4", "mm_base")
        self.joint3_chain = pk.SerialChain(self.joint6_chain, "mani_3", "mm_base")
        self.joint2_chain = pk.SerialChain(self.joint6_chain, "mani_2", "mm_base")
        self.joint1_chain = pk.SerialChain(self.joint6_chain, "mani_1", "mm_base")
        # convert all chains to the gpu if one is available
        self.joint6_chain = self.joint6_chain.to(dtype=dtype, device=device)
        self.joint5_chain = self.joint5_chain.to(dtype=dtype, device=device)
        self.joint4_chain = self.joint4_chain.to(dtype=dtype, device=device)
        self.joint3_chain = self.joint3_chain.to(dtype=dtype, device=device)
        self.joint2_chain = self.joint2_chain.to(dtype=dtype, device=device)
        self.joint1_chain = self.joint1_chain.to(dtype=dtype, device=device)
        
        # -------- dynamics (lightweight) ----------
        # Parse URDF inertials to estimate joint inertias (scalar about joint axis).
        # For URDFs where all joints rotate about z (axis="0 0 1"), we take Izz.
        # self._joint_inertias = self._parse_urdf_joint_inertias(self._urdf_file, default_inertia=1.0)
        # ---- parse URDF inertials & joint dynamics once ----
        self._link_inertial = self._parse_urdf_inertials(self._urdf_file)  # dict per link
        self._joint_damping = self._parse_urdf_joint_damping(self._urdf_file, default=0.0)  # np.ndarray (dof,)
        # Optional linear damping (per joint), default zero
        # self._joint_damping = np.zeros(self.dof, dtype=float)
        # If your pk jacobians are [linear; angular] (they are in your torque helper),
        # leave this True. If you later find it's [angular; linear], flip this to False.
        self._jac_linear_first = False




    def forward_kinematics(self, joints_angles):
        angles = np.append((self.num_links-6) * [0.0], joints_angles)
        angles = torch.tensor(angles, dtype=torch.float64).to(device)
        fk_results = self.joint6_chain.forward_kinematics(angles, end_only=False)
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
        fk_results = self.joint6_chain.forward_kinematics(angles, end_only=False)
        # Extract the (x, y) positions of the arm's first joint and the end effector
        # The position of all joints
        p1 = np.squeeze(fk_results['mani_1'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p2 = np.squeeze(fk_results['mani_2'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p3 = np.squeeze(fk_results['mani_3'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p4 = np.squeeze(fk_results['mani_4'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p5 = np.squeeze(fk_results['mani_5'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)
        p6 = np.squeeze(fk_results['mani_6'].cpu().get_matrix()[:, :3, 3].numpy(),axis=0)

        if local_frame:
            joints = {'q1': p1, 'q2': p2, 'q3': p3, 'q4': p4, 'q5': p5, 'q6': p6}  # Local frame coordinates of the joints
        elif pose is None:# make this use z as well
            joints = {'q1': self.local_to_world(p1), 'q2': self.local_to_world(p2), 'q3': self.local_to_world(p3), 'q4': self.local_to_world(p4), 'q5': self.local_to_world(p5), 'q6': self.local_to_world(p6)}
        else:
            joints = {'q1': self.local_to_world(p1,pose=pose), 'q2': self.local_to_world(p2,pose=pose), 'q3': self.local_to_world(p3,pose=pose), 'q4': self.local_to_world(p4,pose=pose), 'q5': self.local_to_world(p5,pose=pose), 'q6': self.local_to_world(p6,pose=pose)}
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
            joints_angles = self.joints_angles
        else:
            joints_angles = joints_angles
        size_ja = len(joints_angles)
        if joint==6:
            joints_angles6 = torch.tensor(joints_angles[:size_ja], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint6_chain.jacobian(joints_angles6).cpu().numpy(), axis=0)
        elif joint==5:
            joints_angles5 = torch.tensor(joints_angles[:size_ja-1], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint5_chain.jacobian(joints_angles5).cpu().numpy(), axis=0)
        elif joint==4:
            joints_angles4 = torch.tensor(joints_angles[:size_ja-2], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint4_chain.jacobian(joints_angles4).cpu().numpy(), axis=0)
        elif joint==3:
            joints_angles3 = torch.tensor(joints_angles[:size_ja-3], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint3_chain.jacobian(joints_angles3).cpu().numpy(), axis=0)
        elif joint==2:
            joints_angles2 = torch.tensor(joints_angles[:size_ja-4], dtype=torch.float64).to(device)
            J = np.squeeze(self.joint2_chain.jacobian(joints_angles2).cpu().numpy(), axis=0)
        elif joint==1:
            joints_angles1 = torch.tensor(joints_angles[:size_ja-5], dtype=torch.float64).to(device)
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


# # ----------------- DYNAMICS HELPERS -----------------
#     def _parse_urdf_joint_inertias(self, urdf_path: str, default_inertia: float = 1.0):
#         """
#         Attempt to read per-joint scalar inertias from the URDF.
#         Heuristic:
#           - For joint i, look at child link 'mani_i' inertial -> inertia matrix (Ixx,Iyy,Izz).
#           - If joint axis is z (typical in your URDF), use Izz.
#           - If missing, fall back to average of (Ixx,Iyy,Izz) if present, else default_inertia.
#         NOTE: This ignores offsets between joint origin and COM (parallel-axis), so treat as an estimate.
#         """
#         inertias = [default_inertia] * self.dof
#         try:
#             tree = ET.parse(urdf_path)
#             root = tree.getroot()
#         except Exception as e:
#             print(f"[RobotKinematics] URDF parse failed: {e}. Using defaults.")
#             return inertias

#         # URDF is <robot><link>... We’ll assume links named mani_1..mani_6 exist.
#         ns = ""  # URDF typically has no namespace
#         for i in range(1, self.dof + 1):
#             link_name = f"mani_{i}"
#             link_elem = None
#             for le in root.findall("link"):
#                 if le.get("name", "") == link_name:
#                     link_elem = le
#                     break
#             if link_elem is None:
#                 continue

#             inertial = link_elem.find("inertial")
#             if inertial is None:
#                 continue
#             inertia = inertial.find("inertia")
#             if inertia is None:
#                 continue

#             def getf(tag):
#                 v = inertia.get(tag, None)
#                 return float(v) if v is not None else None

#             Ixx, Iyy, Izz = getf("ixx"), getf("iyy"), getf("izz")
#             # Prefer Izz (assuming revolute about z); else average available diagonals
#             if Izz is not None and np.isfinite(Izz):
#                 inertias[i - 1] = float(Izz)
#             else:
#                 vals = [v for v in [Ixx, Iyy, Izz] if (v is not None and np.isfinite(v))]
#                 if vals:
#                     inertias[i - 1] = float(sum(vals) / len(vals))
#                 else:
#                     inertias[i - 1] = default_inertia

#         return inertias

#     def set_dynamics_params(self, joint_inertias=None, joint_damping=None):
#         """
#         Override estimated inertias/damping.
#         joint_inertias: iterable of length dof (scalars)
#         joint_damping:  iterable of length dof (scalars)
#         """
#         if joint_inertias is not None:
#             ji = np.asarray(list(joint_inertias), dtype=float).reshape(-1)
#             assert ji.size == self.dof, "joint_inertias must have length dof"
#             self._joint_inertias = ji.tolist()
#         if joint_damping is not None:
#             jd = np.asarray(list(joint_damping), dtype=float).reshape(-1)
#             assert jd.size == self.dof, "joint_damping must have length dof"
#             self._joint_damping = jd

#     def get_joint_inertias(self):
#         """Return current per-joint scalar inertias (list of length dof)."""
#         return list(self._joint_inertias)

#     def mass_matrix(self, q):
#         """
#         Lightweight mass matrix estimate.
#         Returns diagonal M(q) with the per-joint scalar inertias.
#         Signature matches what your stats script expects.
#         """
#         ji = np.asarray(self._joint_inertias, dtype=float)
#         return np.diag(ji)

#     def compute_torques(self, q, qdot, qddot):
#         """
#         Simple torque model τ = M(q) q̈ + B q̇
#         - M(q) is diagonal with joint inertias
#         - B is diagonal damping (configurable via set_dynamics_params)
#         This is NOT a full rigid-body inverse dynamics; it’s suitable
#         for energy/power estimates when full dynamics are unavailable.
#         """
#         q     = np.asarray(q, dtype=float).reshape(self.dof,)
#         qdot  = np.asarray(qdot, dtype=float).reshape(self.dof,)
#         qddot = np.asarray(qddot, dtype=float).reshape(self.dof,)

#         Mdiag = np.asarray(self._joint_inertias, dtype=float).reshape(self.dof,)
#         Bdiag = np.asarray(self._joint_damping,  dtype=float).reshape(self.dof,)
#         tau = Mdiag * qddot + Bdiag * qdot
#         return tau

    def _rpy_to_R(self, r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        return Rz @ Ry @ Rx

    def _skew(self, v):
        x, y, z = v
        return np.array([[0,-z, y],[z,0,-x],[-y,x,0]], dtype=float)

    def _parse_urdf_inertials(self, urdf_path):
        """
        Return dict:
        link_name -> {
            'm': float,
            'I_com_local': (3x3) inertia about COM expressed in the inertial frame,
            'R_link_inertial': (3x3) rotation from inertial frame -> link frame,
            'com_local': (3,) COM position in the link frame (from <inertial origin xyz>),
        }
        Assumes URDF inertial origin is at the COM (standard).
        """
        out = {}
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[RobotKinematics] URDF parse failed: {e}")
            return out

        for link in root.findall('link'):
            lname = link.get('name', '')
            inertial = link.find('inertial')
            if inertial is None:
                continue

            # mass
            m = 0.0
            mass_el = inertial.find('mass')
            if mass_el is not None and mass_el.get('value') is not None:
                try: m = float(mass_el.get('value'))
                except: m = 0.0

            # inertia (diagonal + off-diagonals)
            I_el = inertial.find('inertia')
            if I_el is None:
                continue
            def g(tag, default=0.0):
                v = I_el.get(tag)
                try: return float(v) if v is not None else default
                except: return default
            Ixx, Iyy, Izz = g('ixx'), g('iyy'), g('izz')
            Ixy, Ixz, Iyz = g('ixy'), g('ixz'), g('iyz')
            I_com_local = np.array([[Ixx, Ixy, Ixz],
                                    [Ixy, Iyy, Iyz],
                                    [Ixz, Iyz, Izz]], dtype=float)

            # inertial origin: position (COM) and orientation wrt link frame
            com_local = np.zeros(3, dtype=float)
            R_li = np.eye(3)
            origin = inertial.find('origin')
            if origin is not None:
                xyz = origin.get('xyz')
                rpy = origin.get('rpy')
                if xyz is not None:
                    parts = [float(s) for s in xyz.strip().split()]
                    if len(parts)==3: com_local = np.array(parts, dtype=float)
                if rpy is not None:
                    parts = [float(s) for s in rpy.strip().split()]
                    if len(parts)==3: R_li = self._rpy_to_R(parts[0], parts[1], parts[2])

            out[lname] = {
                'm': m,
                'I_com_local': I_com_local,
                'R_link_inertial': R_li,
                'com_local': com_local
            }
        return out

    def _parse_urdf_joint_damping(self, urdf_path, default=0.0):
        """
        Return np.ndarray (dof,) with joint damping from <joint><dynamics damping=.../>.
        Assumes joint names are 'joint1'..'joint6' as in your URDF snippet.
        Missing entries -> default.
        """
        B = np.full(self.dof, float(default), dtype=float)
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[RobotKinematics] URDF parse failed (damping): {e}")
            return B

        for j in root.findall('joint'):
            jname = j.get('name', '')
            dyn = j.find('dynamics')
            if dyn is None: continue
            damps = dyn.get('damping')
            if damps is None: continue
            try:
                dval = float(damps)
            except:
                continue
            # Map joint name "jointk" -> index k-1
            if jname.startswith('joint'):
                try:
                    k = int(jname.replace('joint', '')) - 1
                    if 0 <= k < self.dof:
                        B[k] = dval
                except:
                    pass
        return B

    def _pad_to_dof(self, Jsmall, upto):
        """Pad a (6×upto) jacobian to (6×dof) by adding right zeros."""
        J = np.zeros((6, self.dof), dtype=float)
        J[:, :upto] = Jsmall
        return J

    def _split_J(self, J):
        """Return (Jv, Jw) as (3×n, 3×n) according to row order."""
        if self._jac_linear_first:
            return J[0:3, :], J[3:6, :]
        else:
            return J[3:6, :], J[0:3, :]

    def mass_matrix(self, q):
        """
        High-fidelity joint-space inertia using link inertias and COM shifts:
        M = Σ_i ( m_i Jv_com^T Jv_com + Jw^T I_world Jw )
        where:
        Jv_com = Jv_link - [r_world]_x Jw_link
        I_world = (R_world_link * R_link_inertial) * I_com_local * (...)^T
        r_world = R_world_link * com_local
        Returns np.ndarray (dof×dof).
        """
        q = np.asarray(q, dtype=float).reshape(self.dof,)
        # FK to get each link pose
        fk = self.forward_kinematics(q.tolist())  # dict of pk transforms

        # Collect link names in order mani_1..mani_6
        link_names = [f"mani_{i}" for i in range(1, self.dof+1)]

        M = np.zeros((self.dof, self.dof), dtype=float)

        for i, lname in enumerate(link_names, start=1):
            # inertial params for this link
            L = self._link_inertial.get(lname, None)
            if L is None:
                continue
            m = float(L['m'])
            I_com_local = np.array(L['I_com_local'], dtype=float)
            R_li = np.array(L['R_link_inertial'], dtype=float)
            com_local = np.array(L['com_local'], dtype=float)

            # link world pose
            if lname not in fk:
                # fallback: skip if FK didn't give this link
                continue
            T = fk[lname].cpu().get_matrix().numpy() if hasattr(fk[lname], 'cpu') else fk[lname]
            T = np.squeeze(T, axis=0)  # (4×4)
            R_wl = T[:3, :3]  # world <- link

            # world COM offset and inertia
            r_world = R_wl @ com_local  # COM shift in world
            R_wi = R_wl @ R_li          # world <- inertial
            I_world = R_wi @ I_com_local @ R_wi.T

            # Jacobian for this link (6×i), then pad to 6×dof
            J_small = self.calculate_jacobians(joint=i, joints_angles=q.tolist())  # (6×i)
            J = self._pad_to_dof(J_small, i)
            Jv_link, Jw_link = self._split_J(J)  # (3×dof), (3×dof)

            # shift linear jacobian to COM: v_com = v_link + ω×r = v_link - [r]_x ω
            S_r = self._skew(r_world)
            Jv_com = Jv_link - S_r @ Jw_link

            # accumulate
            M += m * (Jv_com.T @ Jv_com) + (Jw_link.T @ I_world @ Jw_link)

        # make symmetric numerically
        M = 0.5 * (M + M.T)
        return M

    def effective_joint_inertia(self, q):
        """Return diag(M(q)) as a length-dof vector (effective inertia about each joint axis)."""
        M = self.mass_matrix(q)
        return np.diag(M)

    def compute_torques(self, q, qdot, qddot):
        """
        Torque model: τ = M(q) q̈ + B q̇
        Uses high-fidelity M(q) above and damping from URDF (if any).
        """
        q     = np.asarray(q, dtype=float).reshape(self.dof,)
        qdot  = np.asarray(qdot, dtype=float).reshape(self.dof,)
        qddot = np.asarray(qddot, dtype=float).reshape(self.dof,)

        M = self.mass_matrix(q)
        B = np.asarray(self._joint_damping, dtype=float).reshape(self.dof,)

        tau = M @ qddot + B * qdot
        return tau

    def get_joint_inertias(self, q=None):
        """
        If q provided: return effective inertia diag at configuration q (preferred).
        Else: return per-link scalar Izz at COM from URDF (rough constants).
        """
        if q is not None:
            return self.effective_joint_inertia(q)
        # Fallback “constants” from URDF (about link inertial axes) — not about joint axes.
        out = []
        for i in range(1, self.dof+1):
            lname = f"mani_{i}"
            L = self._link_inertial.get(lname, None)
            if L is None:
                out.append(1.0)
            else:
                I = L['I_com_local']
                out.append(float(I[2,2]))  # Izz in inertial frame
        return np.array(out, dtype=float)

# if __name__ == '__main__':
    # a = RobotKinematics()
    # t1 = time.time()
    # conf = [0.0]*6
    # conf[1]=-1.57
    # c = a.get_arm_endpoints(local_frame=True, config=conf)
    # print("Python forward kinematics time: " + str((time.time()-t1)*1000) + " ms")
    # print(c)
    # t2 = time.time()

# if __name__ == '__main__':
#     import numpy as np
#     from copy import deepcopy
#     np.set_printoptions(precision=4, suppress=True)

#     a = RobotKinematics()
#     print("\n=== RobotKinematics Dynamics Test ===")

#     # ---------------- helpers ----------------
#     def eig_psd_report(M):
#         # Report eigen min/max and symmetry error
#         symm_err = np.linalg.norm(M - M.T, ord='fro') / max(1.0, np.linalg.norm(M, ord='fro'))
#         w = np.linalg.eigvalsh(0.5*(M+M.T))
#         return float(symm_err), float(w.min()), float(w.max())

#     def compute_M_with_row_order(rk, q, linear_first=True):
#         old = rk._jac_linear_first
#         rk._jac_linear_first = bool(linear_first)
#         M = rk.mass_matrix(q)
#         rk._jac_linear_first = old
#         return M

#     # ---------------- test configurations ----------------
#     qs = []
#     qs.append(np.zeros(6))                                      # all zeros
#     qs.append(np.array([0.3, -1.1, 0.7, 1.2, -0.8, 0.4]))       # mid-range
#     qs.append(np.array([ 1.3, -2.4,  2.2, -1.6,  1.0, -0.7]))   # near limits (ensure inside bounds)
#     # clamp to limits
#     for i,q in enumerate(qs):
#         for j in range(6):
#             lb, ub = a.joints_limits[f'q{j+1}']
#             q[j] = np.clip(q[j], lb, ub)
#         qs[i] = q

#     # ---------------- 1) symmetry/PSD + jac row order sanity ----------------
#     print("\n[1] Mass matrix symmetry & PSD checks")
#     for k, q in enumerate(qs):
#         M_lin = compute_M_with_row_order(a, q, linear_first=True)
#         M_ang = compute_M_with_row_order(a, q, linear_first=False)

#         sym_lin, lam_min_lin, lam_max_lin = eig_psd_report(M_lin)
#         sym_ang, lam_min_ang, lam_max_ang = eig_psd_report(M_ang)

#         print(f"  Config {k}: q = {q}")
#         print(f"    linear-first : symm_err={sym_lin:.2e}, eig[min,max]=[{lam_min_lin:.3e},{lam_max_lin:.3e}]")
#         print(f"    angular-first: symm_err={sym_ang:.2e}, eig[min,max]=[{lam_min_ang:.3e},{lam_max_ang:.3e}]")

#         # Heuristic pick: prefer the variant with fewer/milder negative eigenvalues and lower symmetry error
#         chosen = "linear-first" if lam_min_lin >= lam_min_ang and sym_lin <= sym_ang else "angular-first"
#         print(f"    -> Suggested row order here: {chosen}")

#     print("\n   NOTE:")
#     print("   • Symmetry error should be near 0 (≤1e-8 typical).")
#     print("   • Eigen min should be ≥ 0 (tiny negatives ~ -1e-9 can be numerical).")
#     print("   • If angular-first consistently shows better PSD, set self._jac_linear_first=False in __init__.")

#     # ---------------- 2) scaling test: double inertials -> M ~ 2*M ----------------
#     print("\n[2] Scaling test (×2 masses & inertias)")
#     q = qs[1]
#     M1 = a.mass_matrix(q)
#     backup = deepcopy(a._link_inertial)
#     try:
#         for name,par in a._link_inertial.items():
#             par['m'] *= 2.0
#             par['I_com_local'] = 2.0 * par['I_com_local']
#         M2 = a.mass_matrix(q)
#         rel_err = np.linalg.norm(M2 - 2.0*M1, ord='fro') / max(1.0, np.linalg.norm(M1, ord='fro'))
#         print(f"  ‖M2 - 2*M1‖/‖M1‖ = {rel_err:.3e}  (expect ≈ 0)")
#     finally:
#         a._link_inertial = backup  # restore

#     # ---------------- 3) energy consistency experiment ----------------
#     print("\n[3] Energy consistency (work vs ΔKE) with τ = M(q) qddot (no damping)")
#     a._joint_damping = np.zeros(6)  # disable damping for this check

#     T = 2.0       # total duration [s]
#     dt = 0.01     # timestep [s]
#     t  = np.arange(0.0, T+1e-12, dt)
#     nT = len(t)

#     # Smooth small-amplitude sinusoid (stay away from hard nonlinearities)
#     A = np.array([0.2, 0.3, 0.25, 0.2, 0.15, 0.1])  # amplitudes [rad]
#     f = 0.5                                         # Hz
#     w = 2*np.pi*f

#     # Pick a center configuration to oscillate around (mid-range)
#     q0 = qs[1]
#     qbnd = np.array([a.joints_limits[f'q{i+1}'] for i in range(6)], dtype=float)

#     q   = np.zeros((nT,6))
#     qd  = np.zeros((nT,6))
#     qdd = np.zeros((nT,6))
#     for i,ti in enumerate(t):
#         q[i,:]  = q0 + A*np.sin(w*ti)
#         qd[i,:] = A*w*np.cos(w*ti)
#         qdd[i,:]= -A*w*w*np.sin(w*ti)
#         # clamp q into limits just in case
#         for j in range(6):
#             lb, ub = qbnd[j]
#             q[i,j] = np.clip(q[i,j], lb, ub)

#     # Compute work = ∫ τ^T qdot dt, and KE(t) = 1/2 qdot^T M qdot
#     work = 0.0
#     KE   = np.zeros(nT)
#     for i in range(nT):
#         M = a.mass_matrix(q[i,:])
#         KE[i] = 0.5 * float(qd[i,:].T @ (M @ qd[i,:]))
#         tau = M @ qdd[i,:]  # (no Coriolis/gravity/damping here by design)
#         work += float(np.dot(tau, qd[i,:])) * dt

#     dKE = KE[-1] - KE[0]
#     resid = work - dKE

#     print(f"  ∫τᵀq̇ dt = {work:.6f}   ΔKE = {dKE:.6f}   residual = {resid:.6f}")
#     print("   EXPECTATION:")
#     print("   • Because τ = M q̈ ignores Coriolis/centrifugal terms, residual ≠ 0.")
#     print("   • If amplitudes are small and dt is small, |residual| should be modest (not exploding).")

#     # ---------------- 4) quick effective inertia readout ----------------
#     print("\n[4] Effective joint inertia diag(M) at q0:")
#     Mq0 = a.mass_matrix(q0)
#     print("  diag(M(q0)) =", np.diag(Mq0))

#     print("\n=== Done. If something looks off, see notes above and adjust self._jac_linear_first. ===\n")

if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(precision=6, suppress=True)

    rk = RobotKinematics()
    print("\n=== Dynamics Debug ===")
    print("URDF path:", rk._urdf_file)

    # 0) Show what inertials we parsed
    print("\n[Inertials parsed per link] (mass [kg], COM [m])")
    if not hasattr(rk, "_link_inertial") or len(rk._link_inertial) == 0:
        print("  !! No inertials parsed. Check URDF path.")
    else:
        for i in range(0, 7):  # include mani_0 if available
            lname = f"mani_{i}"
            if lname in rk._link_inertial:
                L = rk._link_inertial[lname]
                m = L["m"]
                com = L["com_local"]
                print(f"  {lname:7s}: m={m:.6f}, com={com}")
        # also dump damping if we have it
        if hasattr(rk, "_joint_damping"):
            print("Joint damping (from URDF dynamics if present):", rk._joint_damping)

    # 1) Choose a reasonable config inside limits
    q = np.array([0.3, -1.1, 0.7, 1.2, -0.8, 0.4], dtype=float)
    for j in range(6):
        lb, ub = rk.joints_limits[f"q{j+1}"]
        q[j] = np.clip(q[j], lb, ub)
    print("\nTest configuration q:", q)

    # 2) Jacobian norms per link (should NOT be all zeros)
    print("\n[Jacobian Frobenius norms per link]")
    any_nonzero = False
    for i in range(1, 7):
        J = rk.calculate_jacobians(joint=i, joints_angles=q.tolist())  # (6 x i)
        if not isinstance(J, np.ndarray):
            J = np.array(J)
        # Handle shape quirks: make 2D
        if J.ndim == 1:
            J = J.reshape(6, 1)
        fnorm = np.linalg.norm(J, ord='fro')
        print(f"  link mani_{i}: shape={J.shape}, ||J||_F={fnorm:.6e}")
        if fnorm > 1e-10:
            any_nonzero = True

    if not any_nonzero:
        print("  !! All Jacobians are near zero. That would make M(q)=0. "
              "Check that calculate_jacobians() and pk.SerialChain are returning valid values.")

    # 3) Try both row orders to see which gives a sane (non-zero) M(q)
    def try_order(linear_first):
        old = getattr(rk, "_jac_linear_first", True)
        rk._jac_linear_first = bool(linear_first)
        try:
            M = rk.mass_matrix(q)
        except Exception as e:
            print(f"  mass_matrix failed (linear_first={linear_first}): {e}")
            M = np.zeros((6, 6))
        rk._jac_linear_first = old
        return M

    M_lin = try_order(True)
    M_ang = try_order(False)

    def summarize(M, tag):
        sym_err = np.linalg.norm(M - M.T, ord='fro') / max(1.0, np.linalg.norm(M, ord='fro'))
        w = np.linalg.eigvalsh(0.5 * (M + M.T))
        print(f"\n[{tag}] M(q) summary:")
        print("diag(M) =", np.diag(M))
        print(f"symmetry error = {sym_err:.3e}")
        print(f"eig min/max = {w.min():.3e} / {w.max():.3e}")

    summarize(M_lin, "linear-first rows assumed (Jv;Jw)")
    summarize(M_ang, "angular-first rows assumed (Jw;Jv)")

    # 4) Pick the better one and set the flag for future runs
    choose_ang = (np.linalg.norm(M_ang) > np.linalg.norm(M_lin))
    print("\nSuggested Jacobian row order:",
          "ANGULAR-FIRST ([Jw; Jv])" if choose_ang else "LINEAR-FIRST ([Jv; Jw])")
    print("Set rk._jac_linear_first =", (not choose_ang), "to force the chosen order (True = linear-first).")
