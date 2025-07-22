#!/usr/bin/env python
import rospy
import rospkg
import logging
import os
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from urdf_parser_py.urdf import URDF, Mesh
import open3d as o3d
import numpy as np
import tf
import tf.transformations
from copy import deepcopy

class Pose(object):
    def __init__(self, position=[0.0, 0.0, 0.0], orientation=0.0, joints={}):
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.theta = orientation
        self.joints = joints

    def get_full_pose(self):
        return [self.x, self.y, self.z, self.theta] + self.joints

    def get_position(self):
        return [self.x, self.y, self.z]

    def get_pose(self):
        return [self.x, self.y, self.z, self.theta]

    def update_pose(self, pose):
        self.x = pose[0]
        self.y = pose[1]
        self.z = pose[2]
        self.theta = pose[3]

    def update_joints(self, joints):
        self.joints = joints


class RobotMeshState(object):
    def __init__(self, robot_kinematics):
        try:
            # Load the URDF model
            self.robot = URDF.from_parameter_server()
        except Exception as e:
            rospy.logwarn(f"Failed to load urdf. Make sure this is a node and the robot/simulation is running")
            exit()

        #########Configs
        joint_states_topic = "/joint_states"
        pose_odom_topic = "/ground_truth_odom"
        joints = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']#, 'torso_lift_joint']
        self.not_body = ["arm_1_link", "arm_2_link", "arm_3_link","arm_4_link", "arm_5_link", "arm_6_link", "arm_6_link",'gripper_link', 'gripper_right_finger_link', 'gripper_left_finger_link']
        robot_base_link = 'base_link'
        ##############

        ## variables
        self.robot_kinematics = robot_kinematics
        self.robot_mesh = None
        self.robot_mesh_dict = {}
        self.og_mesh_dict = {}
        self.robot_state = Pose()
        self.robot_bbs = {}
        self.robot_mesh_bb = self.simplify_robot_mesh()
        self.body_bbs = {link_name: bb for link_name, bb in self.robot_bbs.items() if link_name not in self.not_body}
        # the white list structure is {'joint_to_be_connected': [ 'parent_joint', 'extra joints between parent and matrix', 4x4 numpy transformation]}
        self.whitelist_joints_connections = {'gripper_link': ['arm_6_link', ['arm_7_link'],
                                                              [[ 7.77156117e-16,  1.57209315e-16, -1.00000000e+00,  6.73000000e-02],
                                                               [ 5.82248438e-04, -9.99999830e-01, -1.56775634e-16, -2.08166817e-17],
                                                               [-9.99999830e-01, -5.82248438e-04, -5.97262241e-16,  2.42861287e-17],
                                                               [ 0.0,  0.0,  0.0,  1.0]]] ,
                                             'gripper_right_finger_link':['arm_6_link', ['arm_7_link'],
                                                                          [[6.661338147750939e-16, 3.142017895862992e-16, -1.0, 0.06730000000000003],
                                                                           [0.0005520746142439676, -0.9999998476067986, -3.1387652893455353e-16, 2.4849196345148872e-05],
                                                                           [-0.9999998476067986, -0.0005520746142439676, -6.996106347282826e-16, -0.04501056907376662],
                                                                           [0.0, 0.0, 0.0, 1.0]]],
                                             'gripper_left_finger_link':['arm_6_link', ['arm_7_link'],
                                                                         [[1.1102230246251565e-16, 5.421010862427522e-20, -1.0000000000000002, 0.0673],
                                                                          [0.0005496891015726701, -0.9999998489209347, 5.421010862427522e-20, -2.1783354286175963e-05],
                                                                          [-0.9999998489209345, -0.0005496891015726701, -5.071648814610036e-16, 0.03962849351173512],
                                                                          [0.0, 0.0, 0.0, 1.0]]] }
        self.base_link = robot_base_link
        self.robot_joints = joints
        self.eliminate_nested_bounding_boxes()
        self.robot_joint_positions = None

        ## Subscriptions later change this to be configurable on launch file
        rospy.Subscriber(joint_states_topic, JointState, self.joint_states_callback, queue_size = 1)
        rospy.Subscriber(pose_odom_topic, Odometry, self.odom_callback, queue_size = 1)

    def resolve_package_uri(self, uri):
        if uri.startswith("package://"):
            package_name = uri[len("package://"):].split('/')[0]
            relative_path = uri[len(f"package://{package_name}/"):]
            rospack = rospkg.RosPack()
            package_path = rospack.get_path(package_name)
            return f"{package_path}/{relative_path}"
        return uri

    def load_mesh_from_geometry(self, geometry):
        unsupported_formats = ['.dae']
        if isinstance(geometry, Mesh):
            mesh_path = self.resolve_package_uri(geometry.filename)
            _, ext = os.path.splitext(mesh_path)
            ext = ext.lower()  # Normalize the extension to lowercase for comparison
            if ext not in unsupported_formats:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                return mesh
        # Handle other geometry types (Box, Cylinder, etc.) if necessary
        return None

    def update_robot_pose_and_convert_to_mesh(self):
        meshes = []
        listener = tf.TransformListener()
        for link in self.robot.links:
            if link.name == self.robot.get_root():
                continue

            try:
                (trans, rot) = listener.lookupTransform('/base_link', link.name, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            for visual in link.visuals:
                mesh = self.load_mesh_from_geometry(visual.geometry)
                if mesh:
                    transformation = np.eye(4)
                    transformation[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]
                    transformation[:3, 3] = np.array(trans)
                    mesh.transform(transformation)
                    meshes.append(mesh)

        if meshes:
            combined_mesh = meshes[0]
            for mesh in meshes[1:]:
                combined_mesh += mesh
            return combined_mesh
        return None

    def simplify_robot_mesh(self):
        meshes = []
        listener = tf.TransformListener()
        for link in self.robot.links:

            if link.name == self.robot.get_root():
                continue

            try:
                (trans, rot) = listener.lookupTransform('/base_link', link.name, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            for visual in link.visuals:
                mesh = self.load_mesh_from_geometry(visual.geometry)
                if mesh and len(mesh.vertices) > 0:
                    self.og_mesh_dict[link.name] = deepcopy(mesh).simplify_vertex_clustering(0.05)
                    transformation = np.eye(4)
                    transformation[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]
                    transformation[:3, 3] = np.array(trans)
                    mesh.transform(transformation)
                    self.robot_bbs[link.name] = mesh.get_minimal_oriented_bounding_box()
                    self.robot_mesh_dict[link.name] = mesh.simplify_vertex_clustering(0.05)
                    meshes.append(mesh)

        if meshes:
            combined_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(meshes[0].get_minimal_oriented_bounding_box())
            for mesh in meshes[1:]:
                combined_mesh += o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(mesh.get_minimal_oriented_bounding_box())
            return combined_mesh
        rospy.logerr("Could not simplify robot mesh!")
        return None

    def update_robot_arm_bbs(self, configuration, move_base = True, local_frame = True):
        # calculate bbs for a certain joint configuration and base_position, if in world frame
        # i'm going to try with the bb but i might need to do this on the mesh
        # if bbs is None:
        #     bbs = self.robot_bbs
        # new_bbs = (deepcopy(bbs))
        new_mesh = (deepcopy(self.robot_mesh_dict))
        og_meshes = deepcopy(self.og_mesh_dict)

        # move whole robot to global frame if move_base is true
        # if move_base:
        #     pose_array = np.array([configuration['x'], configuration['y'], configuration['z'], configuration['roll'], configuration['pitch'], configuration['yaw']])
        #     new_bbs = self.transform_whole_robot(new_bbs, pose_array)

        # #apply forward kinematics to obtain positions of joints
        # if not local_frame:
        #     self.robot_kinematics.set_robot_pose(x=configuration['x'], y=configuration['y'], z=configuration['z'], roll=configuration['roll'], pitch=configuration['pitch'], yaw=configuration['yaw'])

        if not local_frame:
            # get matrix to converto robot frame to local frame
            robot_to_world = self.robot_kinematics.create_transformation_matrix(configuration['x'], configuration['y'], configuration['z'], configuration['roll'], configuration['pitch'], configuration['yaw'])

        joint_angles_array = []
        for key in configuration.keys():
            if key.startswith('q'):
                joint_angles_array.append(configuration[key])
        if self.robot_kinematics.dof>0:
            self.robot_kinematics.set_joint_angles(joint_angles_array)
            fk_results = self.robot_kinematics.forward_kinematics(joint_angles_array)

        for joint in new_mesh.keys():
            #if joint is part of the configured movable joints, then move it

            if joint.replace('link','joint') in self.robot_joints and self.robot_kinematics.dof>0:
                # transformation_fk = fk_results[joint]
                transformation_fk = np.squeeze(fk_results[joint].cpu().get_matrix().numpy(),axis=0)
                if not local_frame:
                    transformation_fk = np.dot(robot_to_world, transformation_fk)

                #trans and rot from joint to base_link: get from forward kinematics
                mesh = og_meshes[joint]
                mesh.transform(transformation_fk)
                new_mesh[joint] = mesh

            elif move_base and not local_frame and joint not in self.whitelist_joints_connections.keys():
                mesh = new_mesh[joint]
                mesh.transform(robot_to_world)
                new_mesh[joint] = mesh
            elif joint in self.whitelist_joints_connections.keys() and self.robot_kinematics.dof>0:
                parent_joint, extra_joints, manual_transform = self.whitelist_joints_connections[joint]
                manual_transform = np.array(manual_transform)

                # try to force move all manually added joints to a parent joint
                # transformation_fk = fk_results[parent_joint]
                transformation_fk = np.squeeze(fk_results[parent_joint].cpu().get_matrix().numpy(),axis=0)
                if len(extra_joints) >0:
                    extra_transform = np.squeeze(fk_results[extra_joints[0]].cpu().get_matrix().numpy(),axis=0)
                    # extra_transform =fk_results[extra_joints[0]]
                else:
                    extra_transform = np.eye(4)
                #TODO: THEN MAKE THIS GENERAL FOR SEVERAL EXTRA JOINTS
                if not local_frame:
                    transformation_fk = np.dot(robot_to_world, transformation_fk, extra_transform)
                    transformation_fk = np.dot(transformation_fk, manual_transform)
                else:
                    transformation_fk = np.dot(transformation_fk, extra_transform, manual_transform)

                new_mesh[joint] = og_meshes[joint].transform(transformation_fk)
        return new_mesh
            # for example, for the head we will have to make it connected to the torso and update it as well. maybe use whitelist for that

    def is_point_in_bounding_box(self, point, bounding_box):
        # Transform the point to the local frame of the bounding box
        rotation_matrix = bounding_box.R
        center = bounding_box.center
        point_local = np.linalg.inv(rotation_matrix) @ (point - center)
        min_bound = bounding_box.get_min_bound()
        max_bound = bounding_box.get_max_bound()
        return np.all(min_bound <= point_local) and np.all(point_local <= max_bound)

    def is_bounding_box_included(self, bb1, bb2):
        bb1_points = np.asarray(bb1.get_box_points())
        for point in bb1_points:
            if not self.is_point_in_bounding_box(point, bb2):
                return False
        return True

    def eliminate_nested_bounding_boxes(self):
        remaining_bbs = {}
        for name1, bb1 in self.robot_bbs.items():
            nested = False
            for name2, bb2 in self.robot_bbs.items():
                if name1 != name2 and self.is_bounding_box_included(bb1, bb2):
                    nested = True
                    break
            if not nested:
                remaining_bbs[name1] = bb1
        self.robot_bbs = remaining_bbs

    def get_joint_transformation(self, joint, position):
        # we need self.robot_joint_positions to be set
        while self.robot_joint_positions is None:
            rospy.logwarn("Joint positions not set")
            rospy.sleep(0.1)

        # Get joint origin (translation and rotation from URDF)
        origin_translation = np.array(joint.origin.xyz)
        origin_rotation = tf.transformations.quaternion_matrix(
            tf.transformations.quaternion_from_euler(*joint.origin.rpy)
        )[:3, :3]

        if joint.type == 'revolute' or joint.type == 'continuous':
            rotation_matrix = tf.transformations.quaternion_matrix(
                tf.transformations.quaternion_about_axis(position, joint.axis)
            )[:3, :3]
            translation_vector = np.zeros(3) #np.array(joint.origin.xyz)
        elif joint.type == 'prismatic':
            translation_vector = (position - self.robot_joint_positions.get(joint.name, 0)) * np.array(joint.axis)
            rotation_matrix = np.eye(3)
        else:
            rospy.logwarn(f"Joint type {joint.type} not supported")
            return None

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        return transformation_matrix, origin_translation

    def propagate_transformation(self, bbs, parent_name, transformation, center):
        # out_bbs = list(bbs.values())
        # out_bbs_list = []

        # for bb in out_bbs:
        #     bb.color=[1.0,0.0,0.0]
        # out_bbs_list.extend(out_bbs)
        # o3d.visualization.draw_geometries(out_bbs_list)
        if parent_name in bbs:
            # if the current joint has a saved mesh, update it
            if parent_name in bbs:
                bb = self.transform_obb(bbs[parent_name], transformation, center)
                bbs[parent_name] = bb

            # check all children of all parent nodes
            # check for predefined connections
            if parent_name in self.whitelist_joints_connections:
                child = self.whitelist_joints_connections[parent_name]
                child_joint = next((j for j in self.robot.joints if j.child == child), None)
                if child_joint:
                    self.propagate_transformation(bbs, child, transformation, center)
            else:
                # check for children in the urdf
                for child in [j.child for j in self.robot.joints if j.parent == parent_name]:
                    child_joint = next((j for j in self.robot.joints if j.child == child), None)

                    if child_joint:
                        self.propagate_transformation(bbs, child, transformation, center)

    def transform_whole_robot(self, bbs, positions):
        # Extract translation and yaw from position
        if len(positions) == 3:
            x, y, yaw = positions
            z = 0; roll = 0; pitch = 0
        elif len(positions) == 4:
            x,y,z,yaw = positions
            roll = 0; pitch = 0
        elif len(positions) == 6:
            x,y,z,roll,pitch,yaw = positions
        else:
            rospy.logerr("Invalid number of positions")
            return

        # Create a transformation matrix for base_link
        rotation_matrix = tf.transformations.quaternion_matrix(
            tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        )[:3, :3]

        translation_vector = np.array([x, y, z])

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        for bb_name in bbs.keys():
            bb = self.transform_obb(bbs[bb_name], transformation_matrix, bbs['base_link'].center)
            bbs[bb_name] = bb
        return bbs

    def transform_obb(self, bb, transformation, center):
        translation = transformation[:3, 3]
        rotation = transformation[:3, :3]
        bb.translate(translation, relative=True)
        bb.rotate(rotation, center=center)
        return bb

    def obb_to_mesh(self, obb):
        # Create a mesh box with the dimensions of the OBB
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=obb.extent[0], height=obb.extent[1], depth=obb.extent[2])
        
        # Transform the mesh box to match the orientation and position of the OBB
        mesh_box.translate(obb.center - obb.extent / 2)
        mesh_box.rotate(obb.R, center=obb.center)
        
        return mesh_box

    def convert_bbs_to_mesh(self,bbs):
        combined_mesh = o3d.geometry.TriangleMesh()
        for key, obb in bbs.items():
            mesh = self.obb_to_mesh(obb)
            combined_mesh += mesh
        return combined_mesh

    def simulate_move_joints(self, bbs_og, joint_names, positions):
        bbs = (deepcopy(bbs_og))
        if len(joint_names) != len(positions) and joint_names != 'base_link':
            rospy.logwarn("The number of joint names and positions must be the same")
            return bbs

        # If joint_name is 'base_link', position should be a list of [x, y, yaw]
        if joint_names == 'base_link':# and len(positions) in [3, 4]:
            self.transform_whole_robot(bbs, positions)
        else:
            for joint_name, position in zip(joint_names, positions):
                joint = next((j for j in self.robot.joints if j.name == joint_name), None)
                if joint is None:
                    rospy.logwarn(f"Joint {joint_name} not found")
                    continue


                joint_transformation, joint_origin = self.get_joint_transformation(joint, position)
                if joint_transformation is not None and joint.child in bbs.keys():
                    self.propagate_transformation(bbs, joint.child, joint_transformation,joint_origin) #bbs[joint.child].center)
        return bbs

    def get_volume_collision(self, obb, mesh):
        overlap = mesh.crop(obb)

        # if len(overlap.vertices) > 0:
        if not overlap.is_empty():
            overlap = overlap.compute_convex_hull()

            if overlap.is_watertight():
                vol = overlap.get_volume()
            else:
                print("Mesh could not be repaired to be watertight.")
        else:
            vol = 0
        return vol

    def joint_states_callback(self, msg):
        # get all joints in a dict
        self.robot_joint_positions = dict(zip(msg.name, msg.position))
        # get only the joints we can/want to control
        joints = {joint_name: position for joint_name, position in self.robot_joint_positions.items() if joint_name in self.robot_joints}
        # update the state of the robot
        self.robot_state.update_joints(joints)
        # full robot mesh
        self.robot_mesh = self.update_robot_pose_and_convert_to_mesh()
        # self.robot_mesh_bb = self.simplify_robot_mesh()
        #update only relevant joints
        # print("Number of vertices: " + str(len(self.robot_mesh.vertices)) + "; Number of triangles: " +str((self.robot_mesh.triangles)))

    def odom_callback(self, msg):
        """
        Callback function for the odometry message.

        Args:
            msg (Odometry): The odometry message containing the robot's pose.

        Returns:
            None
        """
        # Extract orientation quaternion
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        # Convert quaternion to Euler angles
        yaw = tf.transformations.euler_from_quaternion(quaternion)[2]

        pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, yaw]
        self.robot_state.update_pose(pose)

    def save_mesh(self, filename):
        while not rospy.is_shutdown():
            if self.robot_mesh is not None:
                o3d.io.write_triangle_mesh(filename, self.robot_mesh)
                
            rospy.Rate(1.0).sleep()

if __name__ == '__main__':
    from robot_kinematics import RobotKinematics
    robot_kinematics = RobotKinematics()
    rospy.init_node('debug_mesh', anonymous=True)
    path = '/home/rui/ds/testbed/'
    mc = RobotMeshState(robot_kinematics)
    start = {'x': -11.0, 'y': -8.0, 'z': 0.0,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'q1': 0.2, 'q2': -1.34, 'q3': -0.2, 'q4': 1.94, 'q5': -1.57, 'q6': 1.37, 'q7': 0.0}
    meshes = mc.update_robot_arm_bbs(start,move_base = True, local_frame = False)
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes.values():
        combined_mesh += mesh
    combined_mesh.paint_uniform_color([1.0,0.0,0.0])
    combined_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([combined_mesh])
    # mc.save_mesh(path+'test.ply')
    rospy.spin()