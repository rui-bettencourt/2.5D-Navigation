#!/usr/bin/env python
import rospy
import rospkg
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF, Mesh
import open3d as o3d
import numpy as np
import tf
from copy import deepcopy

class RobotMeshState(object):
    def __init__(self):
        try:
            # Redirect stderr to suppress warnings from Open3D
            # stderr_fd = sys.stderr.fileno()
            # devnull_fd = os.open(os.devnull, os.O_WRONLY)
            # os.dup2(devnull_fd, stderr_fd)

            # Load the URDF model
            self.robot = URDF.from_parameter_server()
            # Restore stderr
            # os.dup2(stderr_fd, devnull_fd)
            # os.close(devnull_fd)
        except Exception as e:
            rospy.logwarn(f"Failed to load urdf. Make sure this is a node and the robot/simulation is running")
            exit()

        ## Subscriptions
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, queue_size = 1)

        ## variables
        self.robot_mesh = None
        self.robot_bbs = {}
        self.robot_mesh_bb = self.simplify_robot_mesh()
        self.whitelist_joints_connections = {'arm_6_link': 'gripper_link'}
        self.base_link = 'base_link'
        self.eliminate_nested_bounding_boxes()
        self.robot_joint_positions = None

    def resolve_package_uri(self, uri):
        if uri.startswith("package://"):
            package_name = uri[len("package://"):].split('/')[0]
            relative_path = uri[len(f"package://{package_name}/"):]
            rospack = rospkg.RosPack()
            package_path = rospack.get_path(package_name)
            return f"{package_path}/{relative_path}"
        return uri

    def load_mesh_from_geometry(self, geometry):
        if isinstance(geometry, Mesh):
            mesh_path = self.resolve_package_uri(geometry.filename)
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
                    transformation = np.eye(4)
                    transformation[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]
                    transformation[:3, 3] = np.array(trans)
                    mesh.transform(transformation)
                    self.robot_bbs[link.name] = mesh.get_minimal_oriented_bounding_box()
                    meshes.append(mesh)

        if meshes:
            combined_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(meshes[0].get_minimal_oriented_bounding_box())
            for mesh in meshes[1:]:
                combined_mesh += o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(mesh.get_minimal_oriented_bounding_box())
            return combined_mesh
        rospy.logerr("Could not simplify robot mesh!")
        return None

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

    # def simulate_move_joints(self, bbs, joint_names, positions):
    #     if len(joint_names) != len(positions):
    #         rospy.logwarn("The number of joint names and positions must be the same")
    #         return bbs

    #     for joint_name, position in zip(joint_names, positions):
    #         # Find the joint in the URDF
    #         joint = next((j for j in self.robot.joints if j.name == joint_name), None)
    #         if joint is None:
    #             rospy.logwarn(f"Joint {joint_name} not found")
    #             continue

    #         # Compute the transformation for the new joint position
    #         if joint.type == 'revolute' or joint.type == 'continuous':
    #             # Rotation around the joint axis
    #             rotation_matrix = tf.transformations.quaternion_matrix(
    #                 tf.transformations.quaternion_about_axis(position, joint.axis)
    #             )[:3, :3]
    #             translation_vector = np.array(joint.origin.xyz)
    #         elif joint.type == 'prismatic':
    #             # Translation along the joint axis
    #             translation_vector = np.array(joint.origin.xyz) + position * np.array(joint.axis)
    #             rotation_matrix = np.eye(3)
    #         else:
    #             rospy.logwarn(f"Joint type {joint.type} not supported")
    #             continue

    #         # Apply the transformation to the bounding box
    #         if joint.child in bbs:
    #             bb = bbs[joint.child]

    #             # Apply translation and rotation separately
    #             bb.translate(translation_vector, relative=False)
    #             bb.rotate(rotation_matrix, center=bb.center)
                
    #             bbs[joint.child] = bb
    #             rospy.loginfo(f"Moved joint {joint_name} bounding box to new position")
    #         else:
    #             rospy.logwarn(f"Bounding box for joint {joint.child} not found")
        
    #     return bbs

    def get_joint_transformation(self, joint, position):
        # we need self.robot_joint_positions to be set
        while self.robot_joint_positions is None:
            rospy.logwarn("Joint positions not set")
            rospy.sleep(0.1)

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

        return transformation_matrix

    def propagate_transformation(self, bbs, parent_name, transformation, center):
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
            z = 0
        elif len(positions) == 4:
            x,y,z,yaw = positions
        else:
            rospy.logerr("Invalid number of positions")
            return

        # Create a transformation matrix for base_link
        rotation_matrix = tf.transformations.quaternion_matrix(
            tf.transformations.quaternion_from_euler(0, 0, yaw)
        )[:3, :3]

        translation_vector = np.array([x, y, z])

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        for bb_name in bbs.keys():
            bb = self.transform_obb(bbs[bb_name], transformation_matrix, bbs['base_link'].center)
            bbs[bb_name] = bb

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
        if joint_names == 'base_link' and len(positions) in [3, 4]:
            self.transform_whole_robot(bbs, positions)
        else:
            for joint_name, position in zip(joint_names, positions):
                joint = next((j for j in self.robot.joints if j.name == joint_name), None)
                if joint is None:
                    rospy.logwarn(f"Joint {joint_name} not found")
                    continue


                joint_transformation = self.get_joint_transformation(joint, position)
                if joint_transformation is not None:
                    self.propagate_transformation(bbs, joint.child, joint_transformation,bbs[joint.child].center)
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
        self.robot_joint_positions = dict(zip(msg.name, msg.position))
        # full robot mesh
        self.robot_mesh = self.update_robot_pose_and_convert_to_mesh()
        #update only relevant joints
        # print("Number of vertices: " + str(len(self.robot_mesh.vertices)) + "; Number of triangles: " +str((self.robot_mesh.triangles)))

    def save_mesh(self, filename):
        while not rospy.is_shutdown():
            if self.robot_mesh is not None:
                o3d.io.write_triangle_mesh(filename, self.robot_mesh)
                
            rospy.Rate(1.0).sleep()

if __name__ == '__main__':
    rospy.init_node('debug_mesh', anonymous=True)
    path = '/home/rui/ds/testbed/'
    mc = RobotMeshState()
    mc.save_mesh(path+'test.ply')