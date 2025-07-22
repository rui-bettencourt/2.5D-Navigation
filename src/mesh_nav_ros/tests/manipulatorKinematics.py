from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import kdl_parser_py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ManipulatorKinematics:
    def __init__(self, links, base_link, end_link):
        # Load the URDF model
        self.robot = URDF.from_parameter_server()
        self.base_link = base_link
        self.links = links
        self.end_link = end_link

        # Build the KDL chain from URDF
        self.kdl_chain = self.create_kdl_chain()
        self.chain_info = self.extract_chain_info()
        for segment in self.chain_info:
            print(segment)
        # self.visualize_chain()

        # Initialize solvers
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.kdl_chain)
        self.jacobian_solver = kdl.ChainJntToJacSolver(self.kdl_chain)

    def create_kdl_chain(self):
        """
        Creates a KDL chain from the URDF model, including only joints specified in self.links.
        """
        if self.base_link not in [link.name for link in self.robot.links]:
                print(f"Base link '{self.base_link}' not found in URDF.")
        if self.end_link not in [link.name for link in self.robot.links]:
            print(f"End link '{self.end_link}' not found in URDF.")

        kdl_tree = kdl.Tree(self.base_link)
        link_map = {link.name: link for link in self.robot.links}
        joint_map = {joint.name: joint for joint in self.robot.joints}

        def add_joints_to_tree(current_link_name):
            # Recursively add joints to the KDL tree
            for joint_name, joint in joint_map.items():
                if joint.parent == current_link_name:
                    joint_name = joint.name.replace('joint','link')
                    # Check if this joint is in the self.links list
                    # if joint_name in self.links:
                        # Create KDL Joint
                    axis = joint.axis if joint.axis else [0, 0, 1]
                    axis_vector = kdl.Vector(*axis)

                    # # Determine the KDL joint type
                    # if joint.joint_type == 'revolute':
                    #     kdl_joint = kdl.Joint(joint.name, axis_vector, kdl.Joint.RotAxis)
                    # elif joint.joint_type == 'prismatic':
                    #     kdl_joint = kdl.Joint(joint.name, axis_vector, kdl.Joint.TransAxis)
                    # else:
                    #     # For fixed joints, set a None joint in KDL
                    #     kdl_joint = kdl.Joint(joint.name)#, kdl.Joint.Fixed)
                    if joint.joint_type == 'revolute':
                        kdl_joint = kdl.Joint(
                            joint_name,
                            kdl.Vector(0, 0, 0),  # Origin
                            axis_vector,
                            kdl.Joint.RotAxis
                        )
                    elif joint.joint_type == 'prismatic':
                        kdl_joint = kdl.Joint(
                            joint_name,
                            kdl.Vector(0, 0, 0),  # Origin
                            axis_vector,
                            kdl.Joint.TransAxis
                        )
                    else:
                        # For fixed joints, use the name-only constructor
                        kdl_joint = kdl.Joint(joint_name)

                    # Get the origin (offset) of the joint
                    xyz = joint.origin.xyz if joint.origin.xyz else [0, 0, 0]
                    rpy = joint.origin.rpy if joint.origin.rpy else [0, 0, 0]

                    # Convert RPY to a KDL rotation
                    rotation = kdl.Rotation.RPY(*rpy)
                    translation = kdl.Vector(*xyz)

                    # Create the KDL segment with the joint and frame
                    kdl_segment = kdl.Segment(
                        joint_name,
                        kdl_joint,
                        kdl.Frame(rotation, translation),
                        kdl.RigidBodyInertia()  # Optional inertia, can leave as default
                    )

                    # Add the segment to the KDL tree
                    kdl_tree.addSegment(kdl_segment, joint.parent.replace('joint','link'))

                    # Recursively add joints from the child link
                    add_joints_to_tree(joint.child)

        # Start building the chain from the base link
        add_joints_to_tree(self.base_link)
        
        # Extract the chain from base_link to end_link
        print(f"Number of segments in the KDL tree: {kdl_tree.getNrOfSegments()}")
        if not kdl_tree.getNrOfSegments():
            raise ValueError("KDL tree could not be constructed.")


        kdl_chain = kdl_tree.getChain(self.base_link, self.end_link)
        if kdl_chain.getNrOfSegments() == 0:
                print("Error: The KDL chain is empty.")
                print("URDF Joint Connections:")
                for joint in self.robot.joints:
                    print(f"Joint {joint.name}: Parent={joint.parent}, Child={joint.child}, Type={joint.joint_type}")

        else:
            print(f"KDL chain successfully created with {kdl_chain.getNrOfSegments()} segments.")

        return kdl_chain

    def forward_kinematics(self, joint_positions):
        """
        Calculate the forward kinematics for a given joint configuration
        """
        joint_array = kdl.JntArray(len(joint_positions))
        for i, pos in enumerate(joint_positions):
            joint_array[i] = pos

        end_effector_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_array, end_effector_frame)

        position = [end_effector_frame.p[0], end_effector_frame.p[1], end_effector_frame.p[2]]
        orientation = end_effector_frame.M.GetRPY()  # Roll, Pitch, Yaw

        return position, orientation

    def inverse_kinematics(self, desired_position, desired_orientation, initial_guess=None):
        """
        Calculate the inverse kinematics for a desired end-effector pose.
        """
        target_frame = kdl.Frame()
        target_frame.p = kdl.Vector(*desired_position)
        target_frame.M = kdl.Rotation.RPY(*desired_orientation)

        # Use an initial guess for the joint positions if provided, otherwise zeros
        joint_positions = kdl.JntArray(self.kdl_chain.getNrOfJoints())
        if initial_guess:
            for i, pos in enumerate(initial_guess):
                joint_positions[i] = pos

        solution = kdl.JntArray(self.kdl_chain.getNrOfJoints())
        ik_result = self.ik_solver.CartToJnt(joint_positions, target_frame, solution)

        if ik_result >= 0:
            return [solution[i] for i in range(solution.rows())]
        else:
            raise ValueError("Inverse kinematics solution not found")

    def calculate_jacobian(self, joint_positions):
        """
        Calculate the Jacobian matrix at a given joint configuration
        """
        joint_array = kdl.JntArray(len(joint_positions))
        for i, pos in enumerate(joint_positions):
            joint_array[i] = pos

        jacobian = kdl.Jacobian(self.kdl_chain.getNrOfJoints())
        self.jacobian_solver.JntToJac(joint_array, jacobian)

        # Convert the KDL Jacobian to a list of lists for easier use
        jacobian_matrix = [[jacobian[row, col] for col in range(jacobian.columns())] for row in range(jacobian.rows())]
        return jacobian_matrix

    def extract_chain_info(self):
        chain_info = []
        for i in range(self.kdl_chain.getNrOfSegments()):
            segment = self.kdl_chain.getSegment(i)
            joint = segment.getJoint()
            
            chain_info.append({
                'segment_name': segment.getName(),
                'joint_name': joint.getName(),
                'joint_type': joint.getType(),
                'axis': [joint.JointAxis().x(), joint.JointAxis().y(), joint.JointAxis().z()],
                'pose': {
                    'position': [segment.getFrameToTip().p[0],
                                segment.getFrameToTip().p[1],
                                segment.getFrameToTip().p[2]],
                    'orientation': segment.getFrameToTip().M.GetRPY()
                }
            })
        return chain_info

    def visualize_chain(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_coords, y_coords, z_coords = [0], [0], [0]  # Start from origin

        for segment in self.chain_info:
            position = segment['pose']['position']
            x_coords.append(position[0])
            y_coords.append(position[1])
            z_coords.append(position[2])

            # Optionally annotate joint/segment names
            ax.text(position[0], position[1], position[2], segment['joint_name'], color='red')

        # Plot the chain
        ax.plot(x_coords, y_coords, z_coords, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


# Example usage function to test the class methods
def example_usage():
    # Initialize kinematics for a sample robot (update paths and links as needed)
    base_link = "torso_lift_link"       # Replace with your robot's base link
    end_link = 'arm_7_link' # Replace with your end-effector link
    links = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']

    # Create an instance of ManipulatorKinematics
    kinematics = ManipulatorKinematics(links, base_link, end_link)

    # Define example joint positions (replace with real values as needed)
    joint_positions = [0.07, -0.027, 0.0, 0.01, 0.0, 0.0, 0.0] # Example joint configuration

    # Forward Kinematics
    position, orientation = kinematics.forward_kinematics(joint_positions)
    print("Forward Kinematics:")
    print("Position:", position)
    print("Orientation (RPY):", orientation)

    # Inverse Kinematics
    desired_position = [0.5, 0.2, 0.3]  # Target position for the end effector
    desired_orientation = [0.0, 1.57, 0.0]  # Target orientation in RPY format
    try:
        ik_solution = kinematics.inverse_kinematics(desired_position, desired_orientation)
        print("Inverse Kinematics Solution:", ik_solution)
    except ValueError as e:
        print(e)

    # Jacobian Calculation
    jacobian = kinematics.calculate_jacobian(joint_positions)
    print("Jacobian Matrix:")
    for row in jacobian:
        print(row)


# Run the example usage function if the script is executed directly
if __name__ == "__main__":
    example_usage()
