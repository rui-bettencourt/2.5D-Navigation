from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from urdf_parser_py.urdf import URDF
import numpy as np
import json

def create_ikpy_chain_from_urdf(urdf_file, base_links, end_link):
    """
    Creates an IKPy chain from a URDF file, including only specified joints.
    """

    # Load the chain from URDF file, specifying the base and end link
    chain = Chain.from_urdf_file(
        urdf_file=urdf_file,
        base_elements=base_links,
        last_link_vector=[0, 0, 0],  # Adjust this if you need to define the vector to the end effector
        base_element_type="link",
        # active_links_mask=active_links_mask,
        name="arm_chain"
    )
    # Print all links in the created IKPy chain for verification
    print("\nIKPy Chain Links:")
    for link in chain.links:
        print(f" - {link.name}")
    end_link_index = None

    for i, link in enumerate(chain.links):
        if link.name == end_link:
            end_link_index = i
            break
    if end_link_index is None:
        raise ValueError(f"End link '{end_link}' not found in the chain.")
    final_chain =Chain(name='Sub-chain to Joint1', links=chain.links[:end_link_index+1])

    #     # Create static base and end-effector links
    # static_base_link = URDFLink(
    #             name="StaticBaseLink",
    #             origin_translation=[0.093, 0.014, 0.639],  # Translation between base_link and arm_1_link
    #             origin_orientation=[0,0,-1.57], #[0, 0, np.deg2rad(-90)],                 # Orientation of joint 1
    #             rotation=[0, 0, 0]
    #         )
    # static_end_link = URDFLink(
    #             name="StaticEndEffectorLink",
    #             origin_translation=[0.0, 0.0, 0.028],  # Translation between base_link and arm_1_link
    #             origin_orientation=[0,0,0], #[0, 0, np.deg2rad(-90)],                 # Orientation of joint 1
    #             rotation=[0, 0, 0]
    #         )

    # # Add the static links to the chain
    # final_chain.links.insert(0, static_base_link)  # Prepend base link
    # final_chain.links.append(static_end_link)     # Append end-effector link

    # Print all links in the created IKPy chain for verification
    print("\nIKPy Chain Links:")
    for link in final_chain.links:
        print(f" - {link.name}")

    return final_chain



def calculate_jacobian(chain, joint_positions):
    """
    Calculates the Jacobian matrix for each joint in the IKPy chain.
    """
    jacobian_matrix = chain.jacobian(joint_positions)
    return jacobian_matrix

def get_arm_endpoints(joints_angles, chain, local_frame=False):
    # Forward kinematics using the chain
    # angles = np.append(self.joints_angles,[0.0])
    fk_results = chain.forward_kinematics(joints_angles, full_kinematics=True)
    # Extract the (x, y) positions of the arm's first joint and the end effector
    # The position of all joints
    x1, y1, z1 = fk_results[3][0:3, 3]  # Position of the joints
    x2, y2, z2 = fk_results[4][0:3, 3]
    x3, y3, z3 = fk_results[5][0:3, 3]
    x4, y4, z4 = fk_results[6][0:3, 3]
    x5, y5, z5 = fk_results[7][0:3, 3]
    x6, y6, z6 = fk_results[8][0:3, 3]

    # The position of the end effector is in the final link
    x7, y7, z7 = fk_results[9][0:3, 3]  # Position of the end effector (arm2 endpoint) in the robot frame


    joints = {'joint1': np.asarray([x1,y1,z1]), 'joint2': np.asarray([x2,y2,z2]), 'joint3': np.asarray([x3,y3,z3]), 'joint4': np.asarray([x4,y4,z4]), 'joint5': np.asarray([x5, y5,z5]), 'joint6': np.asarray([x6,y6,z6]), 'joint7': np.asarray([x7,y7,z7])}  # Local frame coordinates of the joints
    return joints

def export_ikpy_chain_to_json_custom(ikpy_chain, file_path):
    """
    Custom function to export an IKPy chain to a JSON file.

    :param ikpy_chain: The IKPy Chain object to export.
    :param file_path: Path to save the JSON file.
    """
    # Create a dictionary representation of the chain
    chain_data = {
        "name": ikpy_chain.name,
        "links": [
            {
                "name": link.name,
                "translation_vector": link.translation_vector.tolist() if hasattr(link, "translation_vector") else None,
                "orientation_matrix": link.orientation_matrix.tolist() if hasattr(link, "orientation_matrix") else None,
                "bounds": link.bounds if hasattr(link, "bounds") else None,
            }
            for link in ikpy_chain.links
        ],
    }

    # Write to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(chain_data, json_file, indent=4)

    print(f"IKPy chain successfully exported to {file_path}")

def main():
    # Specify the URDF file path, base and end links, and relevant joint names
    urdf_file_path = '/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster.urdf'
    base_link = ["base_link"]
    end_link = "arm_7_joint"

    # Create the IKPy chain from the URDF
    arm_chain = create_ikpy_chain_from_urdf(urdf_file_path, base_link, end_link)
    # Save the IKPy chain to a file
    export_ikpy_chain_to_json_custom(arm_chain,'chains/tiago_chain.json')
    print("IKPy chain saved successfully.")

    # Example joint positions (replace with actual joint angles)
    joint_positions = [0.0,0.0,0.0,9.60664004e-02, -5.72965486e-02 ,-1.03480334e-04 , 2.13989422e-02, -4.66338874e-05 , 1.05742071e-03, -3.46437429e-04]  # Initialize positions for each active joint
    print(get_arm_endpoints(joint_positions,arm_chain))
    # # Calculate and print the Jacobian
    # jacobian_matrix = calculate_jacobian(arm_chain, joint_positions)
    # print("Jacobian Matrix for each joint:")
    # print(jacobian_matrix)

if __name__ == "__main__":
    main()

