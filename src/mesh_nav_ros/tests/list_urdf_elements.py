from urdf_parser_py.urdf import URDF

def list_urdf_elements(urdf_file):
    # Load the URDF file
    robot = URDF.from_xml_file(urdf_file)
    
    # List all links
    print("Links in URDF:")
    for link in robot.links:
        print(f" - {link.name}")
    
    # List all joints
    print("\nJoints in URDF:")
    for joint in robot.joints:
        print(f" - {joint.name}, type: {joint.joint_type}, parent: {joint.parent}, child: {joint.child}")

if __name__ == "__main__":
    urdf_file_path = '/home/rui/socrob_ws/src/isr_tiago/simulation/mbot_simulation_environments/robots/tiago_ouster.urdf'
    list_urdf_elements(urdf_file_path)