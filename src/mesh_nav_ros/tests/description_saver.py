#!/usr/bin/env python3

import rospy

def save_robot_description(file_path):
    # Initialize a ROS node (anonymous=True allows multiple nodes without conflict)
    rospy.init_node('save_robot_description_node', anonymous=True)

    # Retrieve the 'robot_description' parameter
    try:
        robot_description = rospy.get_param('robot_description')
    except KeyError:
        rospy.logerr("Parameter 'robot_description' not found on the parameter server.")
        return

    # Write the URDF content to a text file
    with open(file_path, 'w') as file:
        file.write(robot_description)
    rospy.loginfo(f"Successfully saved 'robot_description' to {file_path}")

if __name__ == "__main__":
    # Specify the output file path
    output_file = "robot_description.txt"
    
    # Call the function to save the description
    save_robot_description(output_file)
