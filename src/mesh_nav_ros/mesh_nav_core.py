#!/usr/bin/env python
# import ros and other libraries
import rospy

# import other created files
from mesh_nav_ros.robot_mesh_state import RobotMeshState
from collision_detections import *
from graph_functions import GraphManager

class MeshNav(object):
    def __init__(self):
        # begin node
        rospy.init_node('mesh_nav', anonymous=True)

        # variables
        self.path = '/home/rui/ds/testbed/'     # path for files, change this to save in the package
        self.rate = rospy.Rate(10)              # TODO: make this an argument of launch file

        # create classes needed for navigation
        # self.Meshes = RobotMeshState()
        self.Graph = GraphManager(self.path+"traversablegroundgraph.pkl")

    def run(self):
        while not rospy.is_shutdown():
            # test path planning with random start and end
            start_coord = [-11.0, -8.0, 0.0]  # Replace with your start coordinates
            end_coord = [-2.0, -7.0, 0.0]    # Replace with your end coordinates
            self.Graph.plan(start_coord,end_coord)
            exit()
            self.rate.sleep()


if __name__ == '__main__':
    mn = MeshNav()
    mn.run()
