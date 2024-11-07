#!/usr/bin/env python
# import ros and other libraries
import rospy
import random

# import other created files
from mesh_nav_ros.robot_mesh_state import RobotMeshState
from collision_detections import *
from graph_functions import GraphManager

from collision_detections import *
from joints_explorer import PRM

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
        self.MC = RobotMeshState()

    def run(self):
        all_bbs = []
        obstacle_mesh = o3d.io.read_triangle_mesh(self.path + 'obstacles.ply')
        obstacle_mesh.paint_uniform_color([1.0, 0.72, 0.67]) # pink
        all_bbs.append(obstacle_mesh)
        while not rospy.is_shutdown():
            # test path planning with random start and end
            start_coord = [-11.0, -8.0, 0.0]  # Replace with your start coordinates
            end_coord = [-4.6, -8.5, 0.0]     # Replace with your end coordinates
            path = self.Graph.plan(start_coord,end_coord)
            # then use the code of moving the robot to move to each point in the path
            for pose in path:
                conf_bb = self.MC.simulate_move_joints(self.MC.robot_bbs, 'base_link', pose) # this is just moving the base to the pose. arm config is the online one

                mesh= self.MC.convert_bbs_to_mesh(conf_bb)

                bb = mesh.get_axis_aligned_bounding_box()
                bb.scale(2.0,bb.get_center())
                # simplify obstacle mesh
                obstacle_mesh_cropped = obstacle_mesh.crop(bb)
                obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_vertex_clustering(0.02)
                obstacle_mesh_cropped = obstacle_mesh_cropped.simplify_quadric_decimation(2000)

                if is_colliding_o3d(mesh, obstacle_mesh_cropped):
                    rand_color = [1.0,0,0]
                else:
                    rand_color = [0,random.uniform(0,1),random.uniform(0,1)]

                conf_bb = list(conf_bb.values())

                for bb in conf_bb:
                    bb.color=rand_color
                all_bbs.extend(conf_bb)


            o3d.visualization.draw_geometries(all_bbs)
            # check for collisions all points in the path
            
            # if there is collisions use some sort of elastic band to move away from the collision
            exit()
            self.rate.sleep()


if __name__ == '__main__':
    mn = MeshNav()
    mn.run()
