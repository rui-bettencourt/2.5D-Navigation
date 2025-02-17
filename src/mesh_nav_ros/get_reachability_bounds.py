#!/usr/bin/env python
# import ros and other libraries
import rospy
import numpy as np
from copy import deepcopy


# import other created files
from robot_mesh_state import RobotMeshState
from collision_detections import *
from robot_kinematics import RobotKinematics


class Reachability(object):
    def __init__(self):
        # begin node
        rospy.init_node('reachability', anonymous=True)

        # variables
        self.path = '/home/rui/ds/testbed/'     # path for files, change this to save in the package
        self.rate = rospy.Rate(10)              # TODO: make this an argument of launch file

        # create classes needed for navigation
        # self.Meshes = RobotMeshState()
        self.robot_kinematics = RobotKinematics()
        self.RM = RobotMeshState(robot_kinematics=self.robot_kinematics)
        self.joint_names = ['base_link','arm_1_joint','arm_2_joint','arm_3_joint','arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint']
        self.step_size = 0.1

    def run(self):
        jl = self.RM.robot_kinematics.joints_limits
        new_jl = deepcopy(jl)
        step = 0.5
        q7=0
        for q1 in np.arange(jl['q1'][0], jl['q1'][1], step):
            for q2 in np.arange(jl['q2'][0], jl['q2'][1], step):
                for q3 in np.arange(jl['q3'][0], jl['q3'][1], step):
                    for q4 in np.arange(jl['q4'][0], jl['q4'][1], step):
                        for q5 in np.arange(jl['q5'][0], jl['q5'][1], step):
                            for q6 in np.arange(jl['q6'][0], jl['q6'][1], step):
                                config = {'x': 0.0, 'y': 0.0, 'z': 0.0,'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5, 'q6': q6, 'q7': q7}

                                full_mesh = self.RM.update_robot_arm_bbs(config, move_base = False, local_frame = False)

                                body = {link_name: m for link_name, m in full_mesh.items() if link_name not in self.RM.not_body}
                                arm = {link_name: m for link_name, m in full_mesh.items() if link_name in self.RM.not_body[1:]}

                                # mesh= self.RM.convert_bbs_to_mesh(conf_bb)
                                body_mesh = o3d.geometry.TriangleMesh()
                                for mesh in body.values():
                                    body_mesh += mesh

                                arm_mesh = o3d.geometry.TriangleMesh()
                                for mesh in arm.values():
                                    arm_mesh += mesh

                                if is_colliding_o3d(body_mesh, arm_mesh):
                                    print('Collision:  ', config)
                                    with open('/home/rui/pcl_ws/src/mesh_nav/data/collision_configurations.txt', 'a') as collision_file:
                                        collision_file.write(str(config) + '\n')
                                else:
                                    print('Safe')
                                    with open('/home/rui/pcl_ws/src/mesh_nav/data/safe_configurations.txt', 'a') as safe_file:
                                        safe_file.write(str(config) + '\n')

                                    # body_mesh.paint_uniform_color([1.0,0,0])
                                    # body_mesh.compute_vertex_normals()
                                    # arm_mesh.paint_uniform_color([0.0,0,1])
                                    # arm_mesh.compute_vertex_normals()
                                    # o3d.visualization.draw_geometries([body_mesh,arm_mesh])
                                if rospy.is_shutdown():
                                    exit()


if __name__ == '__main__':
    rb = Reachability()
    rb.run()
