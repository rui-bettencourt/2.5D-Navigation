import numpy as np
import rospy
import tf
linear = False
if linear:
    from mpc_linear import MPC
else:
    from mpc import MPC
import math
import json
from functions import Auxiliary
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time


class MPCLocalPlanner(object):
    def __init__(self):
        # begin node
        rospy.init_node('mpc_local_planner', anonymous=True)

        # variables
        self.hz                     = 10
        self.rate                   = rospy.Rate(self.hz)              # TODO: make this an argument of launch file
        self.real_path              = []                          # Current path
        self.base_pose              = None
        self.joints_configuration   = None
        self.control_robot          = True
        ############# CONFIG ############
        self.N                      = 5 # number of control intervals
        self.T                      = 1.0
        self.threshold_pos          = 0.2
        self.threshold_ori          = 0.2


        # Classes
        self.fs                     = Auxiliary()
        self.mpc                    = MPC(self.T,self.N)

        if self.control_robot:
            # Subscriptions
            self.pose_subs              = rospy.Subscriber('/ground_truth_odom', Odometry, self.posecb)
            self.joints_subs            = rospy.Subscriber('/joint_states', JointState, self.jointscb)

            # Publishers
            self.base_cmd_vel_pub       = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=5)
            self.arm_cmd_pub            = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=5)

    def run(self):
        #####################################################################################################
        with open("/home/rui/pcl_ws/src/mesh_nav/data/test_path.json", "r") as file:
            path = json.load(file)
        # if not self.control_robot:
        state = path[0]
        state['vx']= 0.0
        state['vyaw']= 0.0
        # self.fs.show_world(init=[state['x'], state['y']], gpath=path, title="Initial plot",qs=True)
        #####################################################################################################

        iteration = 0
        wp = 0
        targets = []
        #########
        state = path[0]
        time_start = time.time()
        self.mpc.solve_mpc(state, path[1])
        print("time: ", time.time()-time_start)

        while not rospy.is_shutdown():
            if not self.control_robot or (self.base_pose is not None and self.joints_configuration is not None): # check if the robot already received msgs for state estimation
                #get current state:
                if self.control_robot:
                    state = {**self.base_pose, **self.joints_configuration}
                    # state['x']+=path[0]['x']
                    # state['y']+=path[0]['y']
                    self.real_path.append(state)
                # get current wp index TODO: IMPROVE THIS
                wp = self.get_next_closest_pose(state, path, wp)
                targets.append(path[wp])
                print('goal: ',wp)

                if self.mpc.solve_mpc(state, path[wp]):
                    if not self.control_robot:
                        state = self.fs.simulate_step(dt=self.mpc.dt, state=state, control=self.mpc.get_control(i=0), configs=self.mpc.configs)
                        self.real_path.append(state)
                        self.fs.update_world(self.mpc, title="MPC#" + str(iteration) + " cost: " + str(self.mpc.cost), path=self.real_path)
                    else:
                        control = self.mpc.get_control_pose(i=1)
                        self.control_robot_pubs(control)
                        # self.fs.update_world(self.mpc, title="MPC#" + str(iteration) + " cost: " + str(self.mpc.cost), path=self.real_path, goal=path[wp],linear=linear)
                        print(control)
                else:
                    # self.fs.update_world(self.mpc, title="MPC#" + str(iteration) + " failed iter ", converged=False, path=self.real_path, goal=path[wp])
                    rospy.logerr('MPC could not solve current iteration')

                if math.sqrt((state['x'] - path[-1]['x'])**2 + (state['y'] - path[-1]['y'])**2) < self.threshold_pos:
                    break
                # print('Error qs: ', state['q1']-control['q1'], state['q2']-control['q2'], state['q3']-control['q3'])
                iteration += 1
                if self.control_robot:
                    self.rate.sleep()

        rospy.loginfo("Reached the goal")
        if self.control_robot:
            with open("/home/rui/pcl_ws/src/mesh_nav/data/tests/mpc_path.json", "w") as file:
                json.dump(self.real_path, file, indent=4)
            with open("/home/rui/pcl_ws/src/mesh_nav/data/tests/mpc_path_targets.json", "w") as file:
                json.dump(targets, file, indent=4)
            self.unsuscribe_topics()

    def unsuscribe_topics(self):
        self.pose_subs.unregister()
        self.joints_subs.unregister()

    def get_next_closest_pose(self, state, path, last_goal_index):
        """
        Finds the second closest goal pose ahead in the path.

        Parameters:
            state (dict): The current state with at least 'x' and 'y' keys.
            path (list of dicts): The list of waypoints, each containing at least 'x' and 'y'.

        Returns:
            int: The index of the second closest pose ahead in the path.
        """
        if not path:
            return None  # Return None if path is empty

        x, y = state['x'], state['y']

        # Compute distances and store as (index, distance) tuples
        distances = [(wp['x'] - x) ** 2 + (wp['y'] - y) ** 2 for wp in path]
        
        if last_goal_index==(len(path)-1) or (last_goal_index>0 and distances[last_goal_index-1]<distances[last_goal_index]):
            return last_goal_index
        else:
            return last_goal_index + 1

    def control_robot_pubs(self, control):
        cmd_vel = Twist(linear=Vector3(x=control['vx']), angular=Vector3(z=control['vyaw']))

        joint_traj = JointTrajectory(header=Header(stamp=rospy.Time.now()),
                                     joint_names=['arm_1_joint','arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'],
                                     points=[JointTrajectoryPoint(positions=[control['q1'], control['q2'], control['q3'], control['q4'], control['q5'], control['q6'], control['q7']],
                                                                  time_from_start=rospy.Duration(1))])#1/self.hz))])

        self.base_cmd_vel_pub.publish(cmd_vel)
        self.arm_cmd_pub.publish(joint_traj)

    def posecb(self, msg):
        euler = tf.transformations.euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.base_pose = {'x':      msg.pose.pose.position.x,
                          'y':      msg.pose.pose.position.y,
                          'z':      msg.pose.pose.position.z,
                          'roll':   euler[0],
                          'pitch':  euler[1],
                          'yaw':    euler[2],
                          'vx':     msg.twist.twist.linear.x,
                          'vyaw':   msg.twist.twist.angular.z}

    def jointscb(self, msg):
        self.joints_configuration = {'q1':  msg.position[0],
                                     'q2':  msg.position[1],
                                     'q3':  msg.position[2],
                                     'q4':  msg.position[3],
                                     'q5':  msg.position[4],
                                     'q6':  msg.position[5],
                                     'q7':  msg.position[6]}


if __name__ == '__main__':
    mlp = MPCLocalPlanner()
    mlp.run()    