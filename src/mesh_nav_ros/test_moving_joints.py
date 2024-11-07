#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading

class RobotMover:
    def __init__(self):
        rospy.init_node('robot_mover', anonymous=True)

        # Parameters
        self.distance = rospy.get_param('~distance', 4.0)  # Distance to move in meters
        self.joint_delay = rospy.get_param('~joint_delay', 3.0)  # Delay before joint movement starts in seconds
        self.velocity = rospy.get_param('~velocity', 0.8)  # Velocity of the robot base in m/s

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.torso_pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=10)
        self.arm_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)

        # Calculate times
        self.base_move_time = self.distance / self.velocity
        self.joint_move_time = self.base_move_time - self.joint_delay

        rospy.loginfo("Starting robot mover node")
        
        # Start threads
        base_thread = threading.Thread(target=self.move_base)
        joint_thread = threading.Thread(target=self.move_joints)

        base_thread.start()
        rospy.sleep(self.joint_delay)
        joint_thread.start()

        base_thread.join()
        joint_thread.join()

    def move_base(self):
        twist = Twist()
        twist.linear.x = self.velocity
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while rospy.Time.now() - start_time < rospy.Duration(self.base_move_time):
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop the robot
        twist.linear.x = 0
        self.cmd_vel_pub.publish(twist)

    def move_joints(self):
        torso_trajectory = JointTrajectory()
        arm_trajectory = JointTrajectory()

        # Fill in torso trajectory
        torso_trajectory.joint_names = ['torso_lift_joint']
        torso_point = JointTrajectoryPoint()
        torso_point.positions = [0.3]
        torso_point.time_from_start = rospy.Duration(self.joint_move_time)
        torso_trajectory.points.append(torso_point)

        # Fill in arm trajectory
        arm_trajectory.joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
        arm_point = JointTrajectoryPoint()
        arm_point.positions = [1.57, -0.07, -0.0, 0.04, 0.0, 0.0, 0.0]
        arm_point.time_from_start = rospy.Duration(self.joint_move_time)
        arm_trajectory.points.append(arm_point)

        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < rospy.Duration(self.joint_move_time):
            self.torso_pub.publish(torso_trajectory)
            self.arm_pub.publish(arm_trajectory)
            rate.sleep()

if __name__ == '__main__':
    try:
        RobotMover()
    except rospy.ROSInterruptException:
        pass
