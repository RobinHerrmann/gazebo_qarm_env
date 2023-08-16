'''
Provides functionality to sent command-data to specific ros2_control 
controller types.

Author:  Robin Herrmann
Created: 2023.07.12
'''

import rclpy
from rclpy.node import Node
from rclpy import Parameter

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from rclpy.duration import Duration

import time
import math

# UNUSED
class JointEffortController(Node):
    # Sends command-data to effort_controller
    # currently not used
    def __init__(self, controller_name, joint_names, type):
        self.controller_name = controller_name
        self.joint_names = joint_names
        self.type = type

        super().__init__('{}_publisher'.format(self.controller_name))
        self.set_parameters([Parameter('use_sim_time', value = 1)])

        self.publisher = self.create_publisher(Float64MultiArray, 
                            '/{}/commands'.format(self.controller_name), 10)

    def command(self, data, duration):
        # Init message
        msg = Float64MultiArray()

        for i in range(len(data)):
            data[i] = float(data[i])

        msg.data = data
        self.publisher.publish(msg)
        self.get_logger().info('Publishing:')

        # Spin node
        rclpy.spin_once(self, timeout_sec=0)


class JointTrajectoryController(Node):
    # Sends command-data to JointTrajectoryController

    def __init__(self, controller_name, joint_names, type):
        self.controller_name = controller_name
        self.joint_names = joint_names
        self.type = type

        super().__init__('{}_publisher'.format(self.controller_name))
        self.set_parameters([Parameter('use_sim_time', value = 1)])

        self.publisher = self.create_publisher(JointTrajectory, 
                            '/{}/joint_trajectory'.format(self.controller_name), 10)

    def command(self, data, duration):
        # Init message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # Define Point
        point = JointTrajectoryPoint()

        for i in range(len(data)):
            data[i] = float(data[i])

        if self.type == 'position':
            point.positions = data

        point.time_from_start = Duration(seconds=duration).to_msg()

        # Define Point
        msg.points.append( point )
        self.publisher.publish(msg)
        # self.get_logger().info(f'Publishing: {data}')

        # Spin node
        rclpy.spin_once(self, timeout_sec=0)


def pickup(arm_control, gripper_control):
    # Test function for picking up cube - currently not working
    arm_control.command([0, 0, 0, 0], 0)
    arm_control.command([0, 0.31, -0.61, 0], 1)
    gripper_control.command([0,0,0,0], 1)
    time.sleep(2)
    gripper_control.command([math.pi/2+0.01, 0.01, math.pi/2+0.01, 0.01], 1)
    time.sleep(5)
    arm_control.command([0, 0, 0, 0], 3)

if __name__ == '__main__':
    # Tests
    rclpy.init()

    # gripper_control = JointEffortController( # Optionally use gripper as effort-controlled
    #     'joint_gripper_effort_controller', ['a1_joint', 'a2_joint', 
    #     'b1_joint', 'b2_joint'], 'effort')
    gripper_control = JointTrajectoryController(
        'joint_gripper_position_controller', ['a1_joint', 'a2_joint', 
        'b1_joint', 'b2_joint'], 'position')
    
    arm_control = JointTrajectoryController(
        'joint_arm_position_controller', ['base_yaw_joint', 'yaw_bicep_joint', 
        'bicep_forearm_joint', 'forearm_endeffector_joint'], 'position')
    
    # pickup(arm_control, gripper_control)

    arm_control.command([0, -0.9, -1.3, 0], 1)
    # arm_control.command([0, 0.8, -0.7, 0], 1)
    # arm_control.command([0, 0, 0, 0], 1)

    gripper_control.destroy_node()
    arm_control.destroy_node()
    rclpy.shutdown()