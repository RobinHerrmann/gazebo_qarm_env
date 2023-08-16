'''
The QArmEnv builds on top of gym.env to control the QArm-Simulation
with a RL-Agent.

Author:  Robin Herrmann
Created: 2023.07.12
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.duration import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import JointTrajectoryControllerState

from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import LinkStates, ModelStates, ContactsState

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from ament_index_python.packages import get_package_share_directory

import numpy as np

import math
import time
import os
import xacro

from GazeboConnector import GazeboConnector
from JointControllerInterface import JointTrajectoryController
import observation_utils as obs_ut


class QArmEnv(gym.Env):

    def __init__(self):
        TARGET_NAME = 'target'

        JOINT_PUBLISHER = '/joint_trajectory_controller/joint_trajectory'
        JOINT_SUBSCRIBER = '/joint_states'
        MODEL_SUBSCRIBER = '/gazebo/model_states'
        LINK_SUBSCRIBER = '/gazebo/link_states'
        RGB_SUBSCRIBER = '/depth_camera/image_raw'
        DEPTH_SUBSCRIBER = '/depth_camera/depth/image_raw'
        TARGET_COLLISION_SUBSCRIBER = '/'+TARGET_NAME+'/cube_collision'
        QARM_COLLISION_SUBSCRIBER = '/qarm_collision'

        # Time variables
        self.EPISODE_TIMEOUT = 5 # seconds
        self.STEPWAITTIME = 100 # millisase_link

        # List of models collisions are disallowed with
        self.collision_disallowed_models = ['ground_plane::link::collision', '3DPrinterBed::link_0::collision', 
                                            'qarm_v1::bicep_link::bicep_link_collision', 'qarm_v1::yaw_link::yaw_link_collision',
                                            'qarm_v1::base_link::base_link_collision']

        # Defining gym.Env action_space and observation_space
        self.action_space = spaces.Box(
            low = -0.1,
            high = 0.1, 
            shape=(5,), 
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low = np.array([-1, -1, 0, -1, -1, 0, -1, -1, -1, -math.pi, -math.pi]),
            high = np.array([1 , 1, 1, 1, 1, 1, 1, 1, 1, math.pi, math.pi]),
            dtype=np.float32,
        )
            # OPTIONAL observation space as dictionary
        # self.observation_space = spaces.Dict({
        #     "TCP": spaces.Box(
        #         low = np.array([-1, -1, 0]),
        #         high = np.array([1 , 1, 1]),
        #         dtype=np.float64,
        #     ),
        #     "TARGET": spaces.Box(
        #         low = np.array([-1, -1, 0]),
        #         high = np.array([1 , 1, 1]),
        #         dtype=np.float64,
        #     ),
        #     "RELATIVE": spaces.Box(
        #         low = np.array([-1, -1, -1]),
        #         high = np.array([1 , 1, 1]),
        #         dtype=np.float64,
        #     ),
        #     "GRIPPER": spaces.Box(
        #         low = np.array([-math.pi, -math.pi]),
        #         high = np.array([math.pi, math.pi]),
        #         dtype=np.float64,
        #     ),
        #     # "CAMERA": POSSIBLE EXPANSION
        #     # "COLLISION": POSSIBLE EXPANSION
        # })

        # Initialize ROS node
        rclpy.init()
        self.node = rclpy.create_node(self.__class__.__name__)
        self.node.set_parameters([Parameter('use_sim_time', value = True)])

        # Initialize Gazebo Connector
        self.gzcon = GazeboConnector(self.node)

        # Create subscriber and publishers
        self._pub = self.node.create_publisher(
            JointTrajectory, JOINT_PUBLISHER, qos_profile=qos_profile_sensor_data)
        self._sub_joint = self.node.create_subscription(
            JointState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_model = self.node.create_subscription(
            ModelStates, MODEL_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_link = self.node.create_subscription(
            LinkStates, LINK_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)

        self._sub_cam_rgb = self.node.create_subscription(
            Image, RGB_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_cam_depth = self.node.create_subscription(
            Image, DEPTH_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)

        self._sub_target_collision = self.node.create_subscription(
            ContactsState, TARGET_COLLISION_SUBSCRIBER, self.target_collision_observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_qarm_collision = self.node.create_subscription(
            ContactsState, QARM_COLLISION_SUBSCRIBER, self.qarm_collision_observation_callback, qos_profile=qos_profile_sensor_data)

        self._sub_count = 7
        self._sub_count_additional = 7-1 # count of collision publishers

        # Initialize variables
        self._msg_observation = "None"
        self._msg_joint_state= "None"
        self._msg_model_state = "None"
        self._msg_link_state = "None"

        self._msg_camera_rgb = "None"
        self._msg_camera_depth = "None"

        self._msg_target_collision = ContactsState()
        self._msg_qarm_collision = ContactsState()

        self.last_joints_complete = {"name": [], "position": [0,0,0,0,0,0,0,0]}

        self.last_reset_clock = 0

        # Spawn Target
        xacro_file = os.path.join(get_package_share_directory(
            'qarm_v1'), 'urdf', 'cube.urdf.xacro')    
        xml_cube = xacro.process_file(xacro_file).toxml()#.replace('"', '\\"')
        self.gzcon.spawn_entity(TARGET_NAME, xml_cube, [0.6, 0.0, 0.11])

        # Initial observation
        self.gzcon.unpause()
        time.sleep(5) # important to wait for ros2_control init
        init_observation = self.take_observation()
        self.gzcon.pause()

        self._last_distance = np.linalg.norm(init_observation["RELATIVE"][0:3])

        # Initilize controller connection
        self.arm_control = JointTrajectoryController(
            'joint_arm_position_controller', ['base_yaw_joint', 'yaw_bicep_joint', 
            'bicep_forearm_joint', 'forearm_endeffector_joint'], 'position')

        self.gripper_control = JointTrajectoryController(
            'joint_gripper_position_controller', ['a1_joint', 'a2_joint', 
            'b1_joint', 'b2_joint'], 'position')
    
        # Initial arm position
        self.arm_control.command([0, 0, 0, 0], 0)
        self.gripper_control.command([0, 0, 0, 0], 0)

    def observation_callback(self, message):
        if type(message) == ModelStates:
            self._msg_model_state = message
        elif type(message) == LinkStates:
            self._msg_link_state = message
        elif type(message) == JointState:
            self._msg_joint_state= message
        elif type(message) == Image:
            if message.encoding == "rgb8":
                self._msg_camera_rgb = message
            else:
                self._msg_camera_depth = message
        else:
            self._msg_observation = message
            # self.node.get_logger().error("Received unexpected message type in observation_callback()")
    
    def target_collision_observation_callback(self, message):
        if type(message) == ContactsState:
            self._msg_target_collision = message
        else:
            return
            # self.node.get_logger().error("Received wrong message type in target_collision_observation_callback()")

    def qarm_collision_observation_callback(self, message):
        if type(message) == ContactsState:
            self._msg_qarm_collision = message
        else:
            return
            # self.node.get_logger().error("Received wrong message type in qarm_collision_observation_callback()")


    def take_observation(self):
        # Spin node as often as there are messages to read
        for i in range(self._sub_count):
            rclpy.spin_once(self.node)

        # Take snapshot of observation
        obs_msg_joints = self._msg_joint_state
        obs_msg_models = self._msg_model_state
        obs_msg_links = self._msg_link_state
            # optional: additionally process camera data from self._msg_camera_depth here

        # Process messages into observation np.arrays
        obs_gripper, self.last_joints_complete = obs_ut.process_obs_msg_joints(obs_msg_joints)
        obs_target = obs_ut.process_obs_msg_models(obs_msg_models)
        obs_tcp = obs_ut.process_obs_msg_links(obs_msg_links)
        obs_relative = obs_ut.process_obs_relative(obs_tcp, obs_target)

        # Dict observation
        observation = {
            "TCP": obs_tcp,
            "TARGET": obs_target,
            "RELATIVE": obs_relative,
            "GRIPPER": obs_gripper,
        }

        # self.node.get_logger().info("QArmEnv.step(); Command " + str(observation))

        return observation


    def step(self, action):
        self.ros_clock = self.node.get_clock().now().nanoseconds
        # self.node.get_logger().info("QArmEnv.step(); ros_clock = " + str(self.ros_clock))

        # Take action
        arm_command = self.last_joints_complete['position'][0:4] + action[0:4]
        gripper_command = self.last_joints_complete['position'][4:8] + np.array([action[4], action[4], action[4], action[4]])

        self.arm_control.command(arm_command.tolist(), self.STEPWAITTIME/1000)
        self.gripper_control.command(gripper_command.tolist(), self.STEPWAITTIME/1000)

        # self.node.get_logger().info("QArmEnv.step(); Action taken " + str(arm_command) + " " + str(gripper_command))

        # Wait for action to be completed
        truncated = False
        self._clock_last_step = self.node.get_clock().now().nanoseconds
        # self.node.get_logger().info("QArmEnv.step(); ros_clock = " + str(self._clock_last_step))
        while self.node.get_clock().now().nanoseconds - self._clock_last_step <= self.STEPWAITTIME*1000000:
            # observation = self.take_observation()
            for i in range(self._sub_count):
                rclpy.spin_once(self.node)
            if len(self._msg_qarm_collision.states) > 0:
                # Already check for Collisions
                for i, s in enumerate(self._msg_qarm_collision.states):
                    for dm in self.collision_disallowed_models:
                        if dm in [s.collision1_name, s.collision2_name]:
                            truncated = True
                            # self.node.get_logger().info("Truncated Condition 2: Collision" + str((s.collision1_name, s.collision2_name)))
                            break
                    if truncated == True:
                        break
            if truncated == True:
                break
            continue

        # Read Observation
        observation = self.take_observation()

        # Check Truncation
        truncated = False
        truncated_time = False
            # Condition 1: Target out of reach (750mm reach)
        if np.linalg.norm(observation["TARGET"][0:3]) >= 0.70:
            # self.node.get_logger().info("Truncated Condition 1: Reach")
            truncated = True

            # Condition 2: Robot collision
        if len(self._msg_qarm_collision.states) > 0:
            for i, s in enumerate(self._msg_qarm_collision.states):
                for dm in self.collision_disallowed_models:
                    if dm in [s.collision1_name, s.collision2_name]:
                        truncated = True
                        # self.node.get_logger().info("Truncated Condition 2: Collision" + str((s.collision1_name, s.collision2_name)))
                        break
                if truncated == True:
                    break
            # Condition 4: Timeout
        if self.ros_clock - self.last_reset_clock > self.EPISODE_TIMEOUT * 1000000000:
            truncated = True
            truncated_time = True
            # self.node.get_logger().info("Truncated Condition 3: Time")

            # Condition n: ...
        # optional

        # Check Termination
        terminated = False
            # Condition 1: Close enough to target
        # self.node.get_logger().info(str(np.linalg.norm(observation["RELATIVE"][0:3])))
        if np.linalg.norm(observation["RELATIVE"][0:3]) <= 0.16:
            self.node.get_logger().info("Terminated Condition 1: Close")
            terminated = True
        #     # Condition 2: Finger touches target
        # if len(self._msg_qarm_collision.states) > 0:
        #     for i, s in enumerate(self._msg_qarm_collision.states):
        #         [s.collision1_name, s.collision2_name]
            # Condition n: ...

        # Calculate reward
        distance = np.linalg.norm(observation["RELATIVE"][0:3])
        reward_distance = 100*(self._last_distance - distance)
        self._last_distance = distance

        reward_time = (self.last_reset_clock - self.ros_clock)/1000000000

        reward = reward_distance + reward_time

        if terminated:
            reward += 100
        if truncated and not truncated_time:
            reward -= 50

        # self.node.get_logger().info("QArmEnv.step(); Reward: \n" + str(round(distance, 6)) + "\n" + str(round(self._last_distance, 6)) + " \n" + str(reward_distance) + " " + str(reward_time) + " \n" + str(reward))

        # Info
        info = {}

        # Transform observation for compatibility (dict to array)
        observation = np.concatenate((observation["TCP"], observation["TARGET"], observation["RELATIVE"], observation["GRIPPER"]))

        return (observation, reward, terminated, truncated, info)


    def reset(self, seed = None, options = {}):
        self.gzcon.unpause()

        self.arm_control.command([0, 0, 0, 0], 0)
        self.gripper_control.command([0, 0, 0, 0], 0)
        self.gzcon.reset_world()
        time.sleep(0.1)

        state = self.take_observation()
        self.last_reset_clock = self.node.get_clock().now().nanoseconds

        self._last_distance = np.linalg.norm(state["RELATIVE"][0:3])

        info = {}

        # Transform observation for compatibility (dict to array)
        state = np.concatenate((state["TCP"], state["TARGET"], state["RELATIVE"], state["GRIPPER"]))

        return (state, info)