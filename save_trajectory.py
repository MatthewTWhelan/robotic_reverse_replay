#!/usr/bin/python
'''
A script for storing the previous 10 seconds of Miro's trajectory
'''

import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose2D
import numpy as np
import os
import time


def update_reward(msg):
	global reward_val
	reward_val = msg.data


def callback_body_pose(msg):
	global coords
	coords[0] = msg.x
	coords[1] = msg.y


rospy.init_node("save_trajectory")

# robot name
topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

# subscribe to the reward value
topic = topic_base_name + "/reward_value"
print ("subscribe", topic)
sub_reward = rospy.Subscriber(topic, UInt8, update_reward, queue_size=5, tcp_nodelay=True)
reward_val = 0

# subscribe to Miro's body pose (x, y, theta)
topic = topic_base_name + "/sensors/body_pose"
print ("subscribe", topic)
sub_coords = rospy.Subscriber(topic, Pose2D, callback_body_pose, queue_size=5, tcp_nodelay=True)
coords = np.zeros(2)  # pos 0 is x coordinate and pos 1 is y coordinate

coords_timeline = np.zeros((200, 2)) # at a rate of 10 Hz, we will have 10 data points per second, making 200 over
# 20 seconds

while not rospy.core.is_shutdown():
	rate = rospy.Rate(10) # will run at a rate of 10Hz
	coords_timeline = np.roll(coords_timeline, -1, 0)
	coords_timeline[-1] = coords
	rate.sleep()
	if reward_val != 0:
		np.save("data/trajectory_data_straight_line.npy", coords_timeline)
		print("reward found... saving trajectory data and quitting")
		break

