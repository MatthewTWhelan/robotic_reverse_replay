#!/usr/bin/python
import rospy
from std_msgs.msg import UInt8, UInt16, UInt32, Float32MultiArray, UInt16MultiArray, UInt32MultiArray
from geometry_msgs.msg import Pose2D, TwistStamped
from sensor_msgs.msg import JointState
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import miro2 as miro

class MiroController:

	def __init__(self):

		rospy.init_node("miro_controller")

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe to miro's sensors package
		topic = topic_base_name + "/sensors/package"
		print ("subscribe", topic)
		self.sub_package = rospy.Subscriber(topic, miro.msg.sensors_package, self.callback_sensors, queue_size=5,
		                                    tcp_nodelay=True)
		self.sonar_val = 1

		# subscribe to Miro's body pose (x, y, theta)
		topic = topic_base_name + "/sensors/body_pose"
		print ("subscribe", topic)
		self.sub_coords = rospy.Subscriber(topic, Pose2D, self.callback_body_pose, queue_size=5, tcp_nodelay=True)
		self.coords = np.zeros(2)  # pos 0 is x coordinate and pos 1 is y coordinate

		# subscribe to the reward value
		topic = topic_base_name + "/reward_value"
		print ("subscribe", topic)
		self.sub_reward = rospy.Subscriber(topic, UInt8, self.update_reward, queue_size=5, tcp_nodelay=True)
		self.reward_val = 0

		# publish to wheels
		topic = topic_base_name + "/control/cmd_vel"
		print ("publish", topic)
		self.pub_wheels = rospy.Publisher(topic, TwistStamped, queue_size=0)
		self.msg_wheels = TwistStamped()

		# publish to kinematic joints
		topic = topic_base_name + "/control/kinematic_joints"
		print ("publish", topic)
		self.pub_kin = rospy.Publisher(topic, JointState, queue_size=0)
		self.msg_kin = JointState()
		self.msg_kin.position = [0.0, np.radians(30.0), 0.0, 0.0]

		# publish to cosmetic joints
		topic = topic_base_name + "/control/cosmetic_joints"
		self.pub_cos = rospy.Publisher(topic, Float32MultiArray, queue_size=0)
		self.msg_cos = Float32MultiArray()
		self.msg_cos.data = [0.5, 0.5, 0.0, 0.0, 0.5, 0.5] # all between 0 and 1 [tail droop, tail wag, left eyelid,
		# right eyelid, left ear, right ear]

	def callback_sensors(self, msg):

		self.sonar_val = msg.sonar.range

	def callback_body_pose(self, msg):

		self.coords[0] = msg.x
		self.coords[1] = msg.y

	def update_reward(self, msg):

		self.reward_val = msg.data

	def controller(self):
		t_init = time.time()
		t_last_4s = 0 # used so that the random walk controller only updates every 4 seconds
		vel = 0  # max = 0.4 m/s, min = -0.4 m/s
		alpha = 0  # pos = anticlockwise; max = +- 5 rad/s
		while not rospy.core.is_shutdown():
			t = time.time() - t_init
			#print(int(t))
			#
			# print(self.sonar_val)
			if self.sonar_val < 0.05:
				self.avoid_wall()
				vel, alpha = self.random_walk()
				self.msg_wheels.twist.linear.x = vel
				self.msg_wheels.twist.angular.z = alpha
			if self.reward_val != 0:
				print("Found a reward!")
				print(self.reward_val)
				self.victory_dance()
				t_init = time.time()
				continue
			if int(t) % 4 == 0 and t_last_4s != int(t):
				t_last_4s = int(t)
				if t > 60:
					print("Disappointed Miro")
					self.head_shake_in_disappointment() # reward not found after a long search. Miro gets a little
				# disappointed...
					self.msg_kin.position[1] = np.radians(30) # reset lift
					self.msg_kin.position[2] = np.radians(0) # reset yaw
					self.msg_cos.data[2] = 0 # reset eyelids
					self.msg_cos.data[3] = 0
					self.pub_kin.publish(self.msg_kin)
					self.pub_cos.publish(self.msg_cos)
					t_init = time.time() # start again poor Miro

				vel, alpha = self.random_walk()
				self.msg_wheels.twist.linear.x = vel
				self.msg_wheels.twist.angular.z = alpha

			self.pub_wheels.publish(self.msg_wheels)

	def random_walk(self):
		p_straight = 0.5
		p_anticlock = 0.25
		mag_turn = np.random.rand()
		p = np.random.rand()
		if p < p_straight:
			vel = 0.2
			alpha = 0
		elif p < p_straight + p_anticlock:
			vel = 0.2
			alpha = 1 * mag_turn
		else:
			vel = 0.2
			alpha = -1 * mag_turn

		return vel, alpha

	def avoid_wall(self):
		t_init = time.time()
		self.msg_wheels.twist.linear.x = -0.1
		self.msg_wheels.twist.angular.z = 0
		while time.time() - t_init < 1:
			self.pub_wheels.publish(self.msg_wheels)
		p_anti_clock = 0.8
		self.msg_wheels.twist.linear.x = 0
		if np.random.rand() < p_anti_clock:
			self.msg_wheels.twist.angular.z = 1.5
		else:
			self.msg_wheels.twist.angular.z = -1.5
		while time.time() - t_init < 2:
			self.pub_wheels.publish(self.msg_wheels)

	def victory_dance(self):
		t_init = time.time()
		t = 0
		while t < 10:
			self.msg_wheels.twist.linear.x = 0
			self.msg_wheels.twist.angular.z = 0
			self.pub_wheels.publish(self.msg_wheels)

	def head_shake_in_disappointment(self):
		t_init = time.time()
		self.msg_wheels.twist.linear.x = 0
		self.msg_wheels.twist.angular.z = 0
		t = 0
		while t < 6:
			t = time.time() - t_init
			if t < 2:
				self.msg_kin.position[1] = np.radians(t*15 + 34) # lift, between 8 degs (all way up) and 60 deg (all way
		# down)
				self.msg_cos.data[2] = t - 0.2
				self.msg_cos.data[3] = t - 0.2
			else:
				self.msg_kin.position[2] = np.radians(30.0 * np.sin((t - 1) / (8/3) * 2 * np.pi)) # yaw, between -50 and
			# 50 degs
			self.msg_kin.position[3] = np.radians(0.0) #  # pitch, between -22 degs (up) and 8 degs (down)
			self.pub_kin.publish(self.msg_kin)
			self.pub_cos.publish(self.msg_cos)


if __name__ == "__main__":
	main = MiroController()
	main.controller()