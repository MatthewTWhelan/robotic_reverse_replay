#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose2D
import os
import numpy as np

class RewardFunction:

	def __init__(self):

		rospy.init_node('reward_node', anonymous=True)

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe to robot position
		topic = topic_base_name + "/sensors/body_pose"
		print ("subscribe", topic)
		sub_coords = rospy.Subscriber(topic, Pose2D, self.reward_value, queue_size=5, tcp_nodelay=True)
		self.coords = np.zeros(2)  # pos 0 is x coordinate and pos 1 is y coordinate

		# create reward value publisher
		topic = topic_base_name + "/reward_value"
		print ("publish", topic)
		self.pub_reward = rospy.Publisher(topic, UInt8, queue_size=0)
		self.reward = 0

		# generate a new reward location
		reward_location = np.random.randint(2, 7, (2))  # reward location is located at one of the place cell
		# locations, but only in the inner 8 rows and columns
		self.reward_location_xy = reward_location / 5.0 - 1
		self.reward_location_xy = np.array(([0.8, 0.]))  # for testing
		self.reward = 0
		print("Reward located at ", self.reward_location_xy)

	def reward_value(self, msg):
		# TODO set the reward value here once Miro has reached the reward

		robot_position = msg
		robot_position_x = robot_position.x
		robot_position_y = robot_position.y
		if abs(robot_position.x - self.reward_location_xy[0]) < 0.1 and abs(robot_position.y - \
		                                                                  self.reward_location_xy[1]) < 0.1:
			#print("Found the reward!!!")
			#print(self.coords, self.reward_location_xy)
			self.reward = 2
		else:
			self.reward = 0

	def main(self):

		rate = rospy.Rate(2)  # 2hz
		while not rospy.is_shutdown():
			self.pub_reward.publish(self.reward)
			rate.sleep()


if __name__ == '__main__':
	run = RewardFunction()
	run.main()