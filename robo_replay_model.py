#!/usr/bin/python
'''
This is the full robot replay script. It subscribes (via ROS) to the robot's coordinates, produces the rate
activities according to the model, and replays once a reward has been reached, which is found by subscribing to
the reward topic. The reward is set by running the generate_reward.py script.
'''

import rospy
from std_msgs.msg import UInt8
from geometry_msgs.msg import Pose2D
import numpy as np
import os
import signal
import sys
# import time


class RoboReplay():
	def __init__(self):
		# system handling
		signal.signal(signal.SIGINT, self.signal_handler)

		# ROS stuff
		rospy.init_node("Robo_Replay")

		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe to the reward value
		topic = topic_base_name + "/reward_value"
		print("subscribed to", topic)
		self.sub_reward = rospy.Subscriber(topic, UInt8, self.update_reward, queue_size=5, tcp_nodelay=True)
		self.reward_val = 0

		# subscribe to Miro's body pose (x, y, theta)
		topic = topic_base_name + "/sensors/body_pose"
		print("subscribed to", topic)
		self.sub_coords = rospy.Subscriber(topic, Pose2D, self.callback_body_pose, queue_size=5, tcp_nodelay=True)
		self.coords = np.array((-0.7, 0.0))  # pos 0 is x coordinate and pos 1 is y coordinate

		# model parameters and variable initial conditions
		self.network_size = 100 # A square number
		self.a = 1
		self.epsilon = 2  # Hz min threshold for rates
		self.theta = 40  # Hz max threshold for rates
		self.delta_t = 0.01  # s simulation time steps
		self.w_inh = 0.1

		# set variables initial conditions
		self.rates = np.zeros(self.network_size)
		self.currents = np.zeros(self.network_size)
		self.intrinsic_e = np.ones(self.network_size) * 0.1
		self.network_weights = self.initialise_weights()
		self.stp_d = np.ones(self.network_size)
		self.stp_f = np.ones(self.network_size) * 0.6
		self.I_place = np.zeros(self.network_size)
		self.I_inh = 0
		self.replay = False

		# lists for storing network values during trials. Saves to the data folder once script is exited via ctrl-c
		# being pressed.
		self.time_series = []
		self.rates_series = []
		self.intrinsic_e_series = []

	def signal_handler(self, sig, frame):
		# If saving the entire trajectory for later analysis, uncomment this section and also lines 347-349
		# print('\nSaving trial data')
		# np.save('data/time_series.npy', self.time_series)
		# np.save('data/rates_series.npy', self.rates_series)
		# np.save('data/intrinsic_e_series.npy', self.intrinsic_e_series)

		# clean up the temporary data files
		os.remove('data/intrinsic_e.npy')
		os.remove('data/rates_data.npy')
		os.remove('data/place_data.npy')

		sys.exit(0)

	def update_reward(self, msg):
		self.reward_val = msg.data

	def callback_body_pose(self, msg):
		self.coords[0] = msg.x
		self.coords[1] = msg.y

	def initialise_weights(self):
		# weights are initially all symmetrical, but made to obey the normalisation specification that
		# sum_{k,l} w_{i,j}^{k,l} = 10
		# In addition, as each cell is only connected to 8 others, it would be useless computing the
		# learning rates and activities across a 100x100 weight matrix
		# In weights[i,j], i represents the post-synapse and j the pre-synapse. I.e. for a given row of weights (say
		# weights[i]), the 8 weights in that row will be the incoming weights from its neighbouring pre-synapses in
		# locations [W NW N NE E SE S SW]
		weights = np.zeros((self.network_size, 8))
		for i in range(100):
			for j in range(8):
				if self.is_computable(i, j):
					weights[i, j] = 1.0
			weights[i] = weights[i] / sum(weights[i]) * 8

		return weights

	def is_computable(self, i, j):
		'''

		:param i: integer, indicates which is the selected neuron
		:param j: integer, indicates which neighbour neuron i is receiving something from
		:return: bool
		'''

		# This is a confusing function, so I must describe it in detail. Because of the 2D arrangement of the
		# network, the neurons on the edges will not be connected to any neurons to its side (north most neurons have
		# no connections to its north, etc.). As a result, when performing the computations, such as computing the
		# incoming rates to a neuron from its neighbour neurons, it is worth first determining whether this
		# computation is valid (i.e. there is indeed a neighbour neuron in the specified position). This function
		# therefore determines whether a computation, whether it be incoming rates or updating weights, is valid or
		# not. For simplicity, this is computed for a 50x50 2D neural network.
		# It is important to note here that the order of connections is as follows: [W NW N NE E SE S SW]. So from j
		# = 0 to j = 7, the increment in the index represents a clockwise rotation starting at the point W.

		no_cells_per_row = int(np.sqrt(self.network_size))

		if i % no_cells_per_row == 0 and (j == 0 or j == 1 or j == 7):  # no W connections
			return False
		elif i in range(no_cells_per_row) and (j == 1 or j == 2 or j == 3):  # no N connections
			return False
		elif (i + 1) % no_cells_per_row == 0 and (j == 3 or j == 4 or j == 5):  # no E connections
			return False
		elif i in range(self.network_size - no_cells_per_row, self.network_size) and (j == 5 or j == 6 or j == 7):  # no S
			# connections
			return False
		else:  # it's a valid computation
			return True

	def neighbour_index(self, i, j):
		'''

		:param i: integer, indicates which is the selected neuron
		:param j: integer, indicates which neighbour neuron i is receiving something from
		:return: bool
		'''

		# Due to the 2D structure of the network, it is important to find which index from the vector of neurons
		# should be used as the neighbour neuron. For instance, the 2D network is concatenated by each row. So the
		# first 50 neurons of the vector of neurons represents the first row of the 2D network. The next 50 represent
		# the second row, and so on. Hence, the connection that neuron i receives from its north will be located at
		# i-50. For simplicity, this is computed for a 50x50 2D neural network.
		# It is important to note here that the order of connections is as follows: [W NW N NE E SE S SW]. So from j
		# = 0 to j = 7, the increment in the index represents a clockwise rotation starting at the point W.

		no_cells_per_row = int(np.sqrt(self.network_size))

		if j == 0:  # W connection
			return i - 1
		elif j == 1:  # NW connection
			return i - (no_cells_per_row + 1)
		elif j == 2:  # N connection
			return i - no_cells_per_row
		elif j == 3:  # NE connection
			return i - (no_cells_per_row - 1)
		elif j == 4:  # E connection
			return i + 1
		elif j == 5:  # SE connection
			return i + (no_cells_per_row + 1)
		elif j == 6:  # S connection
			return i + no_cells_per_row
		elif j == 7:  # SW connection
			return i + (no_cells_per_row - 1)
		else:
			return IndexError

	def compute_rates(self, currents):
		rates_update = np.zeros(self.network_size)
		for i in range(self.network_size):
			if currents[i] < self.epsilon:
				rates_update[i] = 0
			else:
				rates_update[i] = min(self.a * (currents[i] - self.epsilon), 100) # upper bound of 200 Hz
				# rates_update[i] = self.a * (currents[i] - self.epsilon)  # no upper bound

		return rates_update

	def update_currents(self, currents, delta_t, intrinsic_e, weights, rates, stp_d, stp_f, I_inh, I_place, replay=False):
		tau_I = 0.05  # s
		currents_update = np.zeros(self.network_size)
		g = 0
		for i in range(self.network_size):
			sum_w_r_df = 0
			if replay:  # g is only nonzero during replays, so if there is no replay it's pointless summing neighbour rates
				for j in range(8):
					if self.is_computable(i, j):
						neighbour = self.neighbour_index(i, j)
						sum_w_r_df += weights[i, j] * rates[neighbour] * stp_d[neighbour] * stp_f[neighbour]
				g = intrinsic_e[i]
			currents_update[i] = currents[i] + (-currents[i] + g * sum_w_r_df - I_inh + I_place[i]) * delta_t  / tau_I

		return currents_update

	def update_intrinsic_e(self, intrinsic_e, delta_t, rates):
		tau_e = 10  # s
		sigma_ss = 0.1
		sigma_max = 4
		r_sigma = 10
		beta = 1
		intrinsic_e_update = np.zeros(self.network_size)
		for i in range(self.network_size):
			sigmoid = (sigma_max - 1) / (1 + np.exp(-beta * (rates[i] - r_sigma)))
			intrinsic_e_update[i] = ((sigma_ss - intrinsic_e[i]) / tau_e + sigmoid) * delta_t + intrinsic_e[i]
			if intrinsic_e_update[i] > sigma_max:
				intrinsic_e_update[i] = sigma_max
		return intrinsic_e_update

	def update_I_inh(self, I_inh, delta_t, w_inh, rates):
		tau_inh = 0.05  # s
		sum_rates = 0
		for i in range(self.network_size):
			sum_rates += rates[i]

		return (-I_inh / tau_inh + w_inh * sum_rates) * delta_t + I_inh

	def update_STP(self, STP_D, STP_F, delta_t, rates):
		'''
		:param STP_F: numpy array, 100x1 current STP_F vector
		:param STP_D: numpy array, 100x1 current STP_D vector
		:param delta_t: float, time step (s)
		:param rates: numpy array, 100x1 of network rates
		:return: two numpy arrays, 100x1 updated vectors of the STP variables
		'''

		tau_f = 1  # s
		tau_d = 1.5  # s
		U = 0.6
		STP_F_next = np.zeros(self.network_size)
		STP_D_next = np.zeros(self.network_size)
		for f in range(self.network_size):
			STP_D_next[f] = ((1.0 - STP_D[f]) / tau_d - rates[f] * STP_D[f] * STP_F[f]) * delta_t + STP_D[f]
			STP_F_next[f] = ((U - STP_F[f]) / tau_f + U * (1 - STP_F[f]) * rates[f]) * delta_t + STP_F[f]
		if max(STP_D_next) > 1 or min(STP_D_next) < 0 or max(STP_F_next) > 1 or min(STP_F_next) < 0:
			print('STP value ouside of bounds')
			print(max(STP_D_next), min(STP_D_next), max(STP_F_next), min(STP_F_next))
		# print(STP_F_next*STP_D_next)
		return STP_D_next, STP_F_next

	def compute_place_cell_activities(self, coord_x, coord_y, reward, movement=False):
		'''

		:param coord_x: float, the x coordinate (m)
		:param coord_y: float, the y coordinate (m)
		:param reward: float, the reward value. If reward != 0, the agent should be resting and the C parameter set
		to 1 Hz
		:param movement: bool, indicates whether the robot moved in the current time step or not
		:return: numpy array, vector of the networks place cell activities
		'''

		d = 0.1  # m
		no_cells_per_m = np.sqrt(self.network_size) / 2
		no_cell_it = int(np.sqrt(self.network_size))  # the number of cells along one row of the network
		if movement or reward != 0:
			C = 50  # Hz
		else:
			C = 0  # Hz
		cells_activity = np.zeros((no_cell_it, no_cell_it))
		place = np.array((coord_x + 1, coord_y + 1))
		for i in range(no_cell_it):
			for j in range(no_cell_it):
				place_cell_field_location = np.array(((i / no_cells_per_m), (j / no_cells_per_m))) + 0.1
				cells_activity[j][i] = C * np.exp(
					-1.0 / (2.0 * d ** 2.0) * np.dot((place - place_cell_field_location),
					                                 (place - place_cell_field_location)))
		cell_activities_array = cells_activity.flatten()
		# print(cell_activities_array)
		return cell_activities_array

	def main(self):
		t = 0 # s
		t_replay = 0 # s
		coords_prev = self.coords.copy()

		# self.intrinsic_e = np.ones(self.network_size) # used to test the network with no intrinsic plasticity
		while not rospy.core.is_shutdown():
			rate = rospy.Rate(int(1 / self.delta_t))
			t += self.delta_t
			coords = self.coords.copy()
			movement = True

			# Can use this to ensure that there is no place cell activity if MiRo is still. Otherwise movement is set
			# always to be true
			# movement_x = coords[0] - coords_prev[0]
			# movement_y = coords[1] - coords_prev[1]
			# if movement_x > 0.0 or movement_y > 0.0: # at least a movement velocity of 0.002 / delta_t is required
			# 	movement = True
			# else:
			# 	movement = False

			# Set current variable values to the previous ones
			coords_prev = coords.copy()
			rates_prev = self.rates.copy()
			currents_prev = self.currents.copy()
			intrinsic_e_prev = self.intrinsic_e.copy()
			stp_d_prev = self.stp_d.copy()
			stp_f_prev = self.stp_f.copy()
			I_place_prev = self.I_place.copy()
			I_inh_prev = self.I_inh
			network_weights_prev = self.network_weights.copy()

			if self.reward_val == 0:
				self.replay = False
				t_replay = 0
				# Run standard activity during exploration
				# Update the variables
				self.currents = self.update_currents(currents_prev, self.delta_t, intrinsic_e_prev,
				                                     network_weights_prev, rates_prev, stp_d_prev, stp_f_prev,
				                                     I_inh_prev, I_place_prev)
				self.rates = self.compute_rates(self.currents)
				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, self.delta_t, rates_prev)
				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, stp_f_prev, self.delta_t, rates_prev)
				self.I_place = self.compute_place_cell_activities(coords_prev[0], coords_prev[1], 0, movement)
				self.I_inh = self.update_I_inh(I_inh_prev, self.delta_t, self.w_inh, rates_prev)

			else:
				# Run a reverse replay
				if not self.replay:
					print('Running reverse replay event')
				self.replay = True
				t_replay += self.delta_t

				if (1 < t_replay < 1.1) or (3 < t_replay < 3.1) or (5 < t_replay < 5.1) or (7 < t_replay < 7.1):
					# if (replay_step < 100):
					I_place = self.I_place
				else:
					I_place = np.zeros(self.network_size)
				# set variables at the next time step to the ones now
				self.currents = self.update_currents(currents_prev, self.delta_t, intrinsic_e_prev,
				                                     network_weights_prev, rates_prev, stp_d_prev, stp_f_prev,
				                                     I_inh_prev, I_place, replay=self.replay)
				self.rates = self.compute_rates(self.currents)
				self.intrinsic_e = self.update_intrinsic_e(intrinsic_e_prev, self.delta_t, rates_prev)
				self.stp_d, self.stp_f = self.update_STP(stp_d_prev, stp_f_prev, self.delta_t, rates_prev)
				self.I_inh = self.update_I_inh(I_inh_prev, self.delta_t, self.w_inh, rates_prev)

			# Saves the data at each time step, useful for live plotting
			np.save('data/intrinsic_e.npy', self.intrinsic_e)
			np.save('data/rates_data.npy', self.rates)
			np.save('data/place_data.npy', self.I_place)

			# Series data can be used to save the whole trajectory for later analysis
			# self.time_series.append(t)
			# self.rates_series.append(self.rates)
			# self.intrinsic_e_series.append(self.intrinsic_e)

			rate.sleep()


if __name__ == '__main__':
	robo_replay = RoboReplay()
	robo_replay.main()