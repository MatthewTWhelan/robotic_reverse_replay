#!/usr/bin/python
import rospy
from geometry_msgs.msg import Pose2D
import numpy as np
from matplotlib import pyplot as plt
import os
import time

class PlotNetworkActivity:

	def __init__(self, network_weights_initial=None, hip_action_weights=None):

		rospy.init_node("network_plotting")

		# robot name
		topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

		# subscribe to Miro's body pose (x, y, theta)
		topic = topic_base_name + "/sensors/body_pose"
		print ("subscribe", topic)
		self.sub_coords = rospy.Subscriber(topic, Pose2D, self.callback_body_pose, queue_size=5, tcp_nodelay=True)

		# Setting up the network
		self.total_number_cells = 100 # square number so that the cells can be evenly divided into a square
		self.network_rates = np.zeros(100) # 10x10 array of neurons
		self.place_cell_activities = np.zeros(100)
		self.no_cells_per_m = np.sqrt(self.total_number_cells) / 2 # the arena is circular with diameter of 2m

		self.coords = np.zeros(2) # pos 0 is x coordinate and pos 1 is y coordinate
		self.all_cells_visited = np.zeros(self.total_number_cells)
		self.all_cells_visited[0:12] = 1
		self.all_cells_visited[18] = 1
		self.all_cells_visited[19] = 1
		self.all_cells_visited[29] = 1
		self.all_cells_visited[39] = 1
		self.all_cells_visited[49] = 1
		self.all_cells_visited[59] = 1
		self.all_cells_visited[69] = 1
		self.all_cells_visited[79] = 1
		self.all_cells_visited[89] = 1
		self.all_cells_visited[20] = 1
		self.all_cells_visited[30] = 1
		self.all_cells_visited[40] = 1
		self.all_cells_visited[50] = 1
		self.all_cells_visited[60] = 1
		self.all_cells_visited[70] = 1
		self.all_cells_visited[80] = 1
		self.all_cells_visited[81] = 1
		self.all_cells_visited[89:100] = 1
		self.time_start = time.time()

		if network_weights_initial is not None:
			self.network_weights = network_weights_initial
		else:
			self.network_weights = self.initialise_weights()
			np.save("data/initial_network_weights.npy", self.network_weights)
		if hip_action_weights is not None:
			self.hip_action_weights = hip_action_weights
		else:
			self.hip_action_weights = self.initialise_hip_action_weights()

		# Constants
		self.U = 0.4  # constant used in STP facilitation
		self.w_inh = 0.0005
		self.total_no_cells = 100 # square number so that the cells can be spaced evenly into a square environment

		# Network initial conditions
		self.rates_hip = np.zeros(self.total_no_cells)
		self.rates_hip_next = np.zeros(self.total_no_cells)
		self.rates_action = np.zeros(4)
		self.rates_action_next = np.zeros(4)

		self.I_E = np.zeros(self.total_no_cells)
		self.I_E_next = np.zeros(self.total_no_cells)

		self.I_inh = 0  # global inhibition is used
		self.I_inh_next = 0

		self.I_theta = 0

		self.I_place = np.zeros(self.total_no_cells)

		self.STP_D = np.ones(self.total_no_cells)
		self.STP_D_next = np.ones(self.total_no_cells)

		self.STP_F = np.ones(self.total_no_cells) * self.U
		self.STP_F_next = np.ones(self.total_no_cells) * self.U

	def callback_body_pose(self, msg):

		self.coords[0] = msg.x
		self.coords[1] = msg.y

	def initialise_weights(self):
		# weights are initially randomised, but made to obey the normalisation specification that
		# sum_{k,l} w_{i,j}^{k,l} = 0.5
		# In addition, as each cell is only connected to 8 others, it would be useless computing the
		# learning rates and activities across a 100x100 weight matrix
		# In weights[i,j], i represents the post-synapse and j the pre-synapse. I.e. for a given row of weights (say
		# weights[i]), the weights will be the incoming weights from its neighbouring pre-synapses
		weights = np.zeros((100, 8))
		for i in range(100):
			weights[i] = np.random.rand(8)
			weights[i] = weights[i] / sum(weights[i]) * 0.5
		return weights

	def initialise_hip_action_weights(self):
		# weights between hippocampal cells and the action cells are initially randomised. These weights are symmetrical
		weights = np.zeros(400)

	def update_cell_firing_rates(self, I_E_next, rho=1, epsilon=0.002):  # Equation 34
		'''
		# TEST complete
		:param I_E_next: numpy array, 100x1 vector of the updated total excitatory currents
		:param rho: float, constant used for the linear rectifier unit
		:param epsilon: float, threshold in linear rectifier unit
		:return: numpy array, 100x1 vector of the network rates
		'''

		rates_next = np.zeros(self.total_no_cells)
		for i in range(self.total_no_cells):
			if I_E_next[i] > epsilon:
				rates_next[i] = rho * (I_E_next[i] - epsilon)
			else:
				rates_next[i] = 0.0
		return rates_next

	def plot(self):
		rate = rospy.Rate(10) # will update the plot at a rate of 10Hz
		all_visited = False
		while not rospy.core.is_shutdown():
			self.place_cell_activities = self.compute_place_cell_activities(self.coords[0], self.coords[1], reward=0)
			for k in range(self.total_number_cells):
				if self.place_cell_activities[k] * 10000 > 15:
					self.all_cells_visited[k] = 1
			if sum(self.all_cells_visited) == 100 and all_visited == False:
				print("All cells visited in a time of ", time.time() - self.time_start, "s")
				all_visited = True
			# plot = np.reshape(self.place_cell_activities * 10000, (-1, 10))
			plot = np.reshape(self.all_cells_visited, (-1, 10))
			plt.clf()
			plt.imshow(plot, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
			plt.colorbar(label="Rate (Hz)")
			plt.draw()
			plt.pause(0.01)

			rate.sleep() # clever function, this estimates the amount of time taken in the script above and pauses
		# just the right amount of time so this whole loop runs at 1 Hz (set above)

	def compute_place_cell_activities(self, coord_x, coord_y, reward):
		'''

		:param coord_x: float, the x coordinate (m)
		:param coord_y: float, the y coordinate (m)
		:param reward: float, the reward value. If reward != 0, the agent should be resting and the C parameter set
		to 0.001 kHz
		:return: numpy array, vector of the networks place cell activities
		'''

		d = 0.1  # m
		no_cell_it = int(np.sqrt(self.total_number_cells))  # the number of cells along one row of the network
		C = 0.005
		cells_activity = np.zeros((10, 10))
		place = np.array((coord_x+1, coord_y+1))
		for i in range(no_cell_it):
			for j in range(no_cell_it):
				place_cell_field_location = np.array(((i / self.no_cells_per_m), (j / self.no_cells_per_m)))
				cells_activity[i][j] = C * np.exp(
					-1.0 / (2.0 * d ** 2.0) * np.dot((place - place_cell_field_location),
					                                 (place - place_cell_field_location)))
		cell_activities_array = cells_activity.flatten()

		return cell_activities_array


if __name__ == "__main__":
	main = PlotNetworkActivity()
	main.plot()
