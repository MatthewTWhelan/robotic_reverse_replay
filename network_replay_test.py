#!/usr/bin/python3
'''
This is a test script for my initial replay network. All it requires as input is some trajectory data.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

def initialise_weights():
	# weights are initially randomised, but made to obey the normalisation specification that
	# sum_{k,l} w_{i,j}^{k,l} = 1
	# In addition, as each cell is only connected to 8 others, it would be useless computing the
	# learning rates and activities across a 100x100 weight matrix
	# In weights[i,j], i represents the post-synapse and j the pre-synapse. I.e. for a given row of weights (say
	# weights[i]), the weights will be the incoming weights from its neighbouring pre-synapses in locations [W NW N
	# NE E SE S SW]
	weights = np.zeros((network_size, 8))
	for i in range(100):
		for j in range(8):
			if is_computable(i, j):
				weights[i,j] = np.random.rand()
		weights[i] = weights[i] / sum(weights[i])
	return weights


def is_computable(i, j):
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

	no_cells_per_row = int(np.sqrt(network_size))

	if i % no_cells_per_row == 0 and (j == 0 or j == 1 or j == 7):  # no W connections
		return False
	elif i in range(no_cells_per_row) and (j == 1 or j == 2 or j == 3):  # no N connections
		return False
	elif (i + 1) % no_cells_per_row == 0 and (j == 3 or j == 4 or j == 5):  # no E connections
		return False
	elif i in range(network_size - no_cells_per_row, network_size) and (j == 5 or j == 6 or j == 7):  # no S
		# connections
		return False
	else:  # it's a valid computation
		return True


def neighbour_index(i, j):
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

	no_cells_per_row = int(np.sqrt(network_size))

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


def compute_rates(currents):
	rates_update = np.zeros(network_size)
	for i in range(network_size):
		if currents[i] < epsilon:
			rates_update[i] = 0
		else:
			rates_update[i] = min(rho * (currents[i] - epsilon), rho * theta)

	return rates_update


def update_currents(currents, delta_t, intrinsic_e, weights, rates, I_inh, I_theta, I_place):
	tau_I = 0.5 # s
	currents_update = np.zeros(network_size)
	for i in range(network_size):
		for j in range(8):
			pass
	for i in range(network_size):
		sum_w_r = 0
		for j in range(8):
			if is_computable(i, j):
				neighbour = neighbour_index(i, j)
				sum_w_r += weights[i, j] * rates[neighbour]
		currents_update[i] = currents[i] + (-currents[i] / tau_I + intrinsic_e[i] * sum_w_r - I_inh - I_theta +
		                                    I_place[i]) * delta_t

	return currents_update


def update_intrinsic_e(intrinsic_e_t, intrinsic_e_r, delta_t, rates):
	tau_e = 120 # s
	eta = 1
	intrinsic_e_update = np.zeros(network_size)
	intrinsic_e_t_update = intrinsic_e_t.copy()
	intrinsic_e_r_update = intrinsic_e_r.copy()
	for i in range(network_size):
		r_e = intrinsic_e_r[i] * np.exp(-intrinsic_e_t[i] / tau_e)
		if rates[i] > r_e:
			intrinsic_e_r_update[i] = rates[i]
			intrinsic_e_t_update[i] = 0
		else:
			intrinsic_e_r_update[i] = r_e
			intrinsic_e_t_update[i] += delta_t
		intrinsic_e_update[i] = eta * intrinsic_e_r_update[i] * np.exp(-intrinsic_e_t_update[i] / tau_e)

	return intrinsic_e_update, intrinsic_e_t_update, intrinsic_e_r_update
	# return np.ones(network_size)

def update_I_inh(I_inh, delta_t, w_inh, rates):
	tau_inh = 0.5 # s
	sum_rates = 0
	for i in range(network_size):
		sum_rates += rates[i]

	return (-I_inh / tau_inh + w_inh * sum_rates) * delta_t + I_inh


def compute_place_cell_activities(coord_x, coord_y, reward, movement = False):
	'''

	:param coord_x: float, the x coordinate (m)
	:param coord_y: float, the y coordinate (m)
	:param reward: float, the reward value. If reward != 0, the agent should be resting and the C parameter set
	to 1 Hz
	:param movement: bool, indicates whether the robot moved in the current time step or not
	:return: numpy array, vector of the networks place cell activities
	'''

	d = 0.1  # m
	no_cells_per_m = int(np.sqrt(network_size)) / 2
	no_cell_it = int(np.sqrt(network_size))  # the number of cells along one row of the network
	if movement or reward != 0:
		C = 40 # Hz
	else:
		C = 0 # Hz
	cells_activity = np.zeros((no_cell_it, no_cell_it))
	place = np.array((coord_x + 1, coord_y + 1))
	for i in range(no_cell_it):
		for j in range(no_cell_it):
			place_cell_field_location = np.array(((i / no_cells_per_m), (j / no_cells_per_m)))
			cells_activity[i][j] = C * np.exp(
				-1.0 / (2.0 * d ** 2.0) * np.dot((place - place_cell_field_location),
				                                 (place - place_cell_field_location)))
	cell_activities_array = cells_activity.flatten()

	return cell_activities_array


# set constants
network_size = 100
rho = 1
epsilon = 2  # Hz min threshold
theta = 40  # Hz max threshold
delta_t = 0.1 # s
w_inh = 0

# set variable initial conditions
rates = np.zeros(network_size)
rates_next = np.zeros(network_size)

currents = np.zeros(network_size)
currents_next = np.zeros(network_size)

intrinsic_e = np.ones(network_size)
intrinsic_e_next = np.zeros(network_size)
intrinsic_e_t = np.zeros(network_size)
intrinsic_e_t_next = np.zeros(network_size)
intrinsic_e_r = np.zeros(network_size)
intrinsic_e_r_next = np.zeros(network_size)

network_weights = initialise_weights()
network_weights_next = np.zeros((network_size, 8))

I_place = 0
I_inh = 0
I_inh_next = 0
I_theta = 0

reward_val = 0

# get trajectory data
trajectory = np.load("data/trajectory_data_new.npy")
place_cell_list = []
currents_list = []
I_inh_list = []
intrinsic_e_list = []
rates_list = []
for step in range(1, np.size(trajectory, 0)):
	movement_x = trajectory[step, 0] - trajectory[step-1, 0]
	movement_y = trajectory[step, 1] - trajectory[step - 1, 1]
	if movement_x > 0.002 or movement_y > 0.002:
		movement = True
	else:
		movement = False
	if step == np.size(trajectory, 0) - 1:
		movement = True
		reward_val = 1

	# set variables at the next time step to the ones now
	network_weights = network_weights_next.copy()
	rates = rates_next.copy()
	currents = currents_next.copy()
	intrinsic_e = intrinsic_e_next.copy()
	intrinsic_e_t = intrinsic_e_t_next.copy()
	intrinsic_e_r = intrinsic_e_r_next.copy()

	I_inh = I_inh_next

	# update all the network variables
	I_place = compute_place_cell_activities(trajectory[step, 0], trajectory[step, 1], reward_val, movement)
	I_inh_next = update_I_inh(I_inh, delta_t, w_inh, rates)
	currents_next = update_currents(currents, delta_t, intrinsic_e, network_weights, rates, I_inh, 0, I_place)
	intrinsic_e_next, intrinsic_e_t_next, intrinsic_e_r_next = update_intrinsic_e(intrinsic_e_t,
	                                                                              intrinsic_e_r, delta_t, rates)
	rates_next = compute_rates(currents_next)

	place_cell_list.append(I_place)
	rates_list.append(rates_next)
	currents_list.append(currents_next)
	intrinsic_e_list.append(intrinsic_e_next)
	I_inh_list.append(I_inh_next)

# now I need to run a replay event here once the forward trajectory has completed
I_p = compute_place_cell_activities(trajectory[-1, 0], trajectory[-1, 1], reward=1, movement=False) * 20
for replay_step in range(100):
	if replay_step < 30: # place pulses every 1 sec
		I_place = I_p
	else:
		I_place = np.zeros(network_size)
	# set variables at the next time step to the ones now
	network_weights = network_weights_next.copy()
	rates = rates_next.copy()
	currents = currents_next.copy()
	intrinsic_e = intrinsic_e_next.copy()
	intrinsic_e_t = intrinsic_e_t_next.copy()
	intrinsic_e_r = intrinsic_e_r_next.copy()

	I_inh = I_inh_next

	# update all the network variables
	I_inh_next = update_I_inh(I_inh, delta_t, w_inh, rates)
	currents_next = update_currents(currents, delta_t, intrinsic_e, network_weights, rates, I_inh, 0, I_place)
	intrinsic_e_next, intrinsic_e_t_next, intrinsic_e_r_next = update_intrinsic_e(intrinsic_e_t,
	                                                                              intrinsic_e_r, delta_t, rates)
	rates_next = compute_rates(currents_next)

	place_cell_list.append(I_place)
	rates_list.append(rates_next)
	currents_list.append(currents_next)
	intrinsic_e_list.append(intrinsic_e_next)
	I_inh_list.append(I_inh_next)

def plot_func(j):
	# This function is passed into the FuncAnimation class, which then provides the live plotting
	print(j)
	print(I_inh_list[j])
	vals = intrinsic_e_list[j]
	vals = np.reshape(vals, (10, 10))
	plot = np.flip(np.transpose(vals), 0)
	# plot = np.transpose(self.network_activity_series[i])
	# plot = self.network_activity_series[i]
	plt.clf()
	# plt.contourf(plot, cmap=plt.cm.hot)
	# plt.cla()
	plt.imshow(plot, cmap='hot', interpolation='nearest', vmin=0, vmax=20 ) #
	plt.colorbar()
	plt.draw()
	# plt.pause(0.1)

ani = FuncAnimation(plt.gcf(), plot_func, interval=100, frames=range(50,199), repeat=False)

plt.show()
