#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_func(j):
	# This function is passed into the FuncAnimation class, which then provides the live plotting

	try:
		rates = np.load('data/rates_data.npy', allow_pickle=True)
		# vals = np.load('data/place_data.npy', allow_pickle=True)
		intrinsic_es = np.load('data/intrinsic_e.npy', allow_pickle=True)
	except:
		rates = np.zeros(1)
		intrinsic_es = np.zeros(1)
	if np.size(rates) == 100 and np.size(intrinsic_es) == 100:
		# pass
		plot = np.reshape(rates, (10, 10))
		plt.clf()
		# plt.contourf(plot, cmap=plt.cm.hot)
		# plt.cla()
		plt.imshow(plot, cmap='hot', origin='lower', interpolation='nearest', vmin=0, vmax=50)
		plt.xticks(range(0,10), range(1,11))
		plt.yticks(range(0, 10), range(1, 11))
		plt.colorbar()
		plt.title('Network Rates')
		plt.draw()
		# plt.pause(0.1)

ani = FuncAnimation(plt.gcf(), plot_func, interval=10)

plt.show()