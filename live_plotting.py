#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,8))
fig.canvas.set_window_title('Network Plots')

plot_rates = np.zeros((10,10))
plot_intrinsic_es = np.zeros((10,10))

im1 = ax1.imshow(plot_rates, cmap='YlOrRd', origin='lower', interpolation='nearest', vmin=0, vmax=100, animated=True)
ax1.title.set_text('Network Rates')
ax1.set_xticks(range(0,10))
ax1.set_xticklabels(range(1,11))
ax1.set_yticks(range(0,10))
ax1.set_yticklabels(range(1,11))
cbar = fig.colorbar(im1, ax=ax1)
cbar.set_label('Hz', rotation=0, labelpad=5)

im2 = ax2.imshow(plot_intrinsic_es, cmap='Blues', origin='lower', interpolation='nearest', vmin=0, vmax=4)
ax2.title.set_text('Intrinsic Plasticity')
ax2.set_xticks(range(0,10))
ax2.set_xticklabels(range(1,11))
ax2.set_yticks(range(0,10))
ax2.set_yticklabels(range(1,11))
cbar = fig.colorbar(im2, ax=ax2)
cbar.set_label('$\sigma$', rotation=0, labelpad=5)


def updatefig_live(*args):
	global plot_rates, plot_intrinsic_es
	try:
		rates = np.load('data/rates_data.npy', allow_pickle=True)
		if np.size(rates) == 100:
			plot_rates = np.reshape(rates, (10, 10))
	except:
		pass
	try:
		intrinsic_es = np.load('data/intrinsic_e.npy', allow_pickle=True)
		if np.size(intrinsic_es) == 100:
			plot_intrinsic_es = np.reshape(intrinsic_es, (10, 10))
	except:
		pass
	im1.set_array(plot_rates)
	im2.set_array(plot_intrinsic_es)
	return im1, im2


def updatefig_saved(i):
	global rates_series, intrinsic_es_series
	plot_rates = np.reshape(rates_series[i], (10, 10))
	plot_intrinsic_es = np.reshape(intrinsic_es_series[i], (10, 10))
	im1.set_array(plot_rates)
	im2.set_array(plot_intrinsic_es)
	return im1, im2


plot_live = True
if plot_live == True:
	# plot live data
	ani_live = animation.FuncAnimation(fig, updatefig_live, interval=10, blit=True)
else:
	# plots saved data
	time_series = np.load('data/time_series_1.npy')
	rates_series = np.load('data/rates_series_1.npy')
	intrinsic_es_series = np.load('data/intrinsic_e_series_1.npy')

	# ani_saved = animation.FuncAnimation(fig, updatefig_saved, interval=10, blit=True)

	# Line plots of the place cell activities over time used for the LM paper.
	fig_size = (2,6)
	replay_fig, replay_axes = plt.subplots(14, 2, figsize=fig_size, sharey=True)
	replay_fig.canvas.set_window_title('Replay Plots')
	colours = ['purple', 'blue', 'green', 'orange', 'red']

	replay_fig.text(0.03, 0.55, 'Neuron Index', rotation=90, fontsize=9, fontstyle='normal', fontweight='light')
	replay_fig.text(0.45, 0.03, 'Time (s)', rotation=0, fontsize=9, fontstyle='normal', fontweight='light')

	axes_i = 0
	# The total cell numbers that were active during exploration without their temporal order were [11, 12, 21, 22,
	# 23, 33, 34, 41, 44, 45, 51, 54, 55, 65, 75, 85]
	cell_indices = [11, 12, 21, 22, 23, 33, 34, 44, 45, 54, 55, 65, 75, 85]
	print(np.size(time_series))
	for num, i in enumerate(cell_indices):
		# if max(rates_series[:,i]) > 10:
		# 	print(i, np.argmax(rates_series[:,i]), max(rates_series[:,i]))
		replay_axes[axes_i, 0].plot(time_series[:], rates_series[:, i], str(float(axes_i) / 25))
		replay_axes[axes_i, 0].set_xlim((22, 33))
		replay_axes[axes_i, 0].set_ylim((0, 100))
		# line_axes[axes_i].get_yaxis().set_visible(False)
		replay_axes[axes_i, 0].get_yaxis().set_ticks([])
		if i != cell_indices[-1]:
			replay_axes[axes_i, 0].get_xaxis().set_visible(False)
		else:
			replay_axes[axes_i, 0].set_xticks((22, 27, 32))
			replay_axes[axes_i, 0].set_xticklabels((0, 5, 10), size=8)
			# replay_axes[axes_i, 0].set_xlabel('Time (s)')
		replay_axes[axes_i, 0].set_ylabel(str(num+1), labelpad=5, rotation=0, size=8, va='center')
		axes_i += 1

	axes_i = 0
	for num, i in enumerate(cell_indices):
		# if max(rates_series[:,i]) > 10:
			# print(i, np.argmax(rates_series[:,i]), max(rates_series[:,i]))
		replay_axes[axes_i, 1].plot(time_series[:], rates_series[:, i], str(float(axes_i) / 25))
		replay_axes[axes_i, 1].set_ylim((0, 100))
		replay_axes[axes_i, 1].set_xlim((37.6, 38))
		replay_axes[axes_i, 1].get_yaxis().set_ticks([])
		if i != cell_indices[-1]:
			replay_axes[axes_i, 1].get_xaxis().set_visible(False)
		else:
			replay_axes[axes_i, 1].set_xticks((37.6, 37.8, 38))
			replay_axes[axes_i, 1].set_xticklabels((15.6, 15.8, 16), size=8, position=(1,0))
			# replay_axes[axes_i, 1].set_xlabel('Time (s)')
		# replay_axes[axes_i, 1].set_ylabel(str(num + 1), labelpad=5, rotation=0, size=8, va='center')
		axes_i += 1

plt.show()
