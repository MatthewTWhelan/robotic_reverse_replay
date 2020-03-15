#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,8))
fig.canvas.set_window_title('Network Plots')

plot_rates = np.zeros((10,10))
plot_intrinsic_es = np.zeros((10,10))

im1 = ax1.imshow(plot_rates, cmap='YlOrRd', origin='lower', interpolation='nearest', vmin=0, vmax=50, animated=True)
ax1.title.set_text('Network Rates')
ax1.set_xticks(range(0,10))
ax1.set_xticklabels(range(1,11))
ax1.set_yticks(range(0,10))
ax1.set_yticklabels(range(1,11))
cbar = fig.colorbar(im1, ax=ax1)
cbar.set_label('Hz', rotation=0, labelpad=5)

im2 = ax2.imshow(plot_intrinsic_es, cmap='Blues', origin='lower', interpolation='nearest', vmin=0, vmax=6)
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


plot_live = False
if plot_live == True:
	# plot live data
	ani_live = animation.FuncAnimation(fig, updatefig_live, interval=10, blit=True)
else:
	# plots saved data
	time_series = np.load('data/time_series.npy')
	rates_series = np.load('data/rates_series.npy')
	intrinsic_es_series = np.load('data/intrinsic_e_series.npy')
	ani_saved = animation.FuncAnimation(fig, updatefig_saved, interval=10, blit=True)

plt.show()
