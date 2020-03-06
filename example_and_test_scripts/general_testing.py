import numpy as np
import matplotlib.pyplot as plt

# r_max_array = np.arange(0, 50, 0.1)
#
# sigma_max = 4
# r_sigma = 10
# beta = 1
# tau_sigma = 600 # s
# cells_sigma_max = 1 + (sigma_max - 1) / (1 + np.exp(-beta * (r_max_array - r_sigma)))
#
# plt.plot(r_max_array, cells_sigma_max)
# plt.show()

sim_time = 1000 # s
delta_t = 0.1 # s
tau = 60 # s
H = 1
sigma = np.zeros(int(sim_time / delta_t))
# for i in range(1000 - 1):
# 	sigma[i+1] = delta_t / tau * (-sigma[i] + H) + sigma[i]

for i in range(int(sim_time / delta_t) - 1):
	if i > 5000:
		H = 0
	if i > 6000:
		H = 1
	sigma[i + 1] = delta_t * (-sigma[i] / tau + H) + sigma[i]
	if sigma[i+1] > 6:
		sigma[i + 1] = 6

plt.plot(sigma)
plt.show()