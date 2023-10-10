import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

gamma1 = (4, 3)
gamma2 = (4, 2)
gamma3 = (5, 2)
gamma4 = (5, 3)
lambda1 = 4
p1 = 0.25
p2 = (1 - p1) * 0.25
p3 = (1 - p1) * (1 - p2) * 0.3
p4 = 1 - p1 - p2 - p3

size = 10000
size1 = int(size * p1)
size2 = int(size * p2)
size3 = int(size * p3)
size4 = int(size * p4)


server1 = stats.gamma.rvs(gamma1[0], 0, 1 / gamma1[1], size=size1)
server2 = stats.gamma.rvs(gamma2[0], 0, 1 / gamma2[1], size=size2)
server3 = stats.gamma.rvs(gamma3[0], 0, 1 / gamma3[1], size=size3)
server4 = stats.gamma.rvs(gamma4[0], 0, 1 / gamma4[1], size=size4)

total = np.concatenate((server1, server2))

az.plot_posterior({"total": total})
plt.show()
