import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

lambda1 = 4
lambda2 = 6

prob1 = 0.4
prob2 = 1 - prob1

size = 10000
size1 = int(size * prob1)
size2 = int(size * prob2)

m1 = stats.expon.rvs(0, 1 / lambda1, size=size1)
m2 = stats.expon.rvs(0, 1 / lambda2, size=size2)

total = np.concatenate((m1, m2))

az.plot_posterior({"total": total})
plt.show()
