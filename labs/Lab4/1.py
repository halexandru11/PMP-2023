import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

lambd = 20
mu = 2
sigma = 0.5
alpha = 0.8


total_service_times_all = []
for i in range(100):
    num_clients = np.random.poisson(lambd)
    plating_payment_times = np.random.normal(mu, sigma, num_clients)
    cooking_times = np.random.exponential(alpha, num_clients)
    total_service_times = plating_payment_times + cooking_times
    total_service_times_all.extend(total_service_times)


plt.hist(total_service_times_all, bins=20, density=True, alpha=0.6, color="g")
plt.xlabel("Timp Total de Servire (minute)")
plt.ylabel("Probabilitate")
plt.title("Distribuția Timpului Total de Servire")
plt.show()
