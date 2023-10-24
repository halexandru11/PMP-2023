import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

lambd = 20
mu = 2
sigma = 0.5
alpha = 4.33  # valoare obtinuta de la 2.py


def generate(alpha):
    total_service_times_all = []
    for _ in range(100):
        num_clients = np.random.poisson(lambd)
        plating_payment_times = np.random.normal(mu, sigma, num_clients)
        cooking_times = np.random.exponential(alpha, num_clients)
        total_service_times = plating_payment_times + cooking_times
        total_service_times_all.extend(total_service_times)
    return total_service_times_all


total_service_times_all = generate(alpha)

plt.hist(total_service_times_all, bins=20, density=True, alpha=0.6, color="g")
plt.xlabel("Timp Total de Servire (minute)")
plt.ylabel("Probabilitate")
plt.title("Distribuția Timpului Total de Servire")
plt.show()
