import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

lambd = 20
mu = 2
sigma = 0.5


def generate(alpha):
    total_service_times_all = []
    for _ in range(100):
        num_clients = np.random.poisson(lambd)
        plating_payment_times = np.random.normal(mu, sigma, num_clients)
        cooking_times = np.random.exponential(alpha, num_clients)
        total_service_times = plating_payment_times + cooking_times
        total_service_times_all.extend(total_service_times)
    return total_service_times_all


def is_good_enough(alpha):
    times = generate(alpha)
    under_15 = 0
    for time in times:
        if time <= 15:
            under_15 += 1
    if under_15 / len(times) >= 0.95:
        return True
    return False


low, high = 0.1, 15
while low < high:
    mid = (low + high) / 2
    if is_good_enough(mid):
        low = mid
    else:
        high = mid

alpha = high
total_service_times_all = generate(alpha)

under_15 = 0
for time in total_service_times_all:
    if time <= 15:
        under_15 += 1

print("Valoarea pentru alpha:", alpha)
print(
    "Probabilitatea ca timpul total de servire sa fie sub 15 minute:",
    under_15 / len(total_service_times_all),
)

plt.hist(total_service_times_all, bins=20, density=True, alpha=0.6, color="g")
plt.xlabel("Timp Total de Servire (minute)")
plt.ylabel("Probabilitate")
plt.title("Distribuția Timpului Total de Servire")
plt.show()
