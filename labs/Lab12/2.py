import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error


N_values = [100, 1000, 10000]
num_simulations = 100
errors = []

for N in N_values:
    error_simulations = [estimate_pi(N) for _ in range(num_simulations)]
    errors.append(error_simulations)

mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

plt.errorbar(N_values, mean_errors, yerr=std_errors, fmt="o-", capsize=5)
plt.xscale("log")
plt.xlabel("Numărul de puncte (N)")
plt.ylabel("Eroare (%)")
plt.title("Estimarea lui π cu erori pentru diferite valori de N")
plt.show()


# Observatie:
# Cu cat creste N, cu atat scade eroarea.
