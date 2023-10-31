import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

lambd = 20
mu = 2
sigma = 0.5
alpha = 3

model = pm.Model()

with model:
    num_clients = pm.Poisson("num_clients", lambd)
    plating_payment_times = pm.Normal("plating_payment_times", mu, sigma)
    cooking_times = pm.Exponential("cooking_times", alpha)
    total_service_times = pm.Deterministic(
        "total_service_times", plating_payment_times + cooking_times
    )

with model:
    trace = pm.sample(100, random_seed=2, return_inferencedata=False)

plt.hist(
    trace["total_service_times"], bins=20, density=True, alpha=0.6, color="g"
)
plt.xlabel("Timp Total de Servire (minute)")
plt.ylabel("Probabilitate")
plt.title("Distribuția Timpului Total de Servire")
plt.show()
