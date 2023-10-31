import matplotlib.pyplot as plt
import pymc as pm

lambd = 20
mu = 2
sigma = 0.5
real_alpha = 3

real_model = pm.Model()

with real_model:
    num_clients = pm.Poisson("num_clients", lambd)
    plating_payment_times = pm.Normal("plating_payment_times", mu, sigma)
    cooking_times = pm.Exponential("cooking_times", real_alpha)
    total_service_times = pm.Deterministic(
        "total_service_times", plating_payment_times + cooking_times
    )

with real_model:
    data = pm.sample(100, random_seed=2, return_inferencedata=False)

model = pm.Model()

with model:
    alpha = pm.Exponential("alpha", lam=1 / real_alpha)

    num_clients = pm.Poisson("num_clients", lambd)
    plating_payment_times = pm.Normal("plating_payment_times", mu, sigma)
    cooking_times = pm.Exponential("cooking_times", alpha)
    total_service_times = pm.Deterministic(
        "total_service_times", plating_payment_times + cooking_times
    )

with model:
    trace = pm.sample(100, random_seed=2, return_inferencedata=False)

print("Estimarea pentru alpha:", trace["alpha"].mean())

plt.hist(
    trace["alpha"],
    bins=30,
    density=True,
    alpha=0.5,
    color="b",
)
plt.xlabel("alpha")
plt.legend()
plt.show()
