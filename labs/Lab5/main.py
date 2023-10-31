import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("trafic.csv")
trafic_observed = data["nr. masini"].values
intervale_orare = [4 * 60, 7 * 60, 8 * 60, 16 * 60, 19 * 60, 24 * 60]
for i in range(len(intervale_orare)):
    intervale_orare[i] -= 4 * 60

with pm.Model() as model:
    lambdas = []
    for i in range(len(intervale_orare) - 1):
        lambda_i = pm.Exponential("lambda_{}".format(i), lam=1)
        lambdas.append(lambda_i)
        trafic_i = pm.Poisson(
            "trafic_{}".format(i),
            mu=lambda_i,
            observed=trafic_observed[
                intervale_orare[i] : intervale_orare[i + 1]
            ],
        )

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step, return_inferencedata=False)

plt.figure(figsize=(12, 6))

for i, lambda_i in enumerate(lambdas):
    plt.subplot(2, 3, i + 1)
    plt.hist(
        trace["lambda_{}".format(i)],
        bins=60,
        density=True,
        alpha=0.6,
        color="blue",
    )
    plt.title(
        "Interval Orar {}-{}".format(
            int(intervale_orare[i] / 60 + 4),
            int(intervale_orare[i + 1] / 60 + 4),
        )
    )
    plt.xlabel("λ")
    plt.ylabel("Densitatea")


plt.tight_layout()
plt.show()
