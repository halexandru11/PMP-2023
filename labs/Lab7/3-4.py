import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

df = pd.read_csv("auto-mpg.csv")
df["horsepower"] = df["horsepower"].replace("?", "0")
df["horsepower"] = df["horsepower"].astype("int64")
df = df.sort_values(by=["horsepower"])
df["mpg"] = df["mpg"].astype("int64")
df = df[["mpg", "horsepower"]]

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0.7, sigma=1)
    eps = pm.HalfCauchy("eps", beta=5)
    mu = pm.Deterministic("mu", alpha + beta * df["horsepower"])
    y = pm.Normal(
        "y",
        mu=mu,
        sigma=eps,
        observed=df["mpg"],
    )
    i_data = pm.sample(1000, tune=1000, return_inferencedata=True)

posterior_g = i_data.posterior.stack(sample=("chain", "draw"))
x = df["horsepower"].to_numpy()
alpha_m = posterior_g["alpha"].mean().item()
beta_m = posterior_g["beta"].mean().item()

plt.plot(
    x,
    alpha_m + beta_m * x,
    c="k",
    label=f"y = {alpha_m:.2f} + {beta_m:.2f} * x",
)
sig = az.plot_hdi(x, posterior_g["mu"].T, hdi_prob=0.95, color="k")
plt.xlabel("horsepower")
plt.ylabel("mpg", rotation=0)
plt.legend()
plt.show()
