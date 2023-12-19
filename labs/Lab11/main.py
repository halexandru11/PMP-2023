import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
mix = np.concatenate(
    [
        np.random.normal(loc=means[i], scale=std_devs[i], size=n_cluster[i])
        for i in range(clusters)
    ]
)

with pm.Model() as model:
    k = pm.DiscreteUniform("k", lower=2, upper=5)
    means = pm.Normal("means", mu=0, sd=10, shape=k)
    std_devs = pm.HalfNormal("std_devs", sd=10, shape=k)
    weights = pm.Dirichlet("weights", a=np.ones(k))
    category = pm.Categorical("category", p=weights, shape=n_total)
    observation = pm.Normal(
        "observation", mu=means[category], sd=std_devs[category], observed=mix
    )
    trace = pm.sample(2000, tune=1000)

az.plot_trace(trace)
plt.show()

az.plot_posterior(trace, var_names=["k", "means", "std_devs", "weights"])
plt.show()
