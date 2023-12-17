import numpy as np
import matplotlib.pyplot as plt
import pymc as pm


dummy_data = np.loadtxt("./data/dummy.csv", delimiter=",", skiprows=1)
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel("x")
plt.ylabel("y")

with pm.Model() as model_p:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
    sigma = pm.HalfNormal("sigma", 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=y_1s)
    idata_p = pm.sample(200, return_inferencedata=True)

alpha_p_post = idata_p.posterior["alpha"].mean(("chain", "draw")).values
beta_p_post = idata_p.posterior["beta"].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], "C2", label=f"model order {order}")

plt.show()
