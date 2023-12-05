import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


def read_data():
    df = pd.read_csv("Admission.csv")
    admission = df["Admission"].values.astype(int)
    gre = df["GRE"].values.astype(int)
    gpa = df["GPA"].values.astype(float)

    return np.array(admission), np.array(gre), np.array(gpa)


def ex1(admission, gre, gpa):
    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=0, sigma=10)
        beta1 = pm.Normal("beta1", mu=0, sigma=10)
        beta2 = pm.Normal("beta2", mu=0, sigma=10)

        mu = beta0 + pm.math.dot(beta1, gre) + pm.math.dot(beta2, gpa)
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
        bd = pm.Deterministic("bd", -beta0 / beta2 - beta1 / beta2 * gre)

        y = pm.Bernoulli("y", p=theta, observed=admission)

        idata = pm.sample(1000, return_inferencedata=True)

    posterior = idata.posterior.stack(sample=("chain", "draw"))

    idx = np.argsort(gre)
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]
    plt.scatter(gre, gpa, c=admission, cmap="bwr")
    az.plot_hdi(gre[idx], bd, color="k")
    plt.xlabel("GRE")
    plt.ylabel("GPA")

    # theta = posterior["theta"].mean("sample")
    # idx = np.argsort(gre)
    # plt.plot(gre[idx], theta[idx], color="C2", lw=3)
    # plt.scatter(gre, admission, marker="o", color="C0", s=100)
    # plt.xlabel("GRE")
    # plt.ylabel("Admission")
    # plt.show()


admission, gre, gpa = read_data()
print(admission)
print(gre)
print(gpa)

ex1(admission, gre, gpa)
