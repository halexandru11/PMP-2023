import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


def read_data():
    df = pd.read_csv("Admission.csv")
    df = df.query("Admission == (0, 1)")
    admission = df["Admission"].values.astype(np.int32)
    gre = df["GRE"].values.astype(np.int32)
    gpa = df["GPA"].values.astype(np.float32)
    return np.array(admission), np.array(gre), np.array(gpa)


def predict(gre, gpa, idata):
    alpha = idata.posterior["alpha"].mean(("chain", "draw"))
    beta1 = idata.posterior["beta1"].mean(("chain", "draw"))
    beta2 = idata.posterior["beta2"].mean(("chain", "draw"))
    mu = alpha + gre * beta1 + gpa * beta2
    theta = 1 / (1 + np.exp(-mu))
    theta = np.array(theta)
    admission_prob = az.hdi(theta, hdi_prob=0.9)
    print(f"admission probability: {admission_prob}")


def compute():
    admission, gre, gpa = read_data()

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=20)
        beta1 = pm.Normal("beta1", mu=0, sigma=2)
        beta2 = pm.Normal("beta2", mu=0, sigma=2)

        mu = alpha + pm.math.dot(gre, beta1) + pm.math.dot(gpa, beta2)
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
        boundary = pm.Deterministic(
            "boundary",
            -alpha / beta2 - beta1 / beta2 * gre,
        )

        y = pm.Bernoulli("y", p=theta, observed=admission)

        idata = pm.sample(200, return_inferencedata=True)

    return admission, gre, gpa, idata


def plot_data(admission, gre, gpa, idata):
    idx = np.argsort(gpa)
    boundary = idata.posterior["boundary"].mean(("chain", "draw"))[idx]
    plt.scatter(gre, gpa, c=[f"C{i}" for i in admission])
    plt.plot(gre[idx], boundary, color="k")
    az.plot_hdi(
        gre,
        idata.posterior["boundary"],
        color="k",
        hdi_prob=0.94,
    )
    plt.xlabel("GRE")
    plt.ylabel("GPA")
    plt.show()


def main():
    # ex 1
    admission, gre, gpa, idata = compute()

    # ex 2
    # plot_data(admission, gre, gpa, idata)

    # ex 3
    predict(gre=550, gpa=3.5, idata=idata)

    # ex 4
    predict(gre=500, gpa=3.2, idata=idata)


if __name__ == "__main__":
    main()
