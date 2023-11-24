import numpy as np
import pymc as pm
import matplotlib.pyplot as plt


def ex1():
    # parametrii alesi de mine
    mu = 3
    sigma = 0.5

    # generez 200 timpi de asteptare
    times = np.random.normal(mu, sigma, size=200)
    average_time = times.mean()  # get the average time of the 200 generations
    return times, average_time


def ex2():
    # la fel ca sus, doar ca cu pymc
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=3, sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=1)
        times = pm.Normal("times", mu=mu, sigma=sigma, observed=ex1()[0])
        trace = pm.sample(1000, tune=1000, return_inferencedata=False)

    plt.show()


ex2()
