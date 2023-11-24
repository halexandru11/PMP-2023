import numpy as np


def ex1():
    # parametrii alesi de mine
    mu = 2
    sigma = 0.5

    # generez 200 timpi de asteptare
    times = np.random.normal(mu, sigma, size=200)
    average_time = times.mean()  # get the average time of the 200 generations
    return average_time
