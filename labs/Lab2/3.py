import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)


def generatie():
    output = [""] * 10
    for i in range(10):
        if np.random.random() < 0.5:
            output[i] = "s"
        else:
            output[i] = "b"
        if np.random.random() < 0.3:
            output[i] += "s"
        else:
            output[i] += "b"
    return output
