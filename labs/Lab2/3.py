import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

# np.random.seed(1)


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


def get_prob():
    output = {"ss": 0.0, "sb": 0.0, "bs": 0.0, "bb": 0.0}

    for _ in range(100):
        iter = generatie()
        output["ss"] += iter.count("ss")
        output["sb"] += iter.count("sb")
        output["bs"] += iter.count("bs")
        output["bb"] += iter.count("bb")

    output["ss"] /= 100
    output["sb"] /= 100
    output["bs"] /= 100
    output["bb"] /= 100

    return output


prob = get_prob()

plt.bar(prob.keys(), prob.values())
plt.show()
