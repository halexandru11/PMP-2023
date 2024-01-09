import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def posterior_grid(grid_points=50, heads=6, tails=9, prior_type="uniform"):
    grid = np.linspace(0, 1, grid_points)

    if prior_type == "uniform":
        prior = np.repeat(1 / grid_points, grid_points)
    elif prior_type == "binary":
        prior = (grid <= 0.5).astype(int)
    elif prior_type == "abs_diff":
        prior = abs(grid - 0.5)
    else:
        raise ValueError(
            "Invalid prior_type. Choose 'uniform', 'binary', or 'abs_diff'."
        )

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (12, 4))
points = 12
h = data.sum()
t = len(data) - h

grid_uniform, posterior_uniform = posterior_grid(
    points, h, t, prior_type="uniform"
)
plt.plot(
    grid_uniform, posterior_uniform, "o-", label="Uniform Prior", alpha=0.5
)

grid_binary, posterior_binary = posterior_grid(
    points, h, t, prior_type="binary"
)
plt.plot(grid_binary, posterior_binary, "o-", label="Binary Prior", alpha=0.5)

grid_abs_diff, posterior_abs_diff = posterior_grid(
    points, h, t, prior_type="abs_diff"
)
plt.plot(
    grid_abs_diff, posterior_abs_diff, "o-", label="Abs Diff Prior", alpha=0.5
)

plt.yticks([])
plt.xlabel("θ")
plt.legend()
plt.show()
