import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]


# Definim modelul PyMC3
with pm.Model() as model:
    n = pm.Poisson("n", mu=10)

    for Y in Y_values:
        for theta in theta_values:
            # Distribuția Binomială dată fiind n și θ
            Y_observed = pm.Binomial(
                f"Y_observed:{Y}{theta}", n=n, p=theta, observed=Y
            )

            # Am colectat toate observațiile într-un array
            observed_data = [Y_observed]

            # Folosim Metropolis-Hastings pentru a obține distribuția a posteriori pentru n
            trace = pm.sample(2000, tune=1000)

            # write the title for each plot
            az.plot_posterior(
                trace,
                var_names=["n"],
                point_estimate="mean",
            )
            plt.title(f"Y = {Y}, theta = {theta}")


plt.show()
