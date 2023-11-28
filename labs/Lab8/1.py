import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az


def read_data():
    df = pd.read_csv("Prices.csv")
    price = df["Price"].values.astype("float")
    speed = df["Speed"].values.astype("float")
    hard_drive = np.log(df["HardDrive"].values.astype("float"))
    return price, speed, hard_drive


def plot_data(price, speed, hard_drive):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(speed, hard_drive, price)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Hard Drive")
    ax.set_zlabel("Price")
    plt.show()


def point_a(price, speed, hard_drive, plot=True):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta1 = pm.Normal("beta1", mu=0, sigma=10)
        beta2 = pm.Normal("beta2", mu=0, sigma=10)
        sigma = pm.Uniform("sigma", lower=0, upper=10)
        mu = alpha + beta1 * speed + beta2 * hard_drive
        price_pred = pm.Normal(
            "price_pred", mu=mu, sigma=sigma, observed=price
        )
        idata = pm.sample(200, return_inferencedata=True)

        if plot:
            az.plot_trace(
                idata, var_names=["alpha", "beta1", "beta2", "sigma"]
            )
            # az.plot_posterior(pm.sample(400))
            plt.show()

    return model, idata, price_pred


def point_b(idata, model, price, speed, hard_drive, print_hdi=True):
    # beta1_hdi = az.hdi(idata.posterior.beta1, hdi_prob=0.95)
    # beta2_hdi = az.hdi(idata.posterior.beta2, hdi_prob=0.95)
    #
    # if print_hdi:
    #     print("-" * 100)
    #     print(beta1_hdi)
    #     print("beta1: ", np.mean(idata.posterior.beta1))
    #     print("-" * 100)
    #     print(beta2_hdi)
    #     print("beta2: ", np.mean(idata.posterior.beta2))

    posterior_data = idata["posterior"]
    alpha_m = posterior_data["alpha"].mean().item()
    beta1_m = posterior_data["beta1"].mean().item()
    beta2_m = posterior_data["beta2"].mean().item()

    # plt.scatter(speed, price, marker="o")
    # plt.xlabel("Speed")
    # plt.ylabel("Price")
    # plt.plot(speed, alpha_m + beta1_m * speed, color="gray")
    #
    # ppc = pm.sample_posterior_predictive(idata, model=model)
    # posterior_predictive = ppc["posterior_predictive"]
    # az.plot_hdi(speed, posterior_predictive["price_pred"], hdi_prob=0.95)
    # plt.show()

    # plot the same as above, but include hard_drive in a 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(speed, hard_drive, price)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Hard Drive")
    ax.set_zlabel("Price")
    ax.plot(
        speed, hard_drive, posterior_predictive["price_pred"], hdi_prob=0.95
    )

    ppc = pm.sample_posterior_predictive(idata, model=model)
    posterior_predictive = ppc["posterior_predictive"]
    az.plot_hdi(
        price,
        posterior_predictive["price_pred"],
        hdi_prob=0.95,
        fill_kwargs={"alpha": 0.5},
        color="gray",
    )
    plt.show()


def point_c(idata, model):
    # alpha_m = idata["alpha"].mean().item()
    # beta1_m = idata["beta1"].mean().item()
    # beta2_m = idata["beta2"].mean().item()

    # price, speed, hard_drive = read_data()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(speed, hard_drive, price)
    # ax.set_xlabel("Speed")
    # ax.set_ylabel("Hard Drive")
    # ax.set_zlabel("Price")
    #
    # ppc = pm.sample_posterior_predictive(idata, model=model)
    # posterior_predictive = ppc["posterior_predictive"]
    # az.plot_hdi(
    #     price,
    #     posterior_predictive["price_pred"],
    #     hdi_prob=0.95,
    #     fill_kwargs={"alpha": 0.5},
    #     color="gray",
    # )

    # plot the 95% HDI for the mean of the posterior predictive in 3d
    # ax.plot(
    #     speed,
    #     hard_drive,
    #     posterior_predictive["price_pred"].mean(axis=0),
    #     color="gray",
    #     alpha=0.5,
    # )

    # plt.show()
    pass


def main():
    price, speed, hard_drive = read_data()
    # plot_data(price, speed, hard_drive)
    model, idata, price_pred = point_a(price, speed, hard_drive, plot=False)
    point_b(idata, model, price, speed, hard_drive)
    # point_c(idata, model)


if __name__ == "__main__":
    np.random.seed(1)
    main()
