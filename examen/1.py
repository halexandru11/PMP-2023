import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az


# punctul a)
def get_data():
    # read file Titanic.csv into pandas dataframe
    df = pd.read_csv("Titanic.csv")
    # remove instances with missing data
    df = df.dropna()
    return df


# Titanic.csv format
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 892,0,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
# 893,1,3,"Wilkes, Mrs. James (Ellen Needs)",female,47,1,0,363272,7,,S
# 894,0,2,"Myles, Mr. Thomas Francis",male,62,0,0,240276,9.6875,,Q


def plot_data(data):
    # just o function to plot the data from the csv
    data.hist()
    plt.show()


# punctul b) si d)
def create_model(data):
    # create the pymc model using the two independent variables "Pclass" and "Age" to predict the dependent variable "Survived"

    with pm.Model() as model:
        # define priors
        # alpha to create an offset
        alpha = pm.Normal("alpha", mu=0, sigma=20)
        # beta[0] to control the influence of "Pclass"
        # beta[1] to control the influence of "Age"
        beta = pm.Normal("beta", mu=0, sigma=20, shape=2)
        # sigma to control the variance
        sigma = pm.Uniform("sigma", lower=0, upper=10)

        # define linear regression
        mu = alpha + beta[0] * data["Pclass"] + beta[1] * data["Age"]

        # chance of survival
        survival = pm.Normal(
            "survival", mu=mu, sigma=sigma, observed=data["Survived"]
        )

        # inference
        idata = pm.sample(1000)

    # plot the posterior distributions
    # commented because i was running the code and the plot would show up
    # following 2 lines should not be commented, they are correct
    # pm.plot_posterior(idata)
    # plt.show()

    wanted_age = 30
    wanted_class = 2
    posterior_g = idata.posterior.stack(samples={"chain", "draw"})
    mu = (
        posterior_g["beta"]
        + wanted_class * posterior_g["beta"][0]
        + wanted_age * posterior_g["beta"][1]
    )
    az.plot_posterior(mu.values, hdi_prob=0.90)
    plt.show()


data = get_data()
create_model(data)

# Punctul c) din figura 1_b.png se observă că media pentru beta0(coeficientul pentru Pclass) este mai mica
#           decat media beta1(coeficientul pentru Age) asadar Age influenteaza mai mult rezultatul (daca
#           pasagerul a supravietuit sau nu)
