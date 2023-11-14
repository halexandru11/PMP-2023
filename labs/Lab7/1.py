import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("auto-mpg.csv")
df["horsepower"] = df["horsepower"].replace("?", "0")
df = df[df["horsepower"] != "0"]
df["horsepower"] = df["horsepower"].astype("int64")
df = df.sort_values(by=["horsepower"])
df["mpg"] = df["mpg"].astype("int64")
df = df[["mpg", "horsepower"]]
df.plot.scatter(x="horsepower", y="mpg")
plt.show()
