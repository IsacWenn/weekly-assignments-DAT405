import numpy as np
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.datasets import load_iris
matplotlib.use("TkAgg")


def read_linear_line(k: float, m: float, steps: float):
    return m + k*steps


def main():
    # Reading the data
    data = pandas.read_csv("data/data_assignment2.csv")

    # 1a
    # Converting the data to a numpy array and reshaping it in order to use it in the linear regression model
    array = numpy.vstack([data['Living_area'], data['Selling_price']])
    x = array[0]
    y = array[1]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Fitting the linear regression model
    model = LinearRegression().fit(x, y)
    # 1b
    # Slope value
    k = float(model.coef_[0][0])
    print(k)
    # Intercept value
    m = float(model.intercept_[0])
    print(m)

    # 1c
    # Predicting the selling price at 100, 150 and 200 square meters
    print(read_linear_line(k, m, 100))
    print(read_linear_line(k, m, 150))
    print(read_linear_line(k, m, 200))

    # Plotting the regresion line
    xfit = np.linspace(0, 250, 1000)  # 1000 evenly spaced points in
    yfit = model.predict(xfit[:, np.newaxis])

    # Making the scatter plot
    plt.scatter(data['Living_area'], data['Selling_price'], marker=".")

    # Plotting the regression line
    plt.plot(xfit, yfit, color='red')

    # Adding labels
    plt.xlabel("Living area")
    plt.ylabel("Selling price")
    plt.show()


if __name__ == "__main__":
    main()
