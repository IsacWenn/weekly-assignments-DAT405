import numpy as np
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
matplotlib.use("TkAgg")


def read_linear_line(k: float, m: float, steps: float):
    return m + k*steps


def main():
    data = pandas.read_csv("data/data_assignment2.csv")

    array = numpy.vstack([data['Living_area'], data['Selling_price']])
    x = array[0]
    y = array[1]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    k = float(model.coef_[0][0])
    print(k)
    print(type(k))
    m = float(model.intercept_[0])
    print(m)
    print(type(m))

    print(read_linear_line(k,m,150))

    # Plotting the regresion line
    xfit = np.linspace(0, 200, 1000)  # 1000 evenly spaced points in
    yfit = model.predict(xfit[:, np.newaxis])

    plt.scatter(data['Living_area'], data['Selling_price'], marker=".")
    plt.plot(xfit, yfit, color='red')
    plt.xlabel("Living area")
    plt.ylabel("Selling price")
    plt.show()


if __name__ == "__main__":
    main()
