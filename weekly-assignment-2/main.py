import numpy as np
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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
    model_linear = LinearRegression().fit(x, y)
    # 1b
    # Slope value
    k = float(model_linear.coef_[0][0])
    print('Slope of the line is:', k)
    # Intercept value
    m = float(model_linear.intercept_[0])
    print('Intercept of the line is:', m)

    # 1c
    # Predicting the selling price at 100, 150 and 200 square meters
    print('Selling price at 100m2:', read_linear_line(k, m, 100))
    print('Selling price at 150m2:', read_linear_line(k, m, 150))
    print('Selling price at 200m2:', read_linear_line(k, m, 200))

    # Plotting the regression line
    xfit = np.linspace(0, 250, 1000)  # 1000 evenly spaced points in
    yfit = model_linear.predict(xfit[:, np.newaxis])

    # 2a
    # Loading the iris data set
    iris = load_iris()

    # Creating the logistic regression model
    model_logistic = linear_model.LogisticRegression(max_iter=1000)
    model_logistic.fit(iris.data, iris.target)
    predicted = model_logistic.predict(iris.data)
    # Calculating the confusion matrix
    cm = metrics.confusion_matrix(iris.target, predicted)
    print('Logistic regression matrix: \n', cm)

    # 2b
    # Creating the k-nearest neighbours model
    model_knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto')
    model_knn.fit(iris.data, iris.target)
    predicted_knn = model_knn.predict(iris.data)
    # Calculating the confusion matrix
    cm_knn = metrics.confusion_matrix(iris.target, predicted_knn)
    print('KNN matrix uniform: \n', cm_knn)

    model_knn = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto')
    model_knn.fit(iris.data, iris.target)
    predicted_knn = model_knn.predict(iris.data)
    # Calculating the confusion matrix
    cm_knn = metrics.confusion_matrix(iris.target, predicted_knn)
    print('KNN matrix distance: \n', cm_knn)

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
