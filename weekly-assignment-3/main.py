# Main Python program (Week 3)
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
matplotlib.use("TkAgg")


def main():
    # Read the data.
    data = pandas.read_csv("data/data_assignment3.csv")
    # show_scatter_and_histo2d_plot(data)
    X = data.loc[:, ['phi', 'psi']]

    k_range = range(1, 5)
    mean_squared_error = []

    i, j = 0, 0
    figure, axis = plt.subplots(2, 2)
    for k in k_range:
        # create and train model
        kmean_model = KMeans(n_clusters=k, random_state=0, n_init=1).fit(X)

        # predict clusters
        y_pred = kmean_model.predict(X)

        # store mean squared error / k for elbow plot
        mean_squared_error.append(kmean_model.inertia_)

        axis[i][j].scatter(x=X['phi'], y=X['psi'], c=y_pred, cmap="viridis", marker=".", s=1)
        axis[i][j].set_xlabel('phi\n')
        axis[i][j].set_ylabel('psi')
        axis[i][j].set_title(f'k = {k}')

        j += 1
        if j % 2 == 0:
            i += 1
            j = 0

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def show_scatter_and_histo2d_plot(data):
    # Creating the scatter and hist2d plots.
    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(data["phi"], data["psi"], marker=".", s=1)
    axis[0].set_xlabel("phi")
    axis[0].set_ylabel("psi")
    axis[1].hist2d(data["phi"], data["psi"], bins=20, cmap="Blues")
    axis[1].set_xlabel("phi")
    axis[1].set_ylabel("psi")

    # Adding a colorbar to the hist2d plot.
    PCM = None
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)

    plt.show()


if __name__ == "__main__":
    main()
