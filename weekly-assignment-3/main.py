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
    # Creating the scatter and hist2d plots.

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=1).fit(X)
    centers = kmeans.cluster_centers_
    y_predict = kmeans.predict(X)

    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(data["phi"], data["psi"], c=y_predict, marker=".", s=1)
    axis[0].set_xlabel("phi")
    axis[0].set_ylabel("psi")
    axis[1].hist2d(data["phi"], data["psi"], bins=10, cmap="Blues")
    axis[1].set_xlabel("phi")
    axis[1].set_ylabel("psi")

    # Adding a colorbar to the hist2d plot.
    PCM = None
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break
    plt.colorbar(PCM, ax=ax)

    # Adding the cluster centers to the scatter plot.
    axis[0].scatter(centers[:, 0], centers[:, 1], c='red', s=20, alpha=0.9)
    axis[1].scatter(centers[:, 0], centers[:, 1], c='red', s=20, alpha=0.9)

    plt.show()

if __name__ == "__main__":
    main()
