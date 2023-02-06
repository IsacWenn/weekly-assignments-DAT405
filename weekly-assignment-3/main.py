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
    center_array = np.array([])
    for i in range(2, 10):
        add_kmeans_centerpoints(X, i, center_array)
    plt.scatter(center_array[:, 0], center_array[:, 1], marker=".", s=1)
    plt.show()

def add_kmeans_centerpoints(X, n: int, center_array: np.array):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)



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
