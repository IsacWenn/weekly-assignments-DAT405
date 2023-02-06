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
    show_scatter_and_histo2d_plot(data, X)



def get_kmeans_centerpoints(X, n: int):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
    return kmeans.cluster_centers_


def show_scatter_and_histo2d_plot(data, X):
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

    centers = get_kmeans_centerpoints(X, 3)

    print(centers)
    xs = np.array([])
    ys = np.array([])

    # plt.plot([3, 4], [6, 7], marker='*', ls='none', ms=20)

    for coord in centers:
        xs = np.append(xs, coord[0])
        ys = np.append(ys, coord[1])
    print(xs)
    print(ys)

    axis[0].plot(xs, ys, marker='o', color='red', ls='none')
    axis[1].plot(xs, ys, marker='o', color='red', ls='none')

    plt.show()


if __name__ == "__main__":
    main()
