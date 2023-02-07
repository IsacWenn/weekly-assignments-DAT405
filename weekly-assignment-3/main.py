# Main Python program (Week 3)
import numpy
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.metrics import silhouette_score

matplotlib.use("TkAgg")


def main():
    # Read the data.
    data = pandas.read_csv("data/data_assignment3.csv")
    # show_scatter_and_histo2d_plot(data)
    X = data.loc[:, ['phi', 'psi']]

    #kmeans_scatterplot_and_hist2d(X, data)

    #elbow_method(X, 11)

    #nearest_neighbour_dbScan(X, 2)

    #sillhouette_score_kmeans(X)

    dbScan_2(X, 0.2, 200, data) # 0.1, 90

    #dbScan_specific(data, 'PRO', 0.5, 200)

    # nearest_neighbour_dbScan(X, 3)

    D = data.loc[:, ['residue name']].drop_duplicates()
    #print(D)


def sillhouette_score_kmeans(X):
    score_list = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_, random_state=0)
        score_list.append(score)

    plt.plot(k_range, score_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.show()


def dbScan_specific(data, residue_name : str,  eps, min_samples):
    # Creating the DBSCAN model.
    df_specific = data[(data['residue name'] == f'{residue_name}')]
    X = df_specific[['phi', 'psi']]
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    number_of_outliers = list(labels).count(-1)
    y_predict = db.fit_predict(X)

    # create
    plt.scatter(X[:, 0], X[:, 1], c=y_predict, cmap="viridis", s=3)
    plt.title(f'DBSCAN {residue_name} eps={eps} min_samples={min_samples}')
    plt.show()


def dbScan_2(X, eps, min_samples, data):
    # Creating the DBSCAN model.
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


    # TODO: BarGraph numbering the amount of outliers given a protein type


    residue_data = data.loc[:, ['residue name']]
    residue_data['outlier'] = list(labels)
    residue_data = residue_data.query("`outlier` == -1")

    residue_data = residue_data['residue name'].value_counts(sort=True).plot.bar()
    plt.show()



    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=1,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()


def dbScan(X, eps, min_samples):
    # Creating the DBSCAN model.
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=3,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=1,
        )
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()


def kmeans_scatterplot_and_hist2d(X, data):
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


def elbow_method(X, n: int):
    distortions = []
    K = range(1, n)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=0, n_init=1)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()


def nearest_neighbour_dbScan(X, n: int):
    nn_model = NearestNeighbors(n_neighbors=n).fit(X)
    distances, _ = nn_model.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.ylabel('eps')
    plt.xlabel('distance to nearest n points')
    plt.show()


if __name__ == "__main__":
    main()
