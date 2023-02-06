# Main Python program (Week 3)
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def main():
    data = pandas.read_csv("data/data_assignment3.csv")
    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(data["phi"], data["psi"], marker=".", s=1)
    axis[0].set_xlabel("phi")
    axis[0].set_ylabel("psi")
    axis[1].hist2d(data["phi"], data["psi"], bins=20, cmap="Blues")
    axis[1].set_xlabel("phi")
    axis[1].set_ylabel("psi")

    PCM = None
    ax = plt.gca()  # get the current axes
    for PCM in ax.get_children():
        if isinstance(PCM, plt.cm.ScalarMappable):
            break

    plt.colorbar(PCM, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
