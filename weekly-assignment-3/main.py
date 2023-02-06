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
    axis[0].scatter(data["phi"], data["psi"], marker=".")
    axis[0].set_xlabel("phi")
    axis[0].set_ylabel("psi")
    axis[1].hist2d(data["phi"], data["psi"], bins=20)

    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.show()


if __name__ == "__main__":
    main()
