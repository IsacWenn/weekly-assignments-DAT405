import pandas
import numpy
import matplotlib.pyplot as plt

YEAR = 2020

def filter_csv_data_by_year(pandas_csv_data: pandas.DataFrame, year: int):

    return pandas_csv_data[pandas_csv_data["Year"] == year]


def main():
    gdp_data = pandas.read_csv("data/gdp-per-capita-worldbank.csv")
    life_expectancy_data = pandas.read_csv("data/life-expectancy.csv")

    gdp_data_year = filter_csv_data_by_year(gdp_data, YEAR)
    life_expectancy_data_year = filter_csv_data_by_year(life_expectancy_data, YEAR)

    

    plt.scatter(10, 10)
    plt.show()


if __name__ == "__main__":
    main()
