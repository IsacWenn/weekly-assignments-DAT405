import pandas
import numpy
import matplotlib.pyplot as plt
import mplcursors

YEAR = 2020

def filter_csv_data_by_year(pandas_csv_data: pandas.DataFrame, year: int):
    return pandas_csv_data[pandas_csv_data["Year"] == year].drop(columns=["Year", "Code"])

def construct_scatter_plot(gdp_data: pandas.DataFrame, life_expectancy_data: pandas.DataFrame):
    data = gdp_data.merge(life_expectancy_data)
    for row in data.iterrows():
        row = row[1] # Converts row from a Tuple[Hashable, Series] to a Series
        plt.scatter(row["GDPPC"], row["Life expectancy"], marker=".")

def main():

    #
    # Reading the data.
    #
    gdp_data = pandas.read_csv("data/gdp-per-capita-worldbank.csv")
    life_expectancy_data = pandas.read_csv("data/life-expectancy.csv")

    #
    # Filtering the data to only include data points from a specific year.
    #
    gdp_data_year = filter_csv_data_by_year(gdp_data, YEAR)
    life_expectancy_data_year = filter_csv_data_by_year(life_expectancy_data, YEAR)

    #
    # Constructing a scatter plot with the filtered data.
    #
    construct_scatter_plot(gdp_data_year, life_expectancy_data_year)

    #
    # Finishing touches and details for the plot.
    #
    plt.xlabel("GDP per capita")
    plt.ylabel("Life expectancy at birth")
    plt.show()


if __name__ == "__main__":
    main()
