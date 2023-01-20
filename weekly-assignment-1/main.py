import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

YEAR = 2020


def filter_csv_data_by_year(pandas_csv_data: pandas.DataFrame, year: int):
    return pandas_csv_data[pandas_csv_data["Year"] == year].drop(columns=["Year", "Code"])


def construct_scatter_plot(gdp_data: pandas.DataFrame, life_expectancy_data: pandas.DataFrame):
    data = gdp_data.merge(life_expectancy_data)
    for row in data.iterrows():
        row = row[1]  # Converts row from a Tuple[Hashable, Series] to a Series
        plt.scatter(row["GDPPC"], row["Life expectancy"], marker=".")


def filter_countries_that_are_one_constant_above_another_constant(mean: float, std: float, life_expectancy_data_year: pandas.DataFrame, return_column: str, filer_column: str):
    list_of_countries = []
    for row in life_expectancy_data_year.iterrows():
        life_expectancy = row[1][filer_column]
        if life_expectancy > (mean[0] + std[0]):
            list_of_countries.append(row[1][return_column])
    return set(list_of_countries)


def filter_countries_that_are_one_constant_below_another_constant(mean: float, std: float, life_expectancy_data_year: pandas.DataFrame, return_column: str, filer_column: str):
    list_of_countries = []
    for row in life_expectancy_data_year.iterrows():
        life_expectancy = row[1][filer_column]
        if life_expectancy < (mean[0] - std[0]):
            list_of_countries.append(row[1][return_column])
    return set(list_of_countries)


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
    # Calculating the mean and standard deviation of the life expectancy data.
    #
    mean = life_expectancy_data_year.mean(axis=0, skipna=True, numeric_only=True)
    std = life_expectancy_data_year.std(axis=0, skipna=True, numeric_only=True)

    mean_gdp = gdp_data_year.mean(axis=0, skipna=True, numeric_only=True)
    std_gdp = gdp_data_year.std(axis=0, skipna=True, numeric_only=True)

    #
    # Filtering the countries that are one standard deviation above the mean.
    #
    print(filter_countries_that_are_one_constant_above_another_constant(mean, std, life_expectancy_data_year, "Entity", "Life expectancy"))
    print(filter_countries_that_are_one_constant_below_another_constant(mean_gdp, std_gdp, gdp_data_year, "Entity", "GDPPC"))
    #
    # Finishing touches and details for the plot.
    #
    plt.xlabel("GDP per capita")
    plt.ylabel("Life expectancy at birth")
    plt.show()


if __name__ == "__main__":
    main()
