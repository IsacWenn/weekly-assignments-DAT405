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


def filter_countries_that_are_one_constant_above_another_constant(mean: float, std: float, life_expectancy_data_year: pandas.DataFrame, return_column: str, filter_column: str):
    set_of_countries = set([])
    for row in life_expectancy_data_year.iterrows():
        life_expectancy = row[1][filter_column]
        if life_expectancy > (mean + std):
            set_of_countries.add(row[1][return_column])
    return set_of_countries


def filter_dataframe_above_mean(mean: float, data_frame: pandas.DataFrame, return_column: str, filter_column: str):
    set_of_countries: set = set([])
    for row in data_frame.iterrows():
        value = row[1][filter_column]
        if value > mean:
            set_of_countries.add(row[1][return_column])
    return set_of_countries


def filter_dataframe_below_mean(mean: float, data_frame: pandas.DataFrame, return_column: str, filter_column: str):
    set_of_countries: set = set([])
    for row in data_frame.iterrows():
        value = row[1][filter_column]
        if value < mean:
            set_of_countries.add(row[1][return_column])
    return set_of_countries


def main():
    #
    # Reading the data.
    #
    gdppc_data = pandas.read_csv("data/gdp-per-capita-worldbank.csv")
    gdp_data = pandas.read_csv("data/gross-domestic-product.csv")
    life_expectancy_data = pandas.read_csv("data/life-expectancy.csv")

    #
    # Filtering the data to only include data points from a specific year.
    #
    gdppc_data_year = filter_csv_data_by_year(gdppc_data, YEAR)
    gdp_data_year = filter_csv_data_by_year(gdp_data, YEAR)
    life_expectancy_data_year = filter_csv_data_by_year(life_expectancy_data, YEAR)

    #
    # Constructing a scatter plot with the filtered data.
    #
    construct_scatter_plot(gdppc_data_year, life_expectancy_data_year)

    combined_df = gdp_data_year.merge(life_expectancy_data_year, on="Entity", how="inner")\
        .merge(gdppc_data_year, on="Entity", how="inner")

    #
    # Calculating the mean and standard deviation of the life expectancy data.
    #
    mean = life_expectancy_data_year.mean(axis=0, skipna=True, numeric_only=True)
    std = life_expectancy_data_year.std(axis=0, skipna=True, numeric_only=True)

    # Alternative way to calculate mean and std with pandas methods #
    std_life = life_expectancy_data_year.describe().loc['std', 'Life expectancy']
    mean_life = life_expectancy_data_year.describe().loc['mean', 'Life expectancy']
    high_life_df = life_expectancy_data_year[life_expectancy_data_year['Life expectancy'] > mean_life + std_life][
        ['Entity', 'Life expectancy']].sort_values(by='Life expectancy', ascending=False).head(99)

    #
    #mean_gdppc = combined_df['GDPPC'].median
    #mean_gdp = combined_df['GDP'].median
    #mean_life_expectancy = combined_df['Life expectancy'].median

    mean_gdppc = combined_df.describe().loc['50%', 'GDPPC']
    mean_gdp = combined_df.describe().loc['50%', 'GDP']
    mean_life_expectancy = combined_df.describe().loc['50%', 'Life expectancy']

    #(combined_df.query(f"`Life expectancy` > {mean_life_expectancy} and `GDPPC` > {mean_gdppc} and `GDP` > @mean_gdp")
    # .loc[:, ['Entity', 'Life expectancy', 'GDP', 'GDPPC']]
    # .sort_values(by='Life expectancy', ascending=False))

    print(combined_df.query(f"`Life expectancy` > {mean_life_expectancy} and `GDPPC` > {mean_gdppc} and `GDP` > {mean_gdp}")
     .loc[:, ['Entity', 'Life expectancy', 'GDP', 'GDPPC']])

    print(combined_df.describe())

    #
    # Filtering the countries that are one standard deviation above the mean.
    #
    print(filter_countries_that_are_one_constant_above_another_constant(mean[0], std[0], life_expectancy_data_year, "Entity", "Life expectancy"))
    #
    # Finishing touches and details for the plot.
    #
    plt.xlabel("GDP per capita")
    plt.ylabel("Life expectancy at birth")
    plt.show()


if __name__ == "__main__":
    main()
