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

    # Alternative way to calculate mean and std with pandas methods #
    std_life = life_expectancy_data_year.describe().loc['std', 'Life expectancy']
    mean_life = life_expectancy_data_year.describe().loc['mean', 'Life expectancy']
    print(life_expectancy_data_year[life_expectancy_data_year['Life expectancy'] > mean_life + std_life][
              ['Entity', 'Life expectancy']].sort_values(by='Life expectancy', ascending=False).head(99))

    # Combining the data into a single DataFrame.
    combined_df = gdp_data_year.merge(life_expectancy_data_year, on="Entity", how="inner") \
        .merge(gdppc_data_year, on="Entity", how="inner")

    # Calculating statistical values of the combined DataFrame.
    low_gdp = combined_df.describe().loc['25%', 'GDP']
    high_life_expectancy = combined_df.describe().loc['75%', 'Life expectancy']

    print(combined_df.query(f"`Life expectancy` > {high_life_expectancy} and `GDP` < {low_gdp}")
          .loc[:, ['Entity', 'Life expectancy', 'GDP', 'GDPPC']].sort_values(by='Life expectancy', ascending=False))

    high_gdp = combined_df.describe().loc['75%', 'GDP']
    low_life_expectancy = combined_df.describe().loc['25%', 'Life expectancy']
    print(combined_df.query(f"`Life expectancy` < {low_life_expectancy} and `GDP` > {high_gdp}")
          .loc[:, ['Entity', 'Life expectancy', 'GDP', 'GDPPC']].sort_values(by='Life expectancy', ascending=False))

    high_gdppc = combined_df.describe().loc['75%', 'GDPPC']
    low_life_expectancy = combined_df.describe().loc['25%', 'Life expectancy']
    print(combined_df.query(f"`Life expectancy` < {low_life_expectancy} and `GDPPC` > {high_gdppc}")
          .loc[:, ['Entity', 'Life expectancy', 'GDP', 'GDPPC']].sort_values(by='Life expectancy', ascending=False))
    
    #
    # Finishing touches and details for the plot.
    #
    plt.xlabel("GDP per capita")
    plt.ylabel("Life expectancy at birth")
    plt.show()


if __name__ == "__main__":
    main()
