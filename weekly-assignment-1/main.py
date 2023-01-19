import pandas
import numpy
import matplotlib.pyplot as plt

def filter_csv_data_by_year(pandas_csv_data: pandas.DataFrame, year: int):
    result_data = pandas.DataFrame()
    for row in pandas_csv_data.iterrows():
        row = row[1]
        type(row)
        if row.get('Year') == year:
            result_data.add(row)
    print(result_data)
    return result_data

def main():
    gdp_data = pandas.read_csv("data/gdp-per-capita-worldbank.csv")
    life_expectancy_data = pandas.read_csv("data/life-expectancy.csv")

    gdp_data_2020 = filter_csv_data_by_year(gdp_data, 2020)
    print(gdp_data_2020)

    plt.scatter(10, 10)
    plt.show()


if __name__ == "__main__":
    main()
