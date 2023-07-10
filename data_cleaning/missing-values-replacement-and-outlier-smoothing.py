import sys
import pandas as pd
import random

random.seed(1)


def outlier_smoothing(house_df: pd.DataFrame, house: int) -> None:
    house_import_kw_median = house_df['IMPORT_KW'].median()
    coefficient = 3.0
    for index, row in house_df.iterrows():
        row_import_kw = row[0]
        month = str(index)[0:7]
        house_import_kw_month_median = house_df.loc[month, 'IMPORT_KW'].median()
        if row_import_kw == 0.0:
            house_df.at[index, 'IMPORT_KW'] = house_import_kw_month_median + random.uniform(-1.0, 1.0)

        if row_import_kw > house_import_kw_median * coefficient:
            house_df.at[index, 'IMPORT_KW'] = house_import_kw_month_median + random.uniform(-3.0, 3.0)

    house_df.to_csv('houses/house{}.csv'.format(house))


def main():
    houses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37,
              39, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 66, 69, 70, 72, 73,
              75, 77]

    for i in houses:
        df = pd.read_csv('../house_data/cleaned/house{}.csv'.format(i), parse_dates=['timestamp'],
                         index_col='timestamp')
        outlier_smoothing(df, i)


if __name__ == '__main__':
    sys.exit(main())



