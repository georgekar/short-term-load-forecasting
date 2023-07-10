import argparse
import sys
import pandas as pd
import random

random.seed(1)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--house", type=str, help='house with missing data')
    parser.add_argument("--missing_data_start_date", type=str, help='missing data start date')
    parser.add_argument("--missing_data_end_date", type=str, help='missing data end date')
    parser.add_argument("--replace_data_start_date", type=str, help='replace data start date')
    parser.add_argument("--replace_data_end_date", type=str, help='replace data end date')
    return parser.parse_args()


def main():
    # Load input arguments
    args = parse_opt()
    house = args.house
    missing_data_start_date = args.missing_data_start_date
    missing_data_end_date = args.missing_data_end_date
    replace_data_start_date = args.replace_data_start_date
    replace_data_end_date = args.replace_data_end_date
    # load dataset
    df = pd.read_csv(r'../house_data/uncleaned/house{}.csv'.format(house), parse_dates=['timestamp'],
                     index_col='timestamp')

    replace_data_df = df.loc[replace_data_start_date:replace_data_end_date]
    coefficients = []
    for i in range(replace_data_df.shape[0]):
        coefficients.append(random.uniform(-1.0, 1.0))
    coefficients = abs(replace_data_df['IMPORT_KW'].values + coefficients)
    df.loc[missing_data_start_date:missing_data_end_date, 'IMPORT_KW'] = coefficients
    df.to_csv('../house_data/cleaned/house{}.csv'.format(house))


if __name__ == '__main__':
    sys.exit(main())
