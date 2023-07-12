import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='3', help='house id')
    return parser.parse_args()


def main():
    # Load input arguments
    args = parse_opt()
    df = pd.read_csv(r'../../data/house_data_cleaned/house{}.csv'.format(args.house), parse_dates=['timestamp'],
                     index_col='timestamp')
    consumption = df['IMPORT_KW']
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the ACF
    plot_acf(consumption, lags=72, zero=False, ax=ax1)
    # Plot the PACF
    plot_pacf(consumption, lags=72, zero=False, ax=ax2)

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
