import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='3', help='house id')
    parser.add_argument('--stationary_check_type', type=str, default='rolling',
                        help='possible values: rolling, adf-kpss')
    return parser.parse_args()


def perform_adf_kpss_check(df: pd.DataFrame, max_d: int) -> pd.DataFrame:
    """ Build dataframe with ADF statistics and p-value for time series after applying difference on time series

    Args:
        df (df): Dataframe of univariate time series
        max_d (int): Max value of how many times apply difference

    Return:
        Dataframe showing values of ADF statistics and p when applying ADF test after applying d times
        differencing on a time-series.

    """

    results = []

    for idx in range(max_d):
        adf_result = adfuller(df, autolag='AIC')
        kpss_result = kpss(df, regression='c', nlags="auto")
        df = df.diff().dropna()
        if adf_result[1] <= 0.05:
            adf_stationary = True
        else:
            adf_stationary = False
        if kpss_result[1] <= 0.05:
            kpss_stationary = False
        else:
            kpss_stationary = True

        stationary = adf_stationary & kpss_stationary

        results.append((idx, adf_result[1], kpss_result[1], adf_stationary, kpss_stationary, stationary))

    # Construct DataFrame
    results_df = pd.DataFrame(results, columns=['d', 'adf_stats', 'p-value', 'is_adf_stationary', 'is_kpss_stationary',
                                                'is_stationary'])

    return results_df


def plot_rolling_mean_and_std(dataframe: pd.DataFrame, window: int) -> None:
    """
    This function plots the dataframes
    rolling mean and rolling standard deviation.
    ---
    Args:
        dataframe (pd.DataFrame): Dataframe contains the timeseries
        window (int): window size
    Returns: None
    """
    df = dataframe.copy()
    df = df.resample('D').sum()
    # Get Things Rolling
    roll_mean = df.rolling(window=window).mean()
    roll_std = df.rolling(window=window).std()

    # Figure
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='w')
    ax.plot(df, label='Original')
    ax.plot(roll_mean, label='Rolling Mean')
    ax.plot(roll_std, label='Rolling STD')

    # Legend & Grid
    ax.legend(loc='upper right')
    ax.set_title('Stationary check using rolling mean and std')
    plt.grid(linestyle=":", color='grey')
    plt.show()


def main():
    # Load input arguments
    args = parse_opt()
    stationary_check = args.stationary_check_type
    df = pd.read_csv(r'../house_data/cleaned/house{}.csv'.format(args.house), parse_dates=['timestamp'],
                     index_col='timestamp')
    if stationary_check == 'rolling':
        plot_rolling_mean_and_std(df['IMPORT_KW'], window=7)
    elif stationary_check == 'adf-kpss':
        print(perform_adf_kpss_check(df['IMPORT_KW'], 3))
    else:
        print('Unknown stationary check type')


if __name__ == '__main__':
    sys.exit(main())
