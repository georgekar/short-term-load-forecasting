import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from statsmodels.tsa.seasonal import seasonal_decompose


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='3', help='house id')
    parser.add_argument('--plot_type', type=str, default='normal',
                        help='possible values: normal, histogram, box-plot, pearson, violin, decompose')
    parser.add_argument('--interval', type=str, default='month', help='show the seasonality on specific interval')
    parser.add_argument('--show_plot', default=True, action='store_true', help='flag for showing the plot or not')
    parser.add_argument('--save_plot', default=True, action='store_true', help='flag of storing the plot or not')
    return parser.parse_args()


def box_plot_seasonality(df: pd.DataFrame, args: argparse.Namespace) -> None:
    interval = args.interval
    if interval == 'month':
        df['month'] = df.index.month
    elif interval == 'week':
        df['week'] = df.index.day_of_week + 1
    else:
        df['hour'] = df.index.hour + 1

    fig, ax = plt.subplots(figsize=(7, 3.5))
    df.boxplot(column='IMPORT_KW', by=interval, ax=ax, )
    df.groupby(interval)['IMPORT_KW'].median().plot(style='o-', linewidth=0.8, ax=ax)
    ax.set_ylabel('Consumption')
    ax.set_title('Consumption distribution by {}'.format(interval))
    fig.suptitle('')
    if args.show_plot:
        plt.show()
    if args.save_plot:
        fig.savefig(r'plots/time_series_analysis/consumption_distribution_by_{}.png'.format(interval))
        plt.close()


def consumption_on_holidays(df: pd.DataFrame, args: argparse.Namespace) -> None:
    df = df.assign(HOLIDAY=df.HOLIDAY.astype(str))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.5))
    sns.violinplot(
        x='IMPORT_KW',
        y='HOLIDAY',
        data=df,
        palette='tab10',
        ax=ax
    )
    ax.set_title('Distribution of consumption between holidays and non-holidays')
    ax.set_xlabel('Demand')
    ax.set_ylabel('Holiday')
    if args.show_plot:
        plt.show()
    if args.save_plot:
        fig.savefig(r'plots/time_series_analysis/consumption_on_holidays.png')
        plt.close()


def correlation_check(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Plots a Pearson Correlation Heatmap.
    ---
    Args:
        df (pd.DataFrame): dataframe to plot
        args (argparse.Namespace): resolving input arguments

    Returns: None
    """
    # Figure
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='w')
    correlations_df = df.corr(method='pearson', min_periods=1)
    sns.heatmap(correlations_df, cmap="Blues", annot=True, linewidth=.1)

    # Labels
    ax.set_title("Pearson Correlation Heatmap", fontsize=15, pad=10)
    ax.set_facecolor(color='white')
    if args.show_plot:
        plt.show()
    if args.save_plot:
        fig.savefig(r'plots/time_series_analysis/pearson_correlation_heatmap.png')
        plt.close()


def decompose_series(df: pd.DataFrame) -> None:
    """
    This function decomposes the time series
    into trend, seasonality and residuals.
    ---
    Args:
        df (pd.DataFrame): Dataframe that contains the timeseries data

    Returns: None
    """
    df = df.resample('D').sum()
    # Decomposition
    decomposition = seasonal_decompose(df['IMPORT_KW'])
    decomposition.plot()

    plt.tight_layout()
    plt.show()


def histogram(df: pd.DataFrame):
    df.hist(bins=50, column='IMPORT_KW')
    plt.show()


def plot_house_data(df: pd.DataFrame, args: argparse.Namespace) -> None:
    house_df = df.resample('D').sum()
    house_df.reset_index(inplace=True)
    house_df.plot.line(x='timestamp', y=['IMPORT_KW'], title='Consumption')
    # Add legend
    plt.legend()
    # Auto space
    plt.tight_layout()
    # Display plot
    if args.show_plot:
        plt.show()
    # if args.save_plot:
    #     plt.savefig(r'plots/aggregated_train.png')
    #     plt.close()


def main():
    # Load input arguments
    args = parse_opt()
    df = pd.read_csv(r'../house_data/cleaned/house{}.csv'.format(args.house), parse_dates=['timestamp'],
                     index_col='timestamp')
    df.drop(columns=['EXPORT_KW', 'PV_KW', 'BATTERY_KW', 'IX'], inplace=True)
    plot_type = args.plot_type
    if plot_type == 'normal':
        plot_house_data(df, args)
    elif plot_type == 'histogram':
        histogram(df)
    elif plot_type == 'box-plot':
        box_plot_seasonality(df, args)
    elif plot_type == 'pearson':
        correlation_check(df, args)
    elif plot_type == 'violin':
        consumption_on_holidays(df, args)
    elif plot_type == 'decompose':
        decompose_series(df)
    else:
        print('Unknown plot type')


if __name__ == '__main__':
    sys.exit(main())

