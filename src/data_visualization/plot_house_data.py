import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='cleaned', help='path to read and save plots')
    return parser.parse_args()


def plot_house_data(house: int, path: str):
    df = pd.read_csv(r'../../data/house_data_{}/house{}.csv'.format(path, house), parse_dates=['timestamp'],
                     index_col='timestamp')
    house_df = df.resample('D').sum()
    house_df.reset_index(inplace=True)
    house_df.plot.line(x='timestamp', y=['IMPORT_KW'], title='Consumption')
    # Add legend
    plt.legend()
    # Auto space
    plt.tight_layout()
    # Save plot
    plt.savefig(r'../plots/house_plots/{}/house{}.png'.format(path, house))
    plt.close()


def main():
    # Load input arguments
    args = parse_opt()
    houses = [2, 3, 69]
    for i in houses:
        plot_house_data(i, args.path)


if __name__ == '__main__':
    sys.exit(main())
