import argparse
import os
import pandas as pd
import sys
import time
import warnings
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='3', help='house id')
    parser.add_argument('--train', default=False, action='store_true',
                        help='if set model will be trained from scratch. Takes approximately 30 mins')
    parser.add_argument("--p", type=int, default=1, help='Trend element: Autoregressive order')
    parser.add_argument("--d", type=int, default=1, help='Trend element: Difference order')
    parser.add_argument("--q", type=int, default=1, help='Trend element: Moving average order')
    parser.add_argument("--P", type=int, default=2, help='Seasonal element: Autoregressive order')
    parser.add_argument("--D", type=int, default=1, help='Seasonal element: Difference order. D=1 would calculate a '
                                                         'first order seasonal difference')
    parser.add_argument("--Q", type=int, default=1, help='Seasonal element: Moving average order. Q=1 would use a '
                                                         'first order errors in the model')
    parser.add_argument("--s", type=int, default=24, help='Seasonal element: Single seasonal period')
    parser.add_argument("--model_name", type=str, default='sarimax_house3.pkl', help='name of the model to be stored')
    return parser.parse_args()


def fit_sarimax(exogenous: pd.DataFrame, y: pd.DataFrame, args: argparse.Namespace, model_path: str) -> None:
    # Supress UserWarnings
    warnings.simplefilter('ignore', category=UserWarning)
    # Set Hyper-Parameters
    p, d, q = args.p, args.d, args.q
    P, D, Q = args.P, args.D, args.Q
    s = args.s

    # Fit SARIMAX
    sarimax_model = SARIMAX(y,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s),
                            exog=exogenous)

    sarimax_model_fit = sarimax_model.fit(disp=0)
    # save the model to disk
    sarimax_model_fit.save(model_path)

    # Summary
    print(sarimax_model_fit.summary())
    # Plot diagnostics
    sarimax_model_fit.plot_diagnostics(figsize=(16, 9))
    plt.show()


def plot_pred_vs_true_augsut2020(y_true, y_pred, rmse):
    plt.figure(figsize=(20, 12))

    plt.plot(y_true, color='green', label='true')
    plt.plot(y_pred, color='red', linestyle='dashed', label='pred')
    plt.plot(label='RMSE')
    plt.title("Actual vs Predicted (August-{}), RMSE: {:.5} kW".format(2020, rmse))
    plt.ylabel("Import [kW]")
    plt.xlabel("hours [#]")

    plt.legend()
    plt.grid()
    plt.show()


def plot_rmses(rmses):
    plt.figure(figsize=(10, 5))
    xticks = []
    for i in range(2, 31):
        xticks.append(i)
    df = pd.DataFrame(data=rmses,
                      index=xticks,
                      columns=['RMSE'])

    plt.plot(df, color='red', label='RMSE')
    plt.plot(label='RMSE')
    plt.title('August 2020 - RMSE per day')
    plt.ylabel("RMSE [kW]")
    plt.xlabel("day [#]")

    plt.legend()
    plt.grid()
    plt.show()


def rmse_per_day(y_true, y_pred):
    import math
    print(y_true.shape)
    y_true = y_true.reshape(int(y_true.shape[0] / 24), 24)
    y_pred = y_pred.reshape(int(y_pred.shape[0] / 24), 24)
    y_true = y_true[1:]
    y_pred = y_pred[1:]
    rmses = []
    for i in range(len(y_true)):
        mse = np.square(np.subtract(y_true[i], y_pred[i])).mean()
        rmse = math.sqrt(mse)
        rmses.append(rmse)
    return rmses


def main():
    # Load input arguments
    args = parse_opt()
    # Load dataset
    df = pd.read_csv('../../data/house_data_cleaned/house{}.csv'.format(args.house), parse_dates=['timestamp'],
                     index_col='timestamp')
    exogenous_columns = ['HOUR_OF_DAY', 'T', 'TD', 'SQ', 'Q', 'DR', 'U']
    target_column = ['IMPORT_KW']
    # Split on train and test. Last month should be kept for test
    train_df = df.loc['2018-08-01 00:00:00':'2020-07-31 23:00:00']
    x_train = train_df[exogenous_columns]
    y_train = train_df[target_column].values.reshape(-1, 1)
    test_df = df.loc['2020-08-01 00:00:00':'2020-08-31 21:00:00']
    x_test = test_df[exogenous_columns]
    y_test = test_df[target_column]
    # Scale data
    # Used in exogenous
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    # Used for target value IMPORT_KW
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = x_scaler.fit_transform(x_train)
    y_train = y_scaler.fit_transform(y_train)
    x_test = x_scaler.transform(x_test)
    x_train = pd.DataFrame(x_train, index=train_df.index, columns=exogenous_columns)
    y_train = pd.DataFrame(y_train, index=train_df.index, columns=[target_column])
    x_test = pd.DataFrame(x_test, index=test_df.index, columns=exogenous_columns)
    model_path = 'model/' + args.model_name
    if args.train:
        if os.path.exists(model_path):
            print("Erasing existing model: {}".format(model_path))
            os.remove(model_path)
        start = time.time()
        fit_sarimax(x_train, y_train, args, model_path)
        end_time = time.time()
        print('Took {} to train the SARIMAX model.'.format(end_time - start))
    else:
        if os.path.exists(model_path):
            print("Loading existing model {}:".format(model_path))
            model = SARIMAXResults.load('models/sarimax_house{}.pkl'.format(args.house))

            pred_start_date = y_test.index[0]
            pred_end_date = y_test.index[-1]
            predictions = model.predict(start=pred_start_date, end=pred_end_date, exog=x_test)
            predictions = y_scaler.inverse_transform(predictions.values.reshape(-1, 1))
            y_true = y_test.values
            rmse_august = np.mean(np.sqrt(metrics.mean_squared_error(y_true[:-22], predictions[:-22])))
            rmses = rmse_per_day(y_true[:-22], predictions[:-22])
            plot_pred_vs_true_augsut2020(y_true[:-22], predictions[:-22], rmse_august)
            plot_rmses(rmses)

        else:
            print("Model {} doesn't exist!".format(model_path))
            sys.exit(0)


if __name__ == '__main__':
    sys.exit(main())
