import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='3', help='house id')
    parser.add_argument("--year", type=str, default='2020', help='year to run validation and plot result')
    parser.add_argument("--horizon", type=int, default=24, help='number of hours to predict in the future')
    parser.add_argument("--window", type=int, default=24, help='window of past hours to consider for the prediction')
    parser.add_argument("--model_type", type=str, default='vanilla', help='either vanilla or encoder_decoder')
    parser.add_argument("--model_name", type=str, default='vanilla_lstm_2.h5', help='name of the model to be stored')
    parser.add_argument("--batch_size", type=int, default=32, help='specify size of batches to split the training data')
    parser.add_argument("--epochs", type=int, default=500, help='specify number of epochs')
    parser.add_argument('--train', default=False, action='store_true', help='if set model will be trained from scratch')
    return parser.parse_args()


# convert series to supervised learning. Code picked
# from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
# Kudos to Jason Brownlee.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def plot_pred_vs_true_augsut2020(path, test_house, y_true, y_pred, rmse):
    plt.figure(figsize=(20, 12))

    plt.plot(y_true[0], color='green', label='true')
    plt.plot(y_pred[0], color='red', linestyle='dashed', label='pred')
    plt.plot(label='RMSE')
    plt.title("Actual vs Predicted (August-{}), RMSE: {:.5} kW".format(2020, rmse))
    plt.ylabel("Import [kW]")
    plt.xlabel("hours [#]")

    plt.legend()
    plt.grid()

    # save plot to file
    if not os.path.exists(path):
        print("Results will be stored under: " + path)
        os.makedirs(path)
    plt.savefig('august_2020_house_' + str(test_house) + '.png')
    plt.close()


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


def get_prediction_for_specific_month(model, df_test, x_scaler, y_scaler, args, month):
    y_pred = []
    for i in range(int(args.window / 24) + 1, 31):
        x_test = df_test[args.year + "-" + month + "-" + '%02d' % (
                    i - (int(args.window / 24))):args.year + "-" + month + "-" + '%02d' % (i - 1)].values
        x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
        x_test = x_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
        tmp = model.predict(x_test, verbose=0).reshape(24, 1)
        tmp = y_scaler.inverse_transform(tmp)
        y_pred.append(tmp)
    y_pred = np.array(y_pred).flatten()
    for i in range(args.window):
        y_pred = np.insert(y_pred, 0, 0)
    return y_pred


def build_model(x_train, args):
    units = 256
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.5, seed=22)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units, input_shape=(x_train.shape[1], x_train.shape[2]),
                                   kernel_initializer=initializer, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=args.horizon))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_model_encoder_decoder(x_train, y_train):
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(200, input_shape=(n_timesteps, n_features)))
    model.add(tf.keras.layers.RepeatVector(n_outputs))
    model.add(tf.keras.layers.LSTM(200, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    return model


# plot diagnostic learning curves
def plot_learning_curves(path, model_name, history):
    # plot loss
    # plt.subplot(211)
    plt.title(model_name)
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='val')
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epochs [#]")
    plt.legend()
    plt.grid()
    # save plot to file
    if not os.path.exists(path):
        print("Results will be stored under: " + path)
        os.makedirs(path)
    plt.savefig(path + model_name + '_diagnostics.png')
    plt.close()


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

    model_path = 'models/' + args.model_name

    train_houses = 2
    test_houses = train_houses

    # load dataset

    dataset = read_csv('../../data/house_data_cleaned/house{}.csv'.format(train_houses), parse_dates=['timestamp'],
                       index_col='timestamp')

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_data = dataset[['IMPORT_KW', 'HOUR_OF_DAY', 'T', 'TD', 'SQ',
                      'Q', 'DR', 'U']].values

    # frame as supervised learning
    reframed = series_to_supervised(X_data, args.window, args.horizon)
    # drop columns we don't want to predict
    reframed.drop(reframed.loc[:, 'var2(t)':'var8(t)'].columns, axis=1, inplace=True)
    for col in range(1, 24):
        reframed.drop(reframed.loc[:, 'var2(t+{})'.format(col):'var8(t+{})'.format(col)].columns, axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values

    # Last month should be kept for test
    n_train_hours = dataset.shape[0] - 31 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    print(train.shape)
    print(test.shape)

    # split into input and outputs
    train_X, train_y = train[:, :-args.horizon], train[:, train.shape[1] - args.horizon:train.shape[1]]
    test_X, test_y = test[:, :-args.horizon], test[:, test.shape[1] - args.horizon:test.shape[1]]

    # reshape input to be 3D [samples, timesteps, features]
    n_features = X_data.shape[1]
    train_X = train_X.reshape((train_X.shape[0], args.window, n_features))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_X = test_X.reshape((test_X.shape[0], args.window, n_features))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

    x_train = x_scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    y_train = y_scaler.fit_transform(train_y.reshape(-1, train_y.shape[-1])).reshape(train_y.shape)
    x_test = x_scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    y_test = y_scaler.transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    if args.train:
        if os.path.exists(model_path):
            print("Erasing existing model: {}".format(model_path))
            os.remove(model_path)

        if args.model_type == 'vanilla':
            # generate model
            model = build_model(x_train, args)
        else:
            model = build_model_encoder_decoder(x_train, y_train)

        tf.keras.utils.plot_model(model=model, show_shapes=True)

        # print summary
        print(model.summary())

        # This will be used to avoid overfitting
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10,
                                              start_from_epoch=int(args.epochs / 2.0))
        history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=2, shuffle=False,
                            callbacks=[es])

        # save model
        model.save(model_path)
    else:
        if os.path.exists(model_path):
            print("Loading existing model {}:".format(model_path))
            model = tf.keras.models.load_model(model_path)
        else:
            print("Model {} doesn't exist!".format(model_path))
            sys.exit(0)

    ################################################
    ################[ EVALUATION ]##################
    ################################################

    year = args.year

    RMSE_all_unseen_houses_august = []

    df_test = pd.read_csv('../../data/house_data_cleaned/house{}.csv'.format(test_houses), parse_dates=['timestamp'],
                          index_col='timestamp')

    df_test = df_test[['IMPORT_KW', 'HOUR_OF_DAY', 'T', 'TD', 'SQ', 'Q', 'DR', 'U']]

    y_august_true = df_test[year + "-08-01":year + "-08-30"][['IMPORT_KW']].values

    y_true = []

    y_true.append(y_august_true)
    y_pred_august = get_prediction_for_specific_month(model, df_test, x_scaler, y_scaler, args, '08')
    y_pred_august = y_pred_august.reshape(y_pred_august.shape[0], 1)
    rmses = rmse_per_day(y_august_true, y_pred_august)
    plot_rmses(rmses)
    y_pred = []
    y_pred.append(y_pred_august)
    rmse_august = np.mean(np.sqrt(metrics.mean_squared_error(y_august_true, y_pred_august)))
    RMSE_all_unseen_houses_august.append(rmse_august)
    plot_pred_vs_true_augsut2020('plots/', test_houses, y_true, y_pred, rmse_august)

    print("RMSE-all-unseen-houses-august: {:.5}".format(
        sum(RMSE_all_unseen_houses_august) / len(RMSE_all_unseen_houses_august)))


if __name__ == '__main__':
    sys.exit(main())
