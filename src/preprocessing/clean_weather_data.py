import pandas as pd


def format_weather_data():
    df = pd.read_csv(r'../../data/weather_data/weather_data_raw.csv', index_col=0,
                     dtype={'YYYYMMDD': 'str', 'HH': 'str'}, skipinitialspace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df['HH'] = df['HH'].str.zfill(2)
    df['HH'].replace({'24': '00'}, inplace=True)
    df['DATE'] = df['YYYYMMDD'] + df['HH']
    df.drop(['T10N', 'STN', 'YYYYMMDD', 'HH'], axis=1, inplace=True)
    # covert to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H')
    # set date as index
    df.set_index('DATE', inplace=True)
    df.to_csv('../../data/weather_data/weather_data.csv')


if __name__ == '__main__':
    format_weather_data()
