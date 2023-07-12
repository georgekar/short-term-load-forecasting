import pandas as pd
import holidays
import time


def is_wkd_or_holiday(timestamp):
    return holidays.country_holidays('NL').get(timestamp, 0) != 0 or timestamp.weekday() > 4


def format_house_data(house_number: int):
    start = time.time()
    # Read import kw csv
    import_df = pd.read_csv(r'../../data/house_raw_data/house{}_import_kw.csv'.format(house_number),
                            usecols=['timestamp', 'value'])
    import_df.rename(columns={'value': 'IMPORT_KW'}, inplace=True)
    # Define format and convert column to date time, first we need to convert timestamp to utc
    import_df['timestamp'] = pd.to_datetime(import_df['timestamp'], utc=True)
    import_df['timestamp'] = import_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    import_df['timestamp'] = pd.to_datetime(import_df['timestamp'])
    # Aggregate per hour
    resample_import_df = import_df.resample('H', on='timestamp').agg({'IMPORT_KW': 'sum'})

    # Read export kw csv
    export_df = pd.read_csv(r'../../data/house_raw_data/house{}_export_kw.csv'.format(house_number),
                            usecols=['timestamp', 'value'])
    export_df.rename(columns={'value': 'EXPORT_KW'}, inplace=True)
    # Define format and convert column to date time, first we need to convert timestamp to utc
    export_df['timestamp'] = pd.to_datetime(export_df['timestamp'], utc=True)
    export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    export_df['timestamp'] = pd.to_datetime(export_df['timestamp'])
    # Aggregate per hour
    resample_export_df = export_df.resample('H', on='timestamp').agg({'EXPORT_KW': 'sum'})

    # Read pv kw csv
    pv_df = pd.read_csv(r'../../data/house_raw_data/house{}_pv_kw.csv'.format(house_number),
                        usecols=['timestamp', 'value'])
    pv_df.rename(columns={'value': 'PV_KW'}, inplace=True)
    # Define format and convert column to date time, first we need to convert timestamp to utc
    pv_df['timestamp'] = pd.to_datetime(pv_df['timestamp'], utc=True)
    pv_df['timestamp'] = pv_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    pv_df['timestamp'] = pd.to_datetime(pv_df['timestamp'])
    # Aggregate per hour
    resample_pv_df = pv_df.resample('H', on='timestamp').agg({'PV_KW': 'sum'})

    # Read battery kw csv
    battery_df = pd.read_csv(r'../../data/house_raw_data/house{}_battery_kw.csv'.format(house_number),
                             parse_dates=['timestamp'], usecols=['timestamp', 'value'])
    battery_df.rename(columns={'value': 'BATTERY_KW'}, inplace=True)
    # Define format and convert column to date time, first we need to convert timestamp to utc
    battery_df['timestamp'] = pd.to_datetime(battery_df['timestamp'], utc=True)
    battery_df['timestamp'] = battery_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    battery_df['timestamp'] = pd.to_datetime(battery_df['timestamp'])
    # Aggregate per hour
    resample_battery_df = battery_df.resample('H', on='timestamp').agg({'BATTERY_KW': 'sum'})

    joined_df = resample_import_df.join(resample_export_df)
    joined_df = joined_df.join(resample_pv_df)
    joined_df = joined_df.join(resample_battery_df)

    joined_df['YEAR'] = joined_df.index.year
    joined_df['MONTH'] = joined_df.index.month
    joined_df['DAY'] = joined_df.index.day
    joined_df['HOUR_OF_DAY'] = joined_df.index.hour

    weather_df = pd.read_csv(r'../../data/weather_data/weather_data.csv', parse_dates=['DATE'], index_col='DATE')
    joined_df = joined_df.join(weather_df)

    joined_df.reset_index(inplace=True)
    joined_df['HOLIDAY'] = joined_df['timestamp'].apply(lambda x: is_wkd_or_holiday(x)).values
    joined_df['HOLIDAY'] = joined_df['HOLIDAY'].astype(int)

    joined_df.fillna(0.0, inplace=True)
    joined_df.to_csv(r'../../data/house_data_uncleaned/house{}.csv'.format(house_number), index=False)
    end_time = time.time()
    print("Finished formatting house {}, took: {}".format(house_number, end_time - start))


if __name__ == '__main__':
    pd.set_option('display.max_columns', 9)
    houses = [2, 3, 69]
    for i in houses:
        format_house_data(i)
