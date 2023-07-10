import time

from dask import dataframe as dd


def format_house_data(house_number: int):
    start = time.time()
    df = dd.read_csv('../raw_data/GridFlexHeetenDataset_limited.csv')
    df = df.loc[(df['house'] == 'House{}'.format(house_number))
                & (df['measurement'].isin(['BATTERY_KW', 'IMPORT_KW', 'EXPORT_KW', 'PV_KW']))]

    import_df = df.loc[df['measurement'] == 'IMPORT_KW']
    export_df = df.loc[df['measurement'] == 'EXPORT_KW']
    pv_df = df.loc[df['measurement'] == 'PV_KW']
    battery_df = df.loc[df['measurement'] == 'BATTERY_KW']

    import_df.to_csv('../house_raw_data/house{}_import_kw.csv'.format(house_number), index=False, single_file=True)
    export_df.to_csv('../house_raw_data/house{}_export_kw.csv'.format(house_number), index=False, single_file=True)
    pv_df.to_csv('../house_raw_data/house{}_pv_kw.csv'.format(house_number), index=False, single_file=True)
    battery_df.to_csv('../house_raw_data/house{}_battery_kw.csv'.format(house_number), index=False, single_file=True)
    end_time = time.time()
    print(f'Extracting data took: {end_time - start}')


if __name__ == '__main__':
    for i in range(1, 78):
        format_house_data(i)
