import time

from dask import dataframe as dd

start = time.time()
df = dd.read_csv('../raw_data/GridFlexHeetenDataset.csv')
df = df.loc[df['measurement'].isin(['BATTERY_KW', 'IMPORT_KW', 'EXPORT_KW', 'PV_KW'])]

df.to_csv('../raw_data/GridFlexHeetenDataset_limited.csv', index=False, single_file=True)
end_time = time.time()
print(f'Limiting raw data took: {end_time - start}')
