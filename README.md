## GridFlexHeeten Dataset

The data in this dataset was collected during the GridFlex Heeten project. The data was collected between August 
2018 and August 2020 in 77 households all situated in Heeten (The Netherlands) and consists of electricity consumption 
and gas usage per minute per household. All participating households specified their data could be used in further 
research and the data of this project was collected in accordance with a privacy-by-design approach.

Raw data can be downloaded through the link [GridFlexHeeten](https://data.4tu.nl/articles/dataset/Energy_consumption_data_of_the_GridFlex_Heeten_project/14447257/1?file=27671892).
The size of the downloaded csv is ~62GB and should be stored under the folder raw_data. Please create the folder if not
exists.

Weather data for the respective period have been retrieved from weather station 278 – Heino, which is 15km from 
the houses. Source: Royal Dutch Meteorological Institute [KNMI](https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens).
After retrieval the data should be stored under the folder [weather_data](weather_data).

## Preprocessing

1. Run script [Clean weather data](preprocessing/clean_weather_data.py) to format weather data properly.
2. Run script [Remove unnecassary data](preprocessing/remove_unnecessary_data_from_raw_file.py) to keep the data we need and limit GridFlexHeeten Dataset's size.
3. Run script [Extract houses from raw data](preprocessing/extract_houses_from_raw.py) to separate raw data per house.
4. Run script [Format house data](preprocessing/format_house_data.py) to combine house's raw data to a single file. We aggregate per hour.

## Data visualization before cleaning

Data visualization is useful to identify weird patterns or missing data.
1. Run script [Plot house data](data_visualization/plot_house_data.py) with argument --path 'uncleaned' to visualise data.

## Data cleaning

1. Run script [Replace long periods](data_cleaning/replace-long-period-missing-data.py) for replacing missing periods > 1 week. Example args: --house 3 --missing_data_start_date '2019-04-03 21:00:00' --missing_data_end_date '2019-04-04 21:00:00' --replace_data_start_date '2020-04-03 21:00:00' --replace_data_end_date '2020-04-04 21:00:00'.
2. Run script [Replace missing values and outlier smoothing](data_cleaning/missing-values-replacement-and-outlier-smoothing.py) Missing values is applied for periods <= 1 week. Smoothing is applied on the whole IMPORT_KW.
3. Run script [Plot house data](data_visualization/plot_house_data.py) to visualise cleaned houses.

## Time series analysis

1. Run script [Create plots](time_series_analysis/create-plots.py) with the different arguments stated within the script.
2. Run script [Stationary check](time_series_analysis/stationary-check.py) with arguments rolling, adf-kpss.
3. Run script [Acf - Pacf](time_series_analysis/acf-pacf-plots.py) to plot Autocorrelation and Partial Autocorellation.

## Short-term load forecasting models

1. Run script [Sarimax](sarimax/sarimax_short-term-load-forecasting.py) --train True for creating and storing a Sarimax model.
2. Run script [LSTM](LSTM/LSTM-short-term-load-forecasting.py) using model_type either vanilla or encoder_decoder. Provide also the respective arguments.