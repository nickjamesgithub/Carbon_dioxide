import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import datetime
from scipy.fftpack import fft, fftshift
from Utils import dendrogram_plot_labels
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pyts.metrics import dtw
from Utilities import haversine

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# HR station details
hr_station_details = pd.read_csv("/Users/tassjames/Desktop/hydrology/hrs_station_details.csv",
                                 index_col=None, header=0, skiprows=11)

# Overlap grid
data_per_segment = 3750
overlap_grid = 0.4

# Loop over data segment and overlap grid
time_series_list = []
omitted_codes = []
labels = []
# station, Lat and long
station_id_list = []
latitude_list = []
longitude_list = []

# Loop over filenames
for filename in all_files:

    # Get filename
    file_id = filename.rsplit('/', 1)[-1]

    # Station, lat/long
    station_id = file_id[:-4]
    station_id_list.append(station_id)
    # Access latitude and longitude values
    latitude = hr_station_details.loc[hr_station_details['AWRC Station Number'] == station_id, 'Latitude'].iloc[0]
    longitude = hr_station_details.loc[hr_station_details['AWRC Station Number'] == station_id, 'Longitude'].iloc[0]
    # Append latitude and longitude to lists
    latitude_list.append(latitude)
    longitude_list.append(longitude)

    # Get dataframe
    df = pd.read_csv(filename, index_col=None, header=0, skiprows=26)
    df_slice = df[["Date", "Flow (ML)", "Bureau QCode"]]
    df["Date"] = pd.to_datetime(df_slice["Date"])
    flow = df_slice[["Date", "Flow (ML)"]]  # Date and Flow
    flow_ts = np.array(df_slice["Flow (ML)"])  # Make flow an array
    # Minimum and Maximum dates
    min_date = np.min(df_slice["Date"])
    max_date = np.max(df_slice["Date"])

    # Datetime, datetime
    min_ = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    max_ = datetime.datetime.strptime(max_date, '%Y-%m-%d')
    # If date outside certain range omit:
    lb = datetime.datetime(1980, 1, 1)
    ub = datetime.datetime(2019, 1, 1)

    # Lower bound and upper bound
    if min_ > lb or max_ < ub:
        # Code of omitted file
        omitted_codes.append(file_id)
        print("Omit ", file_id)
    else:
        labels.append(file_id)
        mask = (df_slice['Date'] >= '1980-01-01') & (df_slice['Date'] <= '2019-01-01')
        df_date_slice = df.loc[mask]
        flow_ts = df_date_slice[["Date", "Flow (ML)"]]  # Date and Flow in new date range
        flow_ts = np.array(flow_ts["Flow (ML)"])

        # Append to time series
        time_series_list.append(flow_ts)

        # Iteration
        print("Run over files", filename)

# Compute distance between all time series and core governing process
core_process_time = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/time_series_core.csv")
core_process_time = core_process_time.iloc[1:,1]

core_process_spectrum = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/time_series_core.csv")
core_process_spectrum = core_process_spectrum.iloc[1:,1]

# Compute latitude and longitude median
latitude_median = np.median(latitude_list)
longitude_median = np.median(longitude_list)

# Compute distance between core time series and all individual time series
dtw_vector_time = []
geographic_distance = []

# Time series i and j
for i in range(len(time_series_list)):
    # Compute distance in time
    time_distance = dtw(time_series_list[i][365:], core_process_time, method='itakura')
    dtw_vector_time.append(time_distance)
    # Compute distance in space
    geo_distance = haversine(longitude_median, latitude_median, longitude_list[i], latitude_list[i])
    geographic_distance.append(geo_distance)
    # Print iteration, time and geographic distance
    print("Iteration", i)
    print("Time distance", time_distance)
    print("Geographic distance", geo_distance)
