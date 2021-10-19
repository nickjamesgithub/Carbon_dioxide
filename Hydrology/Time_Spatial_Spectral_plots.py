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

# Power spectral density
spectral_list = []

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

        # Truncate flow_ts by one year to match with core process
        flow_truncate = flow_ts[365:]
        # Append to time series
        time_series_list.append(flow_truncate)

        # Compute Power spectral density
        flow_ts_normalised = flow_ts - np.mean(flow_ts)
        f, Pxx_density = welch(flow_ts_normalised, fs=1, window='hann', nperseg=data_per_segment,
                               noverlap = overlap_grid * data_per_segment, scaling='density')
        log_spectrum = np.log(Pxx_density)
        print("Spectrum length", len(log_spectrum))
        spectral_list.append(log_spectrum[:-1])

# Plot distance between all time series
norm_l1_distance_matrix = np.zeros((len(time_series_list), len(time_series_list)))
spatial_distance_matrix = np.zeros((len(time_series_list), len(time_series_list)))
spectral_distance_matrix = np.zeros((len(time_series_list), len(time_series_list)))
for i in range(len(time_series_list)):
    for j in range(len(time_series_list)):

        # L1 distance matrix norm
        norm_ts_i = time_series_list[i] / np.sum(time_series_list[i])
        norm_ts_j = time_series_list[j] / np.sum(time_series_list[j])
        norm_l1_distance_matrix[i, j] = np.sum(np.abs(norm_ts_i-norm_ts_j))

        # Spatial distance matrix
        spatial_dist = haversine(longitude_list[i], latitude_list[i],
                                 longitude_list[j], latitude_list[j])
        spatial_distance_matrix[i,j] = spatial_dist

        # Spectral distance matrix
        norm_spectral_i = spectral_list[i]/np.sum(np.abs(spectral_list[i]))
        norm_spectral_j = spectral_list[j]/np.sum(np.abs(spectral_list[j]))
        spectral_distance_matrix[i,j] = np.sum(np.abs(norm_spectral_i - norm_spectral_j))

    print("Iteration", i)

# Plot l2 distance matrix
plt.matshow(norm_l1_distance_matrix)
plt.savefig("Distance_matrix_time")
plt.show()

# Plot spatial distance matrix
plt.matshow(spatial_distance_matrix)
plt.savefig("Distance_matrix_space")
plt.show()

# Plot spectral distance matrix
plt.matshow(spectral_distance_matrix)
plt.savefig("Distance_matrix_spectral")
plt.show()

# Dendrogram
dendrogram_plot_labels(norm_l1_distance_matrix, "_L1_", "_Time_", labels=labels)
dendrogram_plot_labels(spatial_distance_matrix, "_Haversine_", "_Spatial_", labels=labels)
dendrogram_plot_labels(spectral_distance_matrix, "_L1_", "_Spectral_", labels=labels)
