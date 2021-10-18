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

        # Iteration
        print("Run over files", filename)

# Compute distance between all time series and core governing process
core_process_time = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/time_series_core.csv")
core_process_time = core_process_time.iloc[1:,1]

core_process_spectrum = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/log_PSD_core.csv")
core_process_spectrum = core_process_spectrum.iloc[1:,1]

# Compute latitude and longitude median
latitude_median = np.median(latitude_list)
longitude_median = np.median(longitude_list)

# Compute distance between core time series and all individual time series
dtw_vector_time = [] # DTW distance between two time series
geographic_distance = [] # Haversine geographic distance
l1_distance_spectral = [] # L1 distance between 2 spectra

# Consistency between spatial distance and time distance
for i in range(len(time_series_list)):
    # Compute distance in time
    time_distance = dtw(time_series_list[i], core_process_time, method='itakura')
    dtw_vector_time.append(time_distance)

    # Compute distance in space
    geo_distance = haversine(longitude_median, latitude_median, longitude_list[i], latitude_list[i])
    geographic_distance.append(geo_distance)

    # Normalise both log spectra to ameliorate amplitude effects
    core_spectrum_norm = core_process_spectrum / np.sum(np.abs(core_process_spectrum))
    spectral_indiv_norm = spectral_list[i] / np.sum(np.abs(spectral_list[i]))

    # Compute distance in frequency space
    l1_spectrum = np.sum(np.abs(core_spectrum_norm - spectral_indiv_norm))
    l1_distance_spectral.append(l1_spectrum)

    # Print iteration, time and geographic distance
    print("Iteration", i)
    print("Time distance", time_distance)
    print("Geographic distance", geo_distance)
    print("Spectral L^1 distance", l1_spectrum)

# Distribution of distances
plt.hist(dtw_vector_time, bins=60)
plt.title("DTW distance distribution")
plt.show()

plt.hist(geographic_distance, bins=60)
plt.title("Geodesic distance distribution")
plt.show()

plt.hist(l1_distance_spectral, bins=60)
plt.title("Spectral distance distribution")
plt.show()

# Convert to affinity vectors
affinity_time = 1 - dtw_vector_time/np.max(dtw_vector_time)
affinity_psd = 1 - l1_distance_spectral/np.max(l1_distance_spectral)
affinity_spatial = 1 - geographic_distance/np.max(geographic_distance)

# Combine vectors
affinity_vectors = np.concatenate((affinity_time, affinity_psd, affinity_spatial))
affinity_matrix = np.reshape(affinity_vectors, (len(affinity_vectors)//3,3))

# Plot affinity matrix (Nx3)
fig, ax = plt.subplots()
plt.matshow(affinity_matrix)
fig.colorbar()
plt.show()

# Compute distance between vectors in affinity
time_spatial_inc = np.sum(np.abs(affinity_time - affinity_spatial))
frequency_spatial_inc = np.sum(np.abs(affinity_psd - affinity_spatial))

# Print
print("Time/spatial inconsistency", time_spatial_inc)
print("Spectral/spatial inconsistency", frequency_spatial_inc)