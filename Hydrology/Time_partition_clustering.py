import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import datetime
from Utils import dendrogram_plot_labels, dendrogram_plot
from scipy.signal import savgol_filter

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# Overlap grid
data_per_segment = 260
overlap_grid = 0.4

# Loop over data segment and overlap grid
time_series_list = []
omitted_codes = []
labels = []

# Loop over filenames
for filename in all_files:

    # Get filename
    file_id = filename.rsplit('/', 1)[-1]

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

# Sum over all hydrology stations at every point in time, t
time_series_total = np.sum(np.array(time_series_list), axis=0)
counter = 0
time_slice_list = [] # Streamflow data
spectral_list = []
labels = [] # Year
leap_year_list = [1980,1984,1988,1992,1996,2000,2004,2008,2012,2016]
year = 1980

while len(time_slice_list) < 39:
    if year in leap_year_list:
        # Time domain
        ts = time_series_total[counter:counter + 365]
        time_slice_list.append(ts)
        # Spectral domain
        flow_ts_normalised = ts - np.mean(ts)
        f, Pxx_density = welch(flow_ts_normalised, fs=1, window='hann', nperseg=data_per_segment,
                               noverlap = overlap_grid * data_per_segment, scaling='density')
        log_spectrum = np.log(Pxx_density)
        spectral_list.append(log_spectrum)
        counter += 366
        print("Leap Year!")
        labels.append(year)

        # Plot spectrum and smoothing
        plt.plot(log_spectrum)
        plt.savefig(str(year)+ "_spectrum")
        plt.show()

    else:
        # Time domain
        ts = time_series_total[counter:counter + 365]
        time_slice_list.append(ts)
        # Spectral Domain
        flow_ts_normalised = ts - np.mean(ts)
        f, Pxx_density = welch(flow_ts_normalised, fs=1, window='hann', nperseg=data_per_segment,
                               noverlap = overlap_grid * data_per_segment, scaling='density')
        log_spectrum = np.log(Pxx_density)
        spectral_list.append(log_spectrum)

        # Append to counter
        counter += 365
        labels.append(year)

        # Plot spectrum and smoothing
        plt.plot(log_spectrum)
        plt.savefig(str(year) + "_spectrum")
        plt.show()

    year += 1
    print("Counter", counter)
    print("Year", year)

# Loop over the list
time_distance_matrix = np.zeros((len(time_slice_list), len(time_slice_list)))
spectrum_distance_matrix = np.zeros((len(time_slice_list), len(time_slice_list)))
for i in range(len(time_slice_list)):
    for j in range(len(time_slice_list)):
        # Temporal
        ts_i = time_slice_list[i] # /np.sum(time_slice_list[i])
        ts_j = time_slice_list[j] # /np.sum(time_slice_list[j])
        # Spectra
        s_i = spectral_list[i] #/np.sum(spectral_list[i])
        s_j = spectral_list[j] #/np.sum(spectral_list[j])

        # Distance matrix
        time_distance_matrix[i,j] = np.sum(np.abs(ts_i - ts_j))
        spectrum_distance_matrix[i, j] = np.sum(np.abs(s_i - s_j))

# Temporal Heatmap
plt.matshow(time_distance_matrix)
plt.show()

# Spectral Heatmap
plt.matshow(spectrum_distance_matrix)
plt.show()

# Dendrogram
dendrogram_plot_labels(time_distance_matrix, "_L1_", "_Year_Temporal_", labels=labels)
dendrogram_plot_labels(spectrum_distance_matrix, "_L1_", "_Year_Spectral_", labels=labels)