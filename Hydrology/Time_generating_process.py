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

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# Overlap grid
data_per_segment = 3750
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

# x = time_series_list[0]
# for i in range(len(time_series_list)):
#     x += time_series_list[i]
#
# plt.plot(x)
# plt.title("Sum of titles")
# plt.show()

# Offset list
offsets_list = []
max_offset = 365

def ts_offset(ts_i, process_c, max_offset):
    # Learn offset
    l1_norms = []
    for j in range(1,max_offset, 1):
        ts_slice = ts_i[j:]
        process_c_slice = process_c[:-j]
        l1_norm = np.sum(np.abs(ts_slice - process_c_slice)) * 1/len(process_c_slice)
        l1_norms.append(l1_norm)
    argmin_l1 = np.argmin(l1_norms)
    return argmin_l1

# Time series i and j
process_c = time_series_list[0][:-max_offset]
process = time_series_list[0]
counter = 2
offsets = []
# Loop over all the time series
for i in range(1, len(time_series_list)):

    # Time series i
    ts_i = time_series_list[i]

    # Compute argmin
    argmin = ts_offset(ts_i, process, max_offset)
    offsets.append(argmin) # Append minimum to offsets list

    # Slice based on argmin
    ts_optimal = ts_i[argmin:(len(ts_i)-max_offset+argmin)]
    process_c = (process_c + ts_optimal) * 1/counter
    counter += 1
    print("Iteration", i)
    print(argmin)

# Process
plt.plot(df_date_slice["Date"][max_offset:], process_c)
plt.title("Governing Hydrological Process")
plt.savefig("Hydrological_process")
plt.show()

# Plot periodogram
process_c_centred = process_c - np.mean(process_c)
log_periodogram = (np.log(np.abs(fft(fftshift(process_c_centred )))**2))[1:len(process_c)//2]
freqs = np.linspace(0,0.5, len(log_periodogram))
plt.plot(freqs, log_periodogram)
plt.xlabel("Frequency")
plt.ylabel("Log Power")
plt.title("Hydrological process PSD")
plt.savefig("Hydrological_Process_PSD")
plt.show()

# Distribution of offsets
plt.hist(np.array(offsets).flatten(), bins=50)
plt.xlabel("Offsets")
plt.ylabel("Frequency")
plt.title("Offset distribution")
plt.savefig("Distribution_offsets")
plt.show()

# Write script for core process to hydrology folder
process_df = pd.DataFrame(process_c) # Core process time series
process_df.to_csv("/Users/tassjames/Desktop/hydrology/Process/time_series_core.csv")
log_periodogram_df = pd.DataFrame(log_periodogram) # Core Log periodogram
log_periodogram_df.to_csv("/Users/tassjames/Desktop/hydrology/Process/log_periodogram_core.csv")



