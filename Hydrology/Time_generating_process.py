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

def ts_offset_2(ts_i, process_c, max_offset):
    # Learn offset
    l2_norms = []
    for j in range(1,max_offset, 1):
        ts_slice = ts_i[j:]
        process_c_slice = process_c[:-j]
        l2_norm = np.sum((ts_slice - process_c_slice)**2) * 1/len(ts_slice)
        l2_norms.append(l2_norm)
    argmin_l1 = np.argmin(l2_norms)
    return argmin_l1

# Time series i and j
time_series_array = np.array(time_series_list)
mean_process_init = np.mean(time_series_array, axis=0)
counter = 2

# Offsets and optimal time series
offsets = []
optimal_ts_list = []
truncated_end = []
argmin_1_list = []
# Loop over all the time series - STEP 1
for i in range(1, len(time_series_list)):

    # Time series i
    ts_i = time_series_list[i]

    # Compute argmin
    argmin = ts_offset(ts_i, mean_process_init, max_offset)
    argmin_1_list.append(argmin)
    offsets.append(argmin) # Append minimum to offsets list

    # Slice based on argmin
    ts_optimal = ts_i[argmin:(len(ts_i)-max_offset+argmin)]
    truncated_end.append(ts_i[(len(ts_i)-max_offset+argmin):])
    ts_optimal_scaled = ts_optimal * (1/len(time_series_list))
    optimal_ts_list.append(ts_optimal_scaled)
    counter += 1
    print("Iteration", i)
    print(argmin)

# Make optimal time series array and total (summing over rows)
optimal_ts_array = np.array(optimal_ts_list)
optimal_ts_1 = np.sum(optimal_ts_array, axis=0)

# STEP 2
# Offsets and optimal time series
offsets_2 = []
optimal_ts_list_2 = []
optimal_ts_c = optimal_ts_1

# Loop over all the time series - STEP 1
for i in range(1, len(optimal_ts_list)):

    # Time series i
    opt_ts_i = optimal_ts_list[i]

    # Compute argmin
    argmin_2 = ts_offset(opt_ts_i, optimal_ts_1, 60)
    offsets_2.append(argmin_2) # Append minimum to offsets list

    if argmin_2 == 0 or argmin_1_list[i] < argmin_2 or (argmin_1_list[i] + argmin_2) > max_offset:
        print("No further adjustment")
    else:
        # Slice initially aligned time series
        opt_ts_1 = optimal_ts_list.pop(i)
        opt_ts_2_front_removed = list(opt_ts_1[argmin_2:]) # Further adjust the argmin at front
        opt_back_replace = list(truncated_end[i][-argmin_2:])
        opt_ts_2 = opt_ts_2_front_removed + opt_back_replace # Adjust at back (include previously discarded elements)

        # Updated list of aligned time series
        optimal_ts_list.insert(i, opt_ts_2)
        optimal_ts_c = np.sum(optimal_ts_list, axis=0)

    counter += 1
    print("Iteration", i)
    print(argmin_2)

# Make optimal time series array and total (summing over rows)
optimal_ts_array_2 = np.array(optimal_ts_list_2)
optimal_ts_2 = np.sum(optimal_ts_array_2, axis=0)

# Processes
plt.plot(mean_process_init, alpha=0.25, label="Initial (mean process)")
plt.plot(optimal_ts_1, alpha=0.25, label="Post-alignment-1")
plt.plot(optimal_ts_c, alpha=0.25, label="Post-alignment-2")
plt.legend()
plt.savefig("Alignment_procedure_L2")
plt.show()

# Write out optimal time series
optimal_ts_df = pd.DataFrame(optimal_ts_list)
optimal_ts_df.write_csv("/Users/tassjames/Desktop/hydrology/hydrology_ts_aligned_df.csv")

# Mean process and optimal process
mean_process_init_df = pd.DataFrame(mean_process_init) # Mean process
mean_process_init_df.to_csv("/Users/tassjames/Desktop/hydrology/Process/mean_process_init.csv")

optimal_ts_c_df = pd.DataFrame(optimal_ts_c) # Optimal process
optimal_ts_c_df.to_csv("/Users/tassjames/Desktop/hydrology/Process/optimal_process.csv")

# Generate centred process in time
process_c_centred = optimal_ts_c - np.mean(optimal_ts_c)

# Compute Power spectral density
f, Pxx_density = welch(process_c_centred, fs=1, window='hann', nperseg=data_per_segment,
                       noverlap=overlap_grid * data_per_segment, scaling='density')
log_spectrum = np.log(Pxx_density)
print("Spectrum length", len(log_spectrum))

# Plot power spectral density
plt.plot(f, log_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Log Power")
plt.title("Hydrological process PSD")
plt.savefig("Hydrological_Process_PSD_L2")
plt.show()

# Distribution of offsets
plt.hist(np.array(offsets).flatten(), bins=50)
plt.xlabel("Offsets")
plt.ylabel("Frequency")
plt.title("Offset distribution")
plt.savefig("Distribution_offsets_L2")
plt.show()

# Write Power spectrum script
log_spectrum_df = pd.DataFrame(Pxx_density) # Core Log PSD
log_spectrum_df.to_csv("/Users/tassjames/Desktop/hydrology/Process/PSD_core_L2.csv")