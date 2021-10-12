from scipy.fftpack import fft, fftshift, fftfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import datetime
from Utils import dendrogram_plot_labels
from matplotlib import colors

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
max_offset = 180

# Set offset matrix
offset_matrix = np.zeros((len(time_series_list),len(time_series_list)))
l1_matrix = np.zeros((len(time_series_list),len(time_series_list)))

# Time series i and j
for i in range(len(time_series_list)):
    for j in range(len(time_series_list)): #len(time_series_list)

        # l1 lag
        l1_scores_i = []
        for k in range(0,max_offset,1):
            # time series i and j
            if k == 0:
                flow_i = time_series_list[i]
                flow_j = time_series_list[j]
            else:
                # we are jumping into the function and pulling it back k:
                flow_i = time_series_list[i][k:]
                flow_j = time_series_list[j][:-k]
            # L^1 difference
            l1_flow_i = np.sum(np.abs(flow_i-flow_j))
            l1_scores_i.append(l1_flow_i)

        # Combine the L^1 product scores
        l1_scores = l1_scores_i
        idx = np.argmin(l1_scores)
        if i < j:
            offsets_list.append(idx)
        else:
            print("i>j")
        # L^1 distance
        l1_matrix[i,j] = np.sum(np.abs(time_series_list[i]-time_series_list[j]))
    print("Iteration ", i)

# Plot heatmap
plt.matshow(l1_matrix)
plt.title("L^1 matrix")
plt.savefig("L1_time_matrix")
plt.show()

# Compute Mean and diagonal
print("Mean offset matrix", np.mean(offset_matrix))
print("Diagonal offset matrix", np.diag(offset_matrix))

# Plot histogram of offsets
plt.hist(np.array(offset_matrix).flatten(), bins=100)
plt.xlabel("Offsets")
plt.ylabel("Frequency")
plt.show()

# Plot dendrograms
dendrogram_plot_labels(offset_matrix, "_time_offsets_", "_L1_", labels=labels)