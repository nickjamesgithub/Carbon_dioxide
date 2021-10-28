import glob
import datetime
import scipy.optimize as sco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib import cm

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

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

# Time series i and j
process_c = time_series_list[0][:-max_offset] * 1/len(time_series_list)
process = time_series_list[0] * 1/len(time_series_list)
counter = 2
offsets = []
maximums = []
optimal_ts_list = []

# Loop over all the time series
for i in range(1, len(time_series_list)):

    # Time series i
    ts_i = time_series_list[i]

    # Compute argmin
    argmin = ts_offset(ts_i, process, max_offset)
    offsets.append(argmin) # Append minimum to offsets list

    # Slice based on argmin
    ts_optimal = ts_i[argmin:(len(ts_i)-max_offset+argmin)].flatten()
    optimal_ts_list.append(ts_optimal) # Append optimal time series list to core list

# Form covariance matrix of signals
optimal_ts_df = pd.DataFrame(optimal_ts_list).transpose()
optimal_ts_returns = np.log(optimal_ts_df).diff()
optimal_ts_returns.replace(np.nan, 0, inplace=True)
optimal_ts_df.replace(np.nan, 0, inplace=True)
mean_returns = (optimal_ts_returns.fillna(0.00001)).pct_change().mean()

# Covariance cleaning
cov = optimal_ts_returns.cov()
cov.replace(np.nan, 0, inplace=True)
cov.fillna(0.00001)
num_portfolios = 100000
rf = 0.0

def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio

def calc_portfolio_std(weights, mean_returns, cov):
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return portfolio_std

def min_variance(mean_returns, cov):
    num_assets = len(mean_returns)
    args = (mean_returns, cov)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(calc_portfolio_std, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

min_port_variance = min_variance(mean_returns, cov)
weights = min_port_variance.x
print(min_port_variance)
print("Check sum of weights", np.sum(weights))

# Access weights
test = weights[0]
weighted_process = 0
weighted_optimal_ts_list = []
for i in range(len(optimal_ts_list)):
    weight_c = weights[i]
    optimal_ts_c = optimal_ts_list[i]
    weighted_optimal_ts = weight_c * optimal_ts_c
    weighted_optimal_ts_list.append(weighted_optimal_ts) # Append weighted optimal time series list
    print("Optimal Iteration", i)

# Make weighted optimal array
weighted_optimal_ts_array = np.array(weighted_optimal_ts_list)
weighted_optimal_process = np.sum(weighted_optimal_ts_array, axis=0)

# Plot of Distribution of weights
weight_grid = np.linspace(1,len(weights),len(weights))
plt.bar(weight_grid, weights)
plt.ylabel("Weight")
plt.xlabel("Hydrology signal")
plt.savefig("Weight_hydrology_signal")
plt.show()

# Plot core process
plt.plot(weighted_optimal_process)
plt.title("Weighted optimal process")
plt.savefig("Weighted_optimal_process")
plt.show()

# Spectral analysis on weighted optimal process
data_per_segment = 3750
overlap_grid = 0.4
smoothing_rate = 365

# Power Spectrum list
spectral_list = []

for i in range(smoothing_rate, len(weighted_optimal_process)): #

    # flow slice
    core_flow_slice = weighted_optimal_process[(i-smoothing_rate):i]

    # Compute Power spectral density
    core_flow_ts_normalised = weighted_optimal_process - np.mean(weighted_optimal_process)
    f, Pxx_density = welch(core_flow_ts_normalised, fs=1, window='hann', nperseg=data_per_segment,
                           noverlap = overlap_grid * data_per_segment, scaling='density')
    log_spectrum = np.log(Pxx_density)
    print("Iteration", i)
    spectral_list.append(log_spectrum)

# Convert spectral list to an array
spectral_array = np.array(spectral_list)

# 3D Plot Dimensions
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.linspace(0,0.5,len(spectral_list[0]))
Y = np.linspace(1980,2019, len(spectral_list))
X, Y = np.meshgrid(X, Y)

# Plot the surface
surf = ax.plot_surface(X, Y, spectral_array, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Frequency')
ax.set_ylabel('Time')
ax.set_zlabel('Log spectrum')
plt.savefig("Governing_process_spectrum_time_varying")
plt.show()

