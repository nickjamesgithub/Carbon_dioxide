from scipy.fftpack import fft, fftshift, fftfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import datetime

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# Overlap grid
data_per_segment = [3500, 3750, 4000, 4250, 4500, 4750, 5000] # 128, 256, 512, 1024, 2048, 4096
overlap_grid = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

# Loop over data segment and overlap grid
whittle_likelihood_totals = []
parameters = [] # Parameters
power_spectra = []

# Get minimum and maximum dates
minimum_date = []
maximum_date = []
omitted_codes = []

for i in range(len(data_per_segment)):
    for j in range(len(overlap_grid)):
        # Likelihood etc.
        likelihood_list = []  # Whittle Likelihood score
        spectrum_function_list = [] # Spectrum function

        # Loop over filenames
        for filename in all_files:
            # Get filename
            file_id = filename.rsplit('/', 1)[-1]

            df = pd.read_csv(filename, index_col=None, header=0, skiprows=26)
            df_slice = df[["Date", "Flow (ML)", "Bureau QCode"]]
            df["Date"] = pd.to_datetime(df_slice["Date"])
            flow = df_slice[["Date", "Flow (ML)"]] # Date and Flow
            flow_ts = np.array(df_slice["Flow (ML)"])  # Make flow an array
            min_date = np.min(df_slice["Date"])
            minimum_date.append(min_date)
            max_date = np.max(df_slice["Date"])
            maximum_date.append(max_date)

            # Datetime, datetime
            min_ = datetime.datetime.strptime(min_date, '%Y-%m-%d')
            max_ = datetime.datetime.strptime(max_date, '%Y-%m-%d')
            # If date outside certain range omit:
            lb = datetime.datetime(1980, 1, 1)
            ub = datetime.datetime(2019, 1, 1)
            if min_ > lb or max_ < ub:
                # Code of omitted file
                omitted_codes.append(file_id)
            else:
                # Frequency Domain
                freqs = fftfreq(len(flow_ts), d=(1 - 0) / (2 * np.pi))  # Generate Frequency Grid
                freqs = freqs[0:len(freqs) // 2]  # Take first half of frequency grid
                freqs = np.reshape(freqs, (len(freqs),1)) # This has to be a matrix due to the kernel specifications.
                freq_length = len(freqs) # Calculate length of frequency Grid
                flow_periodogram = np.log(np.abs((fft(fftshift(flow_ts)))**2))
                flow_periodogram_mean = (np.log(np.abs((fft(fftshift(flow_ts))) ** 2)) - np.mean(flow_periodogram))
                flow_periodogram_mean_adj = flow_periodogram_mean[0:len(flow_periodogram_mean)//2]
                N = len(flow_periodogram)

                # Compute spectral density etc
                f, Pxx_density = welch(flow_ts, fs=1, window='hann', nperseg=data_per_segment[i],
                                       noverlap = overlap_grid[j] * data_per_segment[i], scaling='density')
                log_spectrum = np.log(Pxx_density)
                log_spectrum_normalised = log_spectrum - np.mean(log_spectrum)

                # Compute Whittle Likelihood
                def whittlelikelihood(y, spectrum_estimate):  # y should be the log adjusted periodogram
                    n = len(y)
                    g = spectrum_estimate
                    likelihood = -np.sum(g + np.exp(y) / np.exp(g))
                    return likelihood

                # Compute Whittle likelihood score
                idx = np.round(np.linspace(0, len(flow_periodogram_mean_adj) - 1, len(log_spectrum_normalised))).astype(int)
                log_periodogram_sliced = flow_periodogram_mean_adj[idx]

                if make_plots:
                    plt.plot(np.linspace(0,0.5,len(log_periodogram_sliced)), log_periodogram_sliced)
                    plt.plot(np.linspace(0,0.5,len(flow_periodogram_mean_adj)),flow_periodogram_mean_adj, alpha=0.2)
                    plt.show()

                likelihood_score = whittlelikelihood(log_periodogram_sliced, log_spectrum_normalised) # Compute likelihood
                likelihood_list.append(likelihood_score)
                print("Whittle likelihood score ", likelihood_score)
                print("Run over files", filename)
                print("Length periodogram", len(log_periodogram_sliced))

            # Cumulative sum for Whittle likelihood and parameters
            whittle_likelihood_sum = np.sum(likelihood_list)
            whittle_likelihood_totals.append([whittle_likelihood_sum, data_per_segment[i], overlap_grid[j]])
            print("Whittle Likelihood ", whittle_likelihood_sum)
            parameters.append([data_per_segment[i], overlap_grid[j]])

# Make list of lists an array
whittle_likelihood_array = np.array(whittle_likelihood_totals)
w_likelihood_argmax = np.argmax(whittle_likelihood_array[:,0])

# Optimal data/segment and overlap grid
optimal_parameters = parameters[w_likelihood_argmax]
print("Optimal parameters ", optimal_parameters)

min_date = max(minimum_date)
print("Minimum Date", min_date)

max_date = min(maximum_date)
print("Maximum Date", max_date)

print("Omitted codes", np.unique(omitted_codes))
print("Length omitted codes", len(np.unique(omitted_codes)))

