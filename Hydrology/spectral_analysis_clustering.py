from scipy.fftpack import fft, fftshift, fftfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import datetime

make_plots = True

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# Overlap grid
data_per_segment = 512
overlap_grid = 0.7

# Loop over data segment and overlap grid
power_spectra = []
omitted_codes = []
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
    if min_ > lb or max_ < ub:
        # Code of omitted file
        omitted_codes.append(file_id)
        print("Omit ", file_id)
    else:
        mask = (df_slice['Date'] >= '1980-01-01') & (df_slice['Date'] <= '2019-01-01')
        df_date_slice = df.loc[mask]
        flow_ts = df_date_slice[["Date", "Flow (ML)"]]  # Date and Flow in new date range
        flow_ts = np.array(flow_ts["Flow (ML)"])
        flow_ts = flow_ts # Normalise by mean of time series

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
        f, Pxx_density = welch(flow_ts, fs=1, window='hann', nperseg=data_per_segment,
                               noverlap = overlap_grid * data_per_segment, scaling='density')
        log_spectrum = np.log(Pxx_density)
        log_spectrum_normalised = log_spectrum - np.mean(log_spectrum)
        print("Spectrum length", len(log_spectrum_normalised))
        power_spectra.append(log_spectrum_normalised)

        # log periodogram sliced
        idx = np.round(np.linspace(0, len(flow_periodogram_mean_adj) - 1, len(log_spectrum_normalised))).astype(int)
        log_periodogram_sliced = flow_periodogram_mean_adj[idx]

        if make_plots:
            plt.plot(np.linspace(0,0.5,len(log_periodogram_sliced)), log_periodogram_sliced)
            plt.plot(np.linspace(0,0.5,len(flow_periodogram_mean_adj)),flow_periodogram_mean_adj, alpha=0.2)
            plt.show()

        # Iteration
        print("Run over files", filename)

# Make list of lists an array
power_spectra_array = np.array(power_spectra)

