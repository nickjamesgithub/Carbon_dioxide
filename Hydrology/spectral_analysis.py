from scipy.fftpack import fft, fftshift, fftfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
import scipy.optimize as spo
from Utilities import gaussian_kernel_gramix

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, skiprows=26)
    df_slice = df[["Date", "Flow (ML)", "Bureau QCode"]]
    flow = df_slice[["Date", "Flow (ML)"]] # Date and Flow
    flow_ts = np.array(df_slice["Flow (ML)"])  # Make flow an array

    # Frequency Domain
    freqs = fftfreq(len(flow_ts), d=(1 - 0) / (2 * np.pi))  # Generate Frequency Grid
    freqs = freqs[0:len(freqs) // 2]  # Take first half of frequency grid
    freqs = np.reshape(freqs, (len(freqs),1)) # This has to be a matrix due to the kernel specifications.
    freq_length = len(freqs) # Calculate length of frequency Grid
    flow_periodogram = np.log(np.abs((fft(fftshift(flow_ts)))**2))
    flow_periodogram_mean = (np.log(np.abs((fft(fftshift(flow_ts))) ** 2)) - np.mean(flow_periodogram))
    flow_periodogram_mean_adj = flow_periodogram_mean[0:len(flow_periodogram_mean)//2]
    N = len(flow_periodogram)

    # Plot of power spectrum/log periodogram
    plt.plot(flow_ts)
    plt.show()

    # Plot of power spectrum/log periodogram
    plt.plot(freqs, flow_periodogram_mean_adj)
    plt.show()

    # Overlap grid
    overlap_grid = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in range(len(overlap_grid)):
        f, Pxx_density = welch(flow_ts, fs=1, window='hann', nperseg=256, noverlap = overlap_grid[i] * 256,
                            scaling='density')
        plt.figure()
        plt.semilogy(f, np.log(Pxx_density))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Power Spectrum')
        plt.show()

        test = np.log(Pxx_density) - np.mean(np.log(Pxx_density))

        # Plot of power spectral density
        plt.plot(f, test)
        plt.plot(np.linspace(0,0.5,len(flow_periodogram_mean_adj)), flow_periodogram_mean_adj, alpha=0.25)
        plt.show()

        block = 1