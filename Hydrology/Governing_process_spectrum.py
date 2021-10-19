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
from matplotlib import cm

# Overlap grid
data_per_segment = 3750
overlap_grid = 0.4

# Import governing time series
governing_flow_ts = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/time_series_core.csv")
flow_ts = governing_flow_ts.iloc[1:,1]
smoothing_rate = 365

# Power Spectrum list
spectral_list = []

for i in range(smoothing_rate, len(flow_ts)): #

    # flow slice
    flow_slice = flow_ts[(i-smoothing_rate):i]

    # Compute Power spectral density
    flow_ts_normalised = flow_ts - np.mean(flow_ts)
    f, Pxx_density = welch(flow_ts_normalised, fs=1, window='hann', nperseg=data_per_segment,
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
Y = np.arange(0,len(spectral_list))
X, Y = np.meshgrid(X, Y)

# Plot the surface
surf = ax.plot_surface(X, Y, spectral_array, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Frequency')
ax.set_ylabel('Time')
ax.set_zlabel('Log spectrum')
plt.savefig("Governing_process_spectrum_time_varying")
plt.show()


