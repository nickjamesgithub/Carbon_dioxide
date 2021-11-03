import glob
import datetime
import scipy.optimize as sco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib import cm

data = "aligned" # initial or aligned
make_plots = False

if data == "initial":
    initial = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/Initial_process.csv")
    process = initial.iloc[1:,1]
    date_index_array_1 = pd.date_range("1980-01-01", "2019-01-01", len(process))

    # Changepoints
    cps = [6967, 7756, 10188, 10612, 11420, 12054, 12528, 13048, 13868]
    for i in range(len(cps)):
        print("Changepoints", date_index_array_1[cps[i]])

if data == "aligned":
    optimal = pd.read_csv("/Users/tassjames/Desktop/hydrology/Process/Aligned_process.csv")
    process = optimal.iloc[1:, 1]
    date_index_array_1 = pd.date_range("1980-01-01", "2019-01-01", len(process))

    cps = [2923, 4460, 5500, 6977, 7397, 7762, 8754, 10187, 13621]
    for i in range(len(cps)):
        print("Changepoints", date_index_array_1[cps[i]])

# Spectral analysis on weighted optimal process
data_per_segment = 3750
overlap_grid = 0.4
smoothing_rate = 365

# Power Spectrum list
spectral_list = []
process = np.array(process)

for i in range(smoothing_rate, len(process)): #

    # flow slice
    core_flow_slice = process[(i-smoothing_rate):i]

    # Compute Power spectral density
    core_flow_ts_normalised = process - np.mean(process)
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
plt.savefig("Governing_process_spectrum_time_varying_"+data)
plt.show()

