import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

matrix_computations = True
sector_plots = True
transitivity_plots = True

# Import Various Sectors
initial = pd.read_csv("/Users/tassjames/Desktop/hydrology/adaptspec/hydrology_aspec/hydrology/estimates1.csv")
aligned = pd.read_csv("/Users/tassjames/Desktop/hydrology/adaptspec/hydrology_aspec/hydrology/estimates2.csv")

# Declare sectors
cities = [initial, aligned]

labels = ["Initial", "Aligned"]
labels_ts = ["Initial", "Aligned"]

# Loop over time-varying surfaces and plot 3d spectra
for i in range(len(cities)):
    gt_spectrum = cities[i]
    gt_spectrum = gt_spectrum.transpose()
    ts_length = len(gt_spectrum.iloc[:,0])
    gt_spectrum = np.array(gt_spectrum)[1:ts_length, :]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(25, 45)
    frequency = np.linspace(0, 0.5, 100)
    date_index = pd.date_range("1980-01-01", "2019-01-01", freq='D').strftime('%Y-%m-%d')
    # time_array = np.array(date_index)
    time_array = np.linspace(1980,2019, len(gt_spectrum))
    X, Y = np.meshgrid(frequency, time_array)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, gt_spectrum, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Time")
    ax.set_zlabel("Log PSD")
    plt.savefig(labels[i]+"_3d_surface")
    plt.show()
