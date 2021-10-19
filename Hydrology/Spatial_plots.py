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

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# HR station details
hr_station_details = pd.read_csv("/Users/tassjames/Desktop/hydrology/hrs_station_details_cluster.csv",
                                 index_col=None, header=0, skiprows=11)

# initialize an axis
fig, ax = plt.subplots(figsize=(8,6))

hr_station_details.plot(x="Longitude", y="Latitude", kind="scatter",
        title=f"Stations in Australia",
        ax=ax, hue='label')# add grid
ax.grid(b=True, alpha=0.5)
plt.show()
