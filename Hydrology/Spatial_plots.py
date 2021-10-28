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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

make_plots = False

# Set path to read in hydrology data
path = '/Users/tassjames/Desktop/hydrology/data' # use your path
all_files = glob.glob(path + "/*.csv")

# HR station details
hr_station_details = pd.read_csv("/Users/tassjames/Desktop/hydrology/hrs_station_details.csv",
                                 index_col=None, header=0, skiprows=11)
# # import omitted codes csv
# omitted_codes = pd.read_csv("/Users/tassjames/Desktop/hydrology/omitted_codes/omitted_codes.csv")
# omitted_codes_slice = omitted_codes.iloc[1:,1:]
#
# block = pd.to_numeric(hr_station_details['AWRC Station Number'])
# test = hr_station_details.loc[~((hr_station_details['AWRC Station Number'].isin(omitted_codes_slice))&(hr_station_details['AWRC Station Number'].isin(omitted_codes_slice))),:]
#
# # Conditional Drop
# conditional_drop = hr_station_details['AWRC Station Number'].isin(omitted_codes_slice)
# hr_station_details.drop(hr_station_details[conditional_drop].index, inplace = True)

# Updated HR station details post drop
hr_station_details_drop = hr_station_details
X = hr_station_details.loc[:,['Latitude','Longitude']]

# Find optimal number of clusters
K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = hr_station_details[['Latitude']]
X_axis = hr_station_details[['Longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# Plot optimal number of clusters
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve - Unweighted K-Means')
plt.savefig("Spatial_elbow")
plt.show()

# k means clustering
kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(X) # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X)
# Append cluster labels to HR station dataset
hr_station_details['cluster_label'] = kmeans.fit_predict(X)

# Initialize an axis
fig, ax = plt.subplots(figsize=(8,6))
hr_station_details.plot(x="Longitude", y="Latitude", kind="scatter",
        title=f"Stations in Australia", c="cluster_label",
        ax=ax, colormap = 'viridis')# add grid
ax.grid(b=True, alpha=0.5)
plt.savefig("Spatial_hydrology_stations_plot")
plt.show()

