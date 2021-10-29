import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from matplotlib import cm

data = "aligned" # raw / aligned

# Import data
if data == "aligned":
    hydrology = pd.read_csv("/Users/tassjames/Desktop/hydrology/hydrology_ts_aligned_df.csv")
    hydrology_slice = hydrology.iloc[1:,1:].transpose()
    # hydrology_returns = np.log(np.nan_to_num((hydrology_slice).diff())) # Compute log returns of hydrology time series

    # Model parameters
    smoothing_rate = 365
    corr_1 = [] # Explanatory variance of first eigenvalue list
    norm_eigenvector_1 = []
    norm_eigenvector_2 = []

    for i in range(smoothing_rate, len(hydrology_slice)): # len(hydrology_slice)
        # Take market correlation matrix
        # market_returns_slice = hydrology_slice.iloc[(i - smoothing_rate):i, :]
        hydrology_new = hydrology_slice.iloc[(i - smoothing_rate):i, :]
        hydrology_correlation = np.nan_to_num(hydrology_new.corr())  # Compute correlation matrix
        h_vals, h_vecs = eigsh(np.array(hydrology_correlation), k=3, which='LM')
        h_vals_1 = h_vals[-1]/len(hydrology_correlation)

        h_vecs_1 = h_vecs[:,0] / np.sum(np.abs(h_vecs[:,0]))
        h_vecs_2 = h_vecs[:, 1] / np.sum(np.abs(h_vecs[:, 1]))

        # Append eigenvector and eigenvalue to list
        norm_eigenvector_1.append(h_vecs_1)
        norm_eigenvector_2.append(h_vecs_2)
        corr_1.append(h_vals_1)
        print(" Simulation " + str(i))

    # Normalized eigenvalue 1
    # Date axis
    plt.plot(np.linspace(0,len(corr_1), len(corr_1)), corr_1)
    plt.ylim(0,1)
    plt.title("Hydrology Process normalised eigenvalue 1")
    plt.savefig("Hydrology_Process_eigenvalue_1_diff_aligned")
    plt.show()

    # Convert into an array
    norm_eigenvector_1_array = np.array(norm_eigenvector_1)

    # 3D Plot Dimensions
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.linspace(1,len(norm_eigenvector_1[0]),len(norm_eigenvector_1[0]))
    Y = np.linspace(1980,2019,len(norm_eigenvector_1))
    X, Y = np.meshgrid(X, Y)

    # Plot the surface
    surf = ax.plot_surface(X, Y, norm_eigenvector_1_array, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Eigenvector coefficient')
    ax.set_ylabel('Time')
    ax.set_zlabel('Absolute size of coefficient')
    plt.savefig("Eigenvector_normalised_TV1_aligned")
    plt.show()

    print("Variance", np.var(norm_eigenvector_1_array))

# Import data
if data == "raw":
    hydrology = pd.read_csv("/Users/tassjames/Desktop/hydrology/hydrology_ts_df.csv")
    hydrology_slice = hydrology.iloc[1:,1:].transpose()
    # hydrology_returns = np.log(np.nan_to_num((hydrology_slice).diff())) # Compute log returns of hydrology time series

    # Model parameters
    smoothing_rate = 365
    corr_1 = [] # Explanatory variance of first eigenvalue list
    norm_eigenvector_1 = []
    norm_eigenvector_2 = []

    for i in range(smoothing_rate, len(hydrology_slice)): # len(market_returns)
        # Take market correlation matrix
        # market_returns_slice = hydrology_slice.iloc[(i - smoothing_rate):i, :]
        hydrology_new = hydrology_slice.iloc[(i - smoothing_rate):i, :]
        hydrology_correlation = np.nan_to_num(hydrology_new.corr())  # Compute correlation matrix
        h_vals, h_vecs = eigsh(np.array(hydrology_correlation), k=3, which='LM')
        h_vals_1 = h_vals[-1]/len(hydrology_correlation)

        h_vecs_1 = h_vecs[:,0] / np.sum(np.abs(h_vecs[:,0]))
        h_vecs_2 = h_vecs[:, 1] / np.sum(np.abs(h_vecs[:, 1]))

        # Append eigenvector and eigenvalue to list
        norm_eigenvector_1.append(h_vecs_1)
        norm_eigenvector_2.append(h_vecs_2)
        corr_1.append(h_vals_1)
        print(" Simulation " + str(i))

    # Normalized eigenvalue 1
    # Date axis
    plt.plot(np.linspace(0,len(corr_1), len(corr_1)), corr_1)
    plt.ylim(0,1)
    plt.title("Hydrology Process normalised eigenvalue 1")
    plt.savefig("Hydrology_Process_eigenvalue_1_diff_raw")
    plt.show()

    # Convert into an array
    norm_eigenvector_1_array = np.array(norm_eigenvector_1)

    # 3D Plot Dimensions
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.linspace(1,len(norm_eigenvector_1[0]),len(norm_eigenvector_1[0]))
    Y = np.linspace(1980,2019,len(norm_eigenvector_1))
    X, Y = np.meshgrid(X, Y)

    # Plot the surface
    surf = ax.plot_surface(X, Y, norm_eigenvector_1_array, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Eigenvector coefficient')
    ax.set_ylabel('Time')
    ax.set_zlabel('Absolute size of coefficient')
    plt.savefig("Eigenvector_normalised_TV1_raw")
    plt.show()

    print("Variance", np.var(norm_eigenvector_1_array))

