import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

# Import data
hydrology = pd.read_csv("/Users/tassjames/Desktop/hydrology/hydrology_ts_df.csv")
hydrology_slice = hydrology.iloc[1:,1:].transpose()
hydrology_returns = np.nan_to_num(np.log(hydrology_slice).diff()) # Compute log returns of market

# Model parameters
smoothing_rate = 365*3
corr_1 = [] # Explanatory variance of first eigenvalue list

for i in range(smoothing_rate, len(hydrology_slice)): # len(market_returns)

    # Take market correlation matrix
    # market_returns_slice = hydrology_slice.iloc[(i - smoothing_rate):i, :]
    market_returns_slice = pd.DataFrame(hydrology_returns[(i - smoothing_rate):i, :])
    market_correlation = np.nan_to_num(market_returns_slice.corr())  # Compute correlation matrix
    m_vals, m_vecs = eigsh(market_correlation, k=len(market_correlation), which='LM')
    m_vals_1 = m_vals[-1]/len(market_correlation)
    corr_1.append(m_vals_1)
    print(" Simulation " + str(i))

# Normalized eigenvalue 1
plt.plot(corr_1)
plt.title("Hydrology Process normalised eigenvalue 1")
plt.savefig("Hydrology_Process_eigenvalue_1_diff")
plt.show()
