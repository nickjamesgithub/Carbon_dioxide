import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift

# Test data read in
data = pd.read_csv("test_download1.csv", skiprows = 26)
data.Date = pd.to_datetime(data.Date)
data.set_index('Date', inplace=True)
data_monthly = data.resample('MS').sum()
flow_monthly = data["Flow (ML)"]

# Plot flow
plt.plot(flow_monthly)
plt.plot(data_monthly)
plt.show()

# Plot periodogram
log_periodogram = np.log(np.abs(fft(fftshift(flow_monthly)))**2)
log_periodogram_norm = log_periodogram - np.mean(log_periodogram)

plt.plot(log_periodogram_norm[0:len(log_periodogram_norm)//2])
plt.show()