from scipy.fftpack import fft, fftshift
import numpy as np
import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("test_download1.csv", skiprows=30)
time = data.iloc[29:,1]
time_normalised = time

# Plot the spectrum
log_periodogram = np.log(np.abs(fft(fftshift(time))))[0:len(time_normalised)//2]
log_periodogram_mean = log_periodogram - np.mean(log_periodogram)
freqs = np.linspace(0,0.5,len(log_periodogram))
max_freq = freqs[np.argmax(log_periodogram_mean)]
print("maximum frequency", max_freq)

# Plot log periodogram
plt.plot(freqs, log_periodogram_mean)
plt.show()
