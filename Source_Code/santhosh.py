import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Load the dataset
file_path = 'dataset3.csv'
data = pd.read_csv(file_path)

# Assuming the first column contains the raw signal data
signal = data.iloc[:, 0].to_numpy()

# Normalize the signal
def normalize_data(signal):
    return (signal - np.mean(signal)) / np.std(signal)

normalized_signal = normalize_data(signal)

# Plot the spectrum analysis graph
def plot_spectrum(signal, sample_rate):
    """Plot the Power Spectral Density of the signal."""
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    psd = np.abs(fft(signal))**2
    plt.figure(figsize=(10, 5))
    plt.plot(freq[:n//2], psd[:n//2])  # Plotting only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Power Spectral Density (PSD) of Signal")
    plt.grid(True)
    plt.show()

# Assume a sample rate (example: 1000 Hz)
sample_rate = 1000

# Generate and display the spectrum analysis graph
plot_spectrum(normalized_signal, sample_rate)

# 1. Time-Domain Signal Plot
plt.figure(figsize=(10, 5))
plt.plot(normalized_signal, color='blue')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Representation of Signal')
plt.grid(True)
plt.show()

# 2. Histogram of Signal Amplitudes
plt.figure(figsize=(10, 5))
plt.hist(normalized_signal, bins=50, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.title('Histogram of Signal Amplitudes')
plt.grid(True)
plt.show()

# 3. Cumulative Distribution Function (CDF)
sorted_signal = np.sort(normalized_signal)
cdf = np.arange(len(sorted_signal)) / float(len(sorted_signal))
plt.figure(figsize=(10, 5))
plt.plot(sorted_signal, cdf, color='red')
plt.xlabel('Amplitude')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function (CDF) of Signal')
plt.grid(True)
plt.show()

# Calculate more signal features
def calculate_signal_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['peak_to_peak'] = np.ptp(signal)
    features['energy'] = np.sum(signal**2)
    features['crest_factor'] = np.max(np.abs(signal)) / features['rms']
    return features

# Calculate and print signal features
signal_features = calculate_signal_features(normalized_signal)
print("Signal Features:")
for feature, value in signal_features.items():
    print(f"{feature}: {value}")
