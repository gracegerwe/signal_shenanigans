import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

# Load the WAV file
filename = "/Users/gracegerwe/Documents/Empath/Neuralink_Raw_Data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav"
sr, data = wav.read(filename)

# Create time axis
time_axis = np.linspace(0, len(data) / sr, num=len(data))

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time_axis, data, lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Neuralink Electrode Signal Waveform")
plt.grid(True)
plt.show(block=False)  # Show without blocking execution
plt.pause(0.1)  # Small delay to allow multiple windows

# Remove DC bias
data_centered = data - np.mean(data)

# Compute FFT on centered data
fft_data = np.abs(fft(data_centered))
freqs = np.fft.fftfreq(len(data_centered), 1/sr)

# Plot FFT Spectrum
plt.figure(figsize=(10, 4))
plt.plot(freqs[:1000], fft_data[:1000])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Neural Data")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

def bandpass_filter(data, lowcut=10, highcut=300, sr=19531, order=4):
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Apply bandpass filter
filtered_data = bandpass_filter(data, lowcut=10, highcut=300, sr=sr)

# Plot filtered signal
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(filtered_data) / sr, num=len(filtered_data)), filtered_data, lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Bandpass Filtered Neuralink Signal (10-300 Hz)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Compute FFT on the filtered data
fft_filtered_data = np.abs(fft(filtered_data))
freqs_filtered = np.fft.fftfreq(len(filtered_data), 1/sr)

# Plot FFT Spectrum of the Filtered Signal
plt.figure(figsize=(10, 4))
plt.plot(freqs_filtered[:1000], fft_filtered_data[:1000])  # Show only first 1000 frequency bins
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Bandpass-Filtered Neural Data (10-300 Hz)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Allow multiple windows

# Show all plots together
plt.show()