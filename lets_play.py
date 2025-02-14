import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

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
plt.show()

from scipy.fftpack import fft

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
plt.show()
