import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pywt
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch
from scipy.signal import medfilt
from sklearn.decomposition import PCA

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

def notch_filter(data, freq=60, sr=19531, quality_factor=30):
    nyquist = sr / 2
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data)

# Apply notch filter at 60 Hz and harmonics
filtered_data = notch_filter(filtered_data, freq=60, sr=sr)
filtered_data = notch_filter(filtered_data, freq=120, sr=sr)  # Second harmonic
filtered_data = notch_filter(filtered_data, freq=180, sr=sr)  # Third harmonic

def wavelet_denoise(data, wavelet='db4', level=2):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    coeffs[1:] = [pywt.threshold(c, np.std(c), mode='soft') for c in coeffs[1:]]  # Denoise detail coefficients
    return pywt.waverec(coeffs, wavelet, mode="per")

# Plot filtered signal
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(filtered_data) / sr, num=len(filtered_data)), filtered_data, lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Bandpass & Notch Filtered Neuralink Signal (10-300 Hz)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Apply wavelet denoising
filtered_data = wavelet_denoise(filtered_data)

# Plot filtered signal
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(filtered_data) / sr, num=len(filtered_data)), filtered_data, lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Wavelet Denoised Neuralink Signal (10-300 Hz)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Compute FFT on the filtered data
fft_filtered_data = np.abs(fft(filtered_data))
freqs_filtered = np.fft.fftfreq(len(filtered_data), 1/sr)

# Define spike detection threshold (4x standard deviation)
threshold = np.std(filtered_data) * 4  

# Find spike locations
spike_indices = np.where(np.abs(data) > threshold)[0]

# Group nearby spikes (Refractory period: at least 5ms apart)
min_spike_distance = int(0.005 * sr)  # 5ms in samples
spike_times = spike_indices[np.insert(np.diff(spike_indices) > min_spike_distance, 0, True)]

# Count spikes
num_spikes = len(spike_times)

print(f"Total detected neural spikes: {num_spikes}")

# Plot detected spikes
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(filtered_data) / sr, num=len(filtered_data)), filtered_data, lw=0.5, label="Signal")
plt.scatter(spike_times / sr, filtered_data[spike_times], color='red', label="Detected Spikes")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Neural Spikes Detected: {num_spikes}")
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Compute inter-spike intervals (time between spikes)
isi = np.diff(spike_times) / sr  # Convert samples to seconds

# Plot ISI histogram
plt.figure(figsize=(6, 4))
plt.hist(isi, bins=50, edgecolor='black')
plt.xlabel("Inter-Spike Interval (s)")
plt.ylabel("Count")
plt.title("ISI Distribution (Neural Spike Timing)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Apply median filter for motion artifacts
filtered_data = medfilt(filtered_data, kernel_size=5)

# Spike detection (identify high amplitude events)
threshold = np.std(filtered_data) * 4  # Adjust for sensitivity
spike_indices = np.where(np.abs(filtered_data) > threshold)[0]

# Noise masking (keep only spikes, set everything else to zero)
spike_signal = np.zeros_like(filtered_data)
spike_signal[spike_indices] = filtered_data[spike_indices]

# Apply PCA to extract dominant neural components
spike_signal_reshaped = spike_signal.reshape(-1, 1)  # PCA needs 2D input
pca = PCA(n_components=1)
pca_signal = pca.fit_transform(spike_signal_reshaped).flatten()

# Compute FFT on PCA-Processed Spikes
fft_pca_spike = np.abs(fft(pca_signal))
freqs_pca = np.fft.fftfreq(len(pca_signal), 1/sr)

# Plot FFT spectrum of the filtered signal
plt.figure(figsize=(10, 4))
plt.plot(freqs_pca[:1000], fft_pca_spike[:1000])  # Show only first 1000 frequency bins
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of PCA-Processed Neural Spikes")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Allow multiple windows

# Plot filtered signal
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(pca_signal) / sr, num=len(pca_signal)), pca_signal, lw=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("{Post PCA}")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Show all plots together
plt.show()