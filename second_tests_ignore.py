import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io.wavfile as wav
import pywt
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.decomposition import PCA
import random
import pandas as pd

# Load the WAV file
filename = "/Users/gracegerwe/Documents/Empath/Neuralink_Raw_Data/0ab237b7-fb12-4687-afed-8d1e2070d621.wav"
sr, data = wav.read(filename)

# Create time axis
time_axis = np.linspace(0, len(data) / sr, num=len(data))

# Remove DC bias
data_centered = data - np.mean(data)

# Bandpass Filter
def bandpass_filter(data, lowcut=10, highcut=300, sr=19531, order=4):
    nyquist = sr / 2
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

filtered_data = bandpass_filter(data_centered, lowcut=10, highcut=300, sr=sr)

# Notch Filter at 60Hz and harmonics
def notch_filter(data, freq=60, sr=19531, quality_factor=30):
    nyquist = sr / 2
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data)

for notch_freq in [60, 120, 180]:
    filtered_data = notch_filter(filtered_data, freq=notch_freq, sr=sr)

# Wavelet Denoising
def wavelet_denoise(data, wavelet='db4', level=2):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    coeffs[1:] = [pywt.threshold(c, np.std(c), mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")

filtered_data = wavelet_denoise(filtered_data)

# Apply Median Filter for Motion Artifacts
filtered_data = medfilt(filtered_data, kernel_size=5)

# **Threshold Selection After Processing**
threshold = np.percentile(np.abs(filtered_data), 99)  # Use 99th percentile dynamically

# Spike Detection
spike_indices = np.where(np.abs(filtered_data) > threshold)[0]

# **Refractory Period Based on ISI**
min_spike_distance = int(0.01 * sr)  # 10ms refractory period
spike_times = spike_indices[np.insert(np.diff(spike_indices) > min_spike_distance, 0, True)]

# **Inter-Spike Interval (ISI) Calculation**
isi = np.diff(spike_times) / sr  # Convert samples to seconds

num_spikes = len(spike_times)
print(f"Total detected neural spikes: {num_spikes}")

# Apply PCA to Extract Dominant Neural Components
spike_signal = np.zeros_like(filtered_data)
spike_signal[spike_times] = filtered_data[spike_times]  # Keep only detected spikes
spike_signal_reshaped = spike_signal.reshape(-1, 1)
pca = PCA(n_components=1)
pca_signal = pca.fit_transform(spike_signal_reshaped).flatten()

# Compute FFT on PCA-Processed Spikes
fft_pca_spike = np.abs(fft(pca_signal))
freqs_pca = np.fft.fftfreq(len(pca_signal), 1/sr)

# Extract Features for Each Spike
window_size = int(0.002 * sr)  # 2ms window (~40 samples at 20kHz)
half_window = window_size // 2
spike_features = []

for spike in spike_times:
    if spike - window_size > 0 and spike + window_size < len(filtered_data):
        spike_waveform = filtered_data[spike - half_window: spike + half_window]

        # Find key spike features
        min_amp = np.min(spike_waveform)  # Negative-going peak
        max_amp = np.max(spike_waveform)  # Positive peak
        mid_amp = spike_waveform[len(spike_waveform) // 2]  # Middle hump

        min_time = (np.argmin(spike_waveform) - half_window) / sr * 1000  # Convert to ms
        max_time = (np.argmax(spike_waveform) - half_window) / sr * 1000
        mid_time = 0  # Middle of waveform

        # Estimate probability (SNR-based)
        noise_std = np.std(filtered_data)  
        snr = (max_amp - min_amp) / (2 * noise_std)  
        spike_prob = 1 / (1 + np.exp(-snr))  # Sigmoid function for probability

        # Store spike details
        spike_features.append([spike, min_amp, mid_amp, max_amp, min_time, mid_time, max_time, spike_prob])

# Merging Plots for Easier Analysis
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Trim filtered_data to match time_axis length
filtered_data = filtered_data[:len(time_axis)]

# Plot 1: Processed Signal with Spikes
axs[0, 0].plot(time_axis, filtered_data, lw=0.5, label="Filtered Signal")
axs[0, 0].scatter(spike_times / sr, filtered_data[spike_times], color='red', label="Detected Spikes")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].set_title(f"Neural Spikes Detected: {num_spikes}")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: ISI Distribution
axs[0, 1].hist(isi, bins=50, edgecolor='black')
axs[0, 1].set_xlabel("Inter-Spike Interval (s)")
axs[0, 1].set_ylabel("Count")
axs[0, 1].set_title("ISI Distribution (Neural Spike Timing)")
axs[0, 1].grid(True)

# Plot 3: FFT of PCA-Processed Spikes
axs[1, 0].plot(freqs_pca[:100], fft_pca_spike[:100])  # First 1000 bins
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Magnitude")
axs[1, 0].set_title("FFT of PCA-Processed Neural Spikes")
axs[1, 0].grid(True)

# Plot 4: PCA Processed Signal
axs[1, 1].plot(np.linspace(0, len(pca_signal) / sr, num=len(pca_signal)), pca_signal, lw=0.5)
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Amplitude")
axs[1, 1].set_title("Post PCA Signal")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Plot All Spikes in Different Colors*
colors = cm.rainbow(np.linspace(0, 1, len(spike_features)))

plt.figure(figsize=(10, 5))
for i, spike_data in enumerate(spike_features):
    spike_idx = spike_data[0]
    spike_waveform = filtered_data[spike_idx - window_size: spike_idx + window_size]
    time_axis_spike = np.linspace(-1, 1, len(spike_waveform))  # Center on spike

    plt.plot(time_axis_spike, spike_waveform, color=colors[i], alpha=0.7)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.title("All Detected Spikes (Each Colored Uniquely)")
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay

# Plot a Single Neuron Spike with Labels
random_spike = random.choice(spike_features)
spike_idx = random_spike[0]
spike_waveform = filtered_data[spike_idx - window_size: spike_idx + window_size]
time_axis_spike = np.linspace(-1, 1, len(spike_waveform))

# Extract key features
min_amp, mid_amp, max_amp, min_time, mid_time, max_time, prob = random_spike[1:]

plt.figure(figsize=(8, 5))
plt.plot(time_axis_spike, spike_waveform, color="black", lw=2, label="Spike Waveform")

# Mark key points
plt.scatter([min_time, mid_time, max_time], [min_amp, mid_amp, max_amp], color=['blue', 'purple', 'red'], s=100, label="Key Points")

# Annotate features
plt.text(min_time, min_amp, "Negative Hump", fontsize=10, verticalalignment='top', color='blue')
plt.text(mid_time, mid_amp, "Middle Hump", fontsize=10, verticalalignment='bottom', color='purple')
plt.text(max_time, max_amp, "Positive Hump", fontsize=10, verticalalignment='bottom', color='red')

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.title(f"Single Neuron Spike Analysis (P={prob:.2f})")
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(0.1)  # Another small delay
plt.show()

# Print Summary Table of Spikes
spike_df = pd.DataFrame(spike_features, columns=["Index", "Neg. Hump Amp", "Mid Hump Amp", "Pos. Hump Amp", 
                                                 "Neg. Hump Time", "Mid Hump Time", "Pos. Hump Time", "Spike Probability"])

print(spike_df.head(10))  # Print first 10 spikes