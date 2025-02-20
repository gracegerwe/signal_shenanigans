import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pywt
import os
import pandas as pd
import matplotlib.cm as cm
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.decomposition import PCA

# Define folder path
folder_path = "/Users/gracegerwe/Documents/Neuralink Raw Data"

# Store all filtered amplitudes before setting threshold
all_filtered_data = []
all_spike_counts = []
all_isi = []
all_spike_features = []
filtered_data_dict = {}
file_count = 0  

# Bandpass Filter
def bandpass_filter(data, lowcut=10, highcut=300, sr=19531, order=4):
    nyquist = sr / 2
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Notch Filter at 60Hz and harmonics
def notch_filter(data, freq=60, sr=19531, quality_factor=30):
    nyquist = sr / 2
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data)

# Wavelet Denoising
def wavelet_denoise(data, wavelet='db4', level=2):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    coeffs[1:] = [pywt.threshold(c, np.std(c), mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")

# Step 1: Loop through all files to collect filtered data (for global threshold)
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)

        # Load the WAV file
        sr, data = wav.read(file_path)

        # Remove DC bias
        data_centered = data - np.mean(data)

        # Bandpass Filter
        filtered_data = bandpass_filter(data_centered, lowcut=10, highcut=300, sr=sr)

        # Notch Filter at 60Hz and harmonics
        for notch_freq in [60, 120, 180]:
            filtered_data = notch_filter(filtered_data, freq=notch_freq, sr=sr)

        # Wavelet Denoising
        filtered_data = wavelet_denoise(filtered_data)

        # Store filtered data for later use
        filtered_data_dict[filename] = filtered_data

        # Apply Median Filter for Motion Artifacts
        filtered_data = medfilt(filtered_data, kernel_size=5)

        # Store filtered data for global thresholding
        all_filtered_data.extend(filtered_data)

        file_count += 1
        if file_count % 50 == 0:
            print(f"Filtered {file_count} files so far...")

# Final confirmation that all files have been filtered
print(f"I have filtered all {file_count} files!")

# Step 2: Compute a **global threshold** based on all files
global_threshold = np.percentile(np.abs(all_filtered_data), 99)
print(f"Global threshold set at: {global_threshold:.2f}")

file_count = 0  # Reset file counter for spike detection
for filename in sorted(filtered_data_dict.keys()):
    if filename.endswith(".wav"):
        # Retrieve precomputed filtered data
        filtered_data = filtered_data_dict[filename]

        # Detect spikes using **global threshold**
        spike_indices = np.where(np.abs(filtered_data) > global_threshold)[0]

        # Apply refractory period (10ms)
        min_spike_distance = int(0.01 * sr)
        if len(spike_indices) > 0:
            spike_times = spike_indices[np.insert(np.diff(spike_indices) > min_spike_distance, 0, True)]
        else:
            spike_times = np.array([])  # Ensure it's an empty array instead of causing an error

        # Compute ISI
        isi = np.diff(spike_times) / sr
        all_isi.extend(isi)
        all_spike_counts.append(len(spike_times))

        # Extract spike features
        window_size = int(0.002 * sr)  # 2ms window (~40 samples at 20kHz)
        half_window = window_size // 2

        for spike in spike_times:
            if spike - half_window > 0 and spike + half_window < len(filtered_data):
                spike_waveform = filtered_data[spike - half_window: spike + half_window]

                # Find spike features
                min_amp = np.min(spike_waveform)
                max_amp = np.max(spike_waveform)
                mid_amp = spike_waveform[len(spike_waveform) // 2]

                min_time = (np.argmin(spike_waveform) - half_window) / sr * 1000
                max_time = (np.argmax(spike_waveform) - half_window) / sr * 1000
                mid_time = 0

                # Estimate probability
                noise_std = np.std(filtered_data)
                snr = (max_amp - min_amp) / (2 * noise_std)
                spike_prob = 1 / (1 + np.exp(-snr))

                all_spike_features.append([spike, min_amp, mid_amp, max_amp, min_time, mid_time, max_time, spike_prob])

        # Print progress every 50 files
        file_count += 1
        if file_count % 50 == 0:
            print(f"Processed spike detection for {file_count} files...")

# Final confirmation that all files have been processed for spikes
print(f"Spike detection completed for all {file_count} files!")

# Convert spike feature list to DataFrame
spike_df = pd.DataFrame(
    all_spike_features,
    columns=["Index", "Neg. Hump Amp", "Mid Hump Amp", "Pos. Hump Amp",
             "Neg. Hump Time", "Mid Hump Time", "Pos. Hump Time", "Spike Probability"]
)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot aggregated results
axs[0, 0].plot(range(len(all_spike_counts)), all_spike_counts, marker='o', linestyle='-', color='b')
axs[0, 0].set_xlabel("File Index")
axs[0, 0].set_ylabel("Spike Count")
axs[0, 0].set_title("Spike Counts Across All Files")
axs[0, 0].grid(True)

# Plot ISI Distribution
axs[0, 1].hist(isi, bins=50, edgecolor='black')
axs[0, 1].set_xlabel("Inter-Spike Interval (s)")
axs[0, 1].set_ylabel("Count")
axs[0, 1].set_title("ISI Distribution (Neural Spike Timing)")
axs[0, 1].grid(True)

axs[1, 0].scatter(spike_df["Index"], spike_df["Neg. Hump Amp"], color='r', label="Neg. Hump Amp", alpha=0.7, s=10)
axs[1, 0].scatter(spike_df["Index"], spike_df["Pos. Hump Amp"], color='g', label="Pos. Hump Amp", alpha=0.7, s=10)
axs[1, 0].set_ylabel("Amplitude (μV)")
axs[1, 0].set_title("Spike Amplitudes Over Time")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].scatter(spike_df["Index"], spike_df["Spike Probability"], color='b', label="Spike Probability", alpha=0.7, s=10)
axs[1, 1].set_ylabel("Spike Probability")
axs[1, 1].set_title("Spike Probability Over Time")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()