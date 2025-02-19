import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pywt
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.decomposition import PCA

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

# **🔹 Merging Plots for Easier Analysis**
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Trim filtered_data to match time_axis length
filtered_data = filtered_data[:len(time_axis)]

# **Plot 1: Processed Signal with Spikes**
axs[0, 0].plot(time_axis, filtered_data, lw=0.5, label="Filtered Signal")
axs[0, 0].scatter(spike_times / sr, filtered_data[spike_times], color='red', label="Detected Spikes")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].set_title(f"Neural Spikes Detected: {num_spikes}")
axs[0, 0].legend()
axs[0, 0].grid(True)

# **Plot 2: ISI Distribution**
axs[0, 1].hist(isi, bins=50, edgecolor='black')
axs[0, 1].set_xlabel("Inter-Spike Interval (s)")
axs[0, 1].set_ylabel("Count")
axs[0, 1].set_title("ISI Distribution (Neural Spike Timing)")
axs[0, 1].grid(True)

# **Plot 3: FFT of PCA-Processed Spikes**
axs[1, 0].plot(freqs_pca[:1000], fft_pca_spike[:1000])  # First 1000 bins
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Magnitude")
axs[1, 0].set_title("FFT of PCA-Processed Neural Spikes")
axs[1, 0].grid(True)

# **Plot 4: PCA Processed Signal**
axs[1, 1].plot(np.linspace(0, len(pca_signal) / sr, num=len(pca_signal)), pca_signal, lw=0.5)
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Amplitude")
axs[1, 1].set_title("Post PCA Signal")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
