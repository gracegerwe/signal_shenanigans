# Neural Data Signal Processing and Spike Sorting Pipeline

## Overview

This repository contains code for analyzing neural recordings from Neuralink's raw electrode data. The pipeline provides a workflow for filtering, denoising, spike detection, feature extraction and spike sorting from multi-channel recordings.

## Features

- **Robust signal filtering** using bandpass, notch, and wavelet denoising techniques
- **Advanced spike detection** with physiological constraints and feature validation
- **Spike sorting** using feature extraction and clustering algorithms
- **Neuron classification** into known electrophysiological cell types
- **Comprehensive visualization** of spikes, waveforms, and firing statistics

## Prerequisites

```
numpy
matplotlib
scipy
pywt (PyWavelets)
pandas
scikit-learn
```

## Signal Processing Pipeline

### 1. Preprocessing

The pipeline applies a series of filters to clean the raw neural recordings:

- **DC Bias Removal**: Centers the signal around zero.
- **Bandpass Filtering (300-3000 Hz)**: Isolates the frequency range of action potentials ([`bandpass_filter`](process_neural_data.py#L29)).
- **Notch Filtering (60Hz, 120Hz, 180Hz)**: Removes power line interference ([`notch_filter`](process_neural_data.py#L38)).
- **Wavelet Denoising**: Applies soft thresholding to wavelet coefficients ([`wavelet_denoise`](process_neural_data.py#L47)).
- **Median Filtering**: Removes motion artifacts

### 2. Spike Detection

The algorithm ([`detect_spikes`](process_neural_data.py#L59)) identifies spikes based on the following criteria:

- Peak amplitude exceeding a data-derived threshold
- Appropriate waveform shape (rapid depolarization and repolarization)
- Physiologically plausible spike width (0.2-2ms)
- Minimum refractory period between spikes (2ms)

### 3. Feature Extraction

For each detected spike, the pipeline extracts:

- **Amplitude features**: Negative peak, middle point, positive peak
- **Timing features**: Time points of key waveform components
- **Quality metrics**: SNR and spike probability

### 4. Spike Sorting

The pipeline ([`sort_spikes`](process_neural_data.py#L160)) uses K-means clustering with feature standardization to identify different neuron types:

- **Fast-Spiking Interneurons**: Narrow spikes with quick repolarization
- **Regular-Spiking Pyramidal Neurons**: Medium-width spikes with pronounced after-hyperpolarization
- **Burst-Spiking Neurons**: Wide spikes with complex waveform shape

- **Visualization:**
  - Plots **raw and filtered neural signals**.
  - Displays **spike waveforms** and **ISI distributions**.
  - Visualizes **neuron classification results**.

## Usage

1. Set the `folder_path` variable to point to your directory containing Neuralink WAV files
2. Run the script to process all files and generate visualizations
3. Examine the console output for statistics on detected spikes and neuron classification

## Visualizations

The pipeline generates several visualizations:

- **Spike waveform overlay**: Shows the consistency and shape of detected spikes
- **Spike counts across recordings**: Reveals changes in neural activity
- **ISI distribution**: Provides insight into firing patterns
- **Multi-channel visualization**: Compares activity across different electrodes

## Analysis Output

For each neuron class, the pipeline outputs:

- Number of spikes
- Average spike width
- Average amplitude characteristics
- Average spike probability
- Overall SNR and quality metrics

## **Example Output**

### **Spike Waveforms (Detected Action Potentials)**



### **Neuron Classification (K-Means Clustering)**



### **ISI Distribution (Neural Firing Rate Analysis)**

## Possible Extensions

- Implement more advanced spike sorting algorithms (e.g., GMM, template matching, t-SNE/UMAP visualization)
- Identify single neurons and track their drift across multiple recording sessions (ISI fingerprints, bust pattern recognition, stability analysis)
- Add connectivity analysis via cross-correlation and coherence measures between channels

## Acknowledgements

This project uses data from the Neuralink Compression Challenge.
