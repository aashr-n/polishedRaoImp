#!/usr/bin/env python3
"""
EEG Stimulation Artifact Detection and Analysis
Based on Kristin Sellers' 2018 artifact rejection protocol
Fixed and improved version with proper structure and error handling
"""

import os
import numpy as np
import scipy.signal as sig
from scipy.signal import welch, square, find_peaks, correlate
from scipy.stats import zscore
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import psd_array_multitaper
from mne.filter import filter_data

# Optional GUI imports (wrapped in try/except for headless environments)
try:
    import tkinter as tk
    from tkinter import filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Warning: GUI libraries not available. File selection will be manual.")


class StimArtifactDetector:
    """
    A class to detect and analyze electrical stimulation artifacts in EEG data.
    """
    
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.raw = None
        self.data = None
        self.sfreq = None
        self.stim_freq = None
        self.stim_start_time = None
        self.stim_end_time = None
        self.selected_channel = None
        
    def select_file(self):
        """Select EEG file using GUI or manual input."""
        if GUI_AVAILABLE and self.filepath is None:
            root = tk.Tk()
            root.withdraw()
            self.filepath = filedialog.askopenfilename(
                title="Select EEG FIF file",
                filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
            )
            if not self.filepath:
                raise SystemExit("No file selected.")
        elif self.filepath is None:
            raise ValueError("No filepath provided and GUI not available.")
            
    def load_data(self):
        """Load EEG data from FIF file."""
        if self.filepath is None:
            self.select_file()
            
        print(f"Loading data from: {self.filepath}")
        self.raw = mne.io.read_raw_fif(self.filepath, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.data = self.raw.get_data()
        print(f"Sample frequency: {self.sfreq} Hz")
        print(f"Data shape: {self.data.shape}")
        
    def compute_channel_psd(self, channel_idx, bandwidth=0.5):
        """Compute PSD for a specific channel."""
        channel_data = self.data[channel_idx:channel_idx+1, :]
        psd, freqs = psd_array_multitaper(
            channel_data, sfreq=self.sfreq, fmin=1.0, fmax=self.sfreq/2,
            bandwidth=bandwidth, adaptive=False, low_bias=True,
            normalization='full', verbose=False
        )
        return psd[0], freqs
        
    def compute_mean_psd(self, channels=None, bandwidth=0.5):
        """Compute mean PSD across channels."""
        if channels is None:
            channels = range(self.data.shape[0])
            
        psd_list = []
        for idx in channels:
            print(f"Computing PSD for channel {idx+1}/{len(channels)}")
            psd_ch, freqs = self.compute_channel_psd(idx, bandwidth)
            psd_list.append(psd_ch)
            
        psds = np.vstack(psd_list)
        return psds.mean(axis=0), freqs
        
    def find_stimulation_frequency(self, psd, freqs, prominence=20, min_freq=0):
        """Find stimulation frequency from PSD peaks."""
        # Find peaks with absolute prominence threshold
        peaks, props = find_peaks(psd, prominence=prominence)
        if len(peaks) == 0:
            raise ValueError(f"No peaks found with prominence ≥ {prominence}")

        # Get peak frequencies and properties
        peak_freqs = freqs[peaks]
        peak_proms = props['prominences']
        peak_heights = psd[peaks]

        # Compute relative prominence (prominence normalized by peak height)
        rel_proms = peak_proms / peak_heights

        # Filter by minimum frequency
        mask_freq = peak_freqs >= min_freq
        peak_freqs = peak_freqs[mask_freq]
        peak_proms = peak_proms[mask_freq]
        rel_proms = rel_proms[mask_freq]
        
        if len(peak_freqs) == 0:
            raise ValueError(f"No peaks found above {min_freq} Hz")

        # Print peak information
        print(f"Peaks ≥ {min_freq} Hz with prominence ≥ {prominence}:")
        for f, p, rp in zip(peak_freqs, peak_proms, rel_proms):
            print(f"  {f:.2f} Hz → abs prom {p:.4f}, rel prom {rp:.4f}")

        # Select the lowest-frequency peak with relative prominence ≥ 0.5
        mask_rel = rel_proms >= 0.5
        if not np.any(mask_rel):
            print("Warning: No peaks with relative prominence ≥ 0.5, using all peaks")
            mask_rel = np.ones_like(rel_proms, dtype=bool)
            
        valid_freqs = peak_freqs[mask_rel]
        stim_freq = np.min(valid_freqs)
        return stim_freq
        
    def detect_stimulation_period(self, channel_idx, win_sec=0.2, step_sec=0.1, 
                                power_threshold=75):
        """Detect stimulation period using sliding window analysis."""
        if self.stim_freq is None:
            raise ValueError("Stimulation frequency not determined yet")
            
        signal = self.data[channel_idx, :]
        times = np.arange(signal.size) / self.sfreq
        
        # Window parameters
        nperseg = int(win_sec * self.sfreq)
        step_samps = int(step_sec * self.sfreq)
        
        segment_centers = []
        segment_power = []

        # Sliding window analysis
        for start in range(0, signal.size - nperseg + 1, step_samps):
            stop = start + nperseg
            
            # Compute PSD for this window
            psd_w, freqs_w = psd_array_multitaper(
                signal[start:stop][None, :], sfreq=self.sfreq,
                fmin=0, fmax=self.sfreq/2, bandwidth=self.stim_freq/5,
                adaptive=True, low_bias=True, normalization='full', verbose=False
            )
            psd_w = psd_w[0]
            
            # Find power at stimulation frequency
            freq_idx = np.argmin(np.abs(freqs_w - self.stim_freq))
            segment_centers.append((start + stop) / 2 / self.sfreq)
            
            # Normalize by total power
            rel_power = (psd_w[freq_idx] / np.sum(psd_w)) * 100
            segment_power.append(rel_power)

        segment_power = np.array(segment_power)
        segment_centers = np.array(segment_centers)

        # Find high-power segments
        thresh_power = np.percentile(segment_power, power_threshold)
        high_idx = segment_power >= thresh_power
        
        if not np.any(high_idx):
            print(f"Warning: No segments above {power_threshold}th percentile")
            return times, signal, segment_centers, segment_power
            
        # Determine stimulation period boundaries
        max_power = segment_power.max()
        max_indices = np.where(segment_power == max_power)[0]
        
        # Expand around maximum power regions
        drop_thresh = 0.1 * max_power
        start_idx = max_indices.min()
        end_idx = max_indices.max()
        
        # Expand leftward
        while start_idx > 0 and segment_power[start_idx] >= drop_thresh:
            start_idx -= 1
            
        # Expand rightward  
        while end_idx < len(segment_power) - 1 and segment_power[end_idx] >= drop_thresh:
            end_idx += 1
            
        self.stim_start_time = segment_centers[start_idx]
        self.stim_end_time = segment_centers[end_idx]
        
        print(f"Detected stimulation period: {self.stim_start_time:.2f}s to {self.stim_end_time:.2f}s")
        
        return times, signal, segment_centers, segment_power
        
    def compute_hilbert_envelope(self, signal, filter_bandwidth=1.5):
        """Compute Hilbert envelope of filtered signal."""
        # Bandpass filter around stimulation frequency
        f_lo = max(self.stim_freq - filter_bandwidth, 0.1)
        f_hi = self.stim_freq + filter_bandwidth
        
        filtered_sig = filter_data(
            signal.copy(), self.sfreq, l_freq=f_lo, h_freq=f_hi,
            method='iir', verbose=False
        )
        
        # Compute analytic signal and envelope
        analytic = sig.hilbert(filtered_sig)
        envelope = np.abs(analytic)
        
        return envelope, filtered_sig
        
    def detect_stimulation_onset(self, envelope, z_threshold=4.0, min_duration_sec=None):
        """Detect sustained stimulation onset using Hilbert envelope."""
        if min_duration_sec is None:
            min_duration_sec = max(3.0 / self.stim_freq, 0.10)
            
        min_dur_samples = int(np.ceil(min_duration_sec * self.sfreq))
        
        # Z-score the envelope
        z_env = zscore(envelope)
        above = z_env > z_threshold
        
        # Find contiguous above-threshold segments
        idx = np.where(above)[0]
        stim_onset = None
        
        if idx.size:
            # Split into contiguous chunks
            chunks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            
            # Find first chunk meeting duration requirement
            for chunk in chunks:
                if chunk.size >= min_dur_samples:
                    stim_onset = int(chunk[0])
                    break
                    
        return stim_onset
        
    def template_matching(self, signal):
        """Perform template matching to find stimulation pulses."""
        # Create template (one period of square wave)
        template_len = int(round(self.sfreq / self.stim_freq))
        template = np.zeros(template_len)
        template[:template_len // 2] = 1
        template -= template.mean()  # Zero-mean
        template /= np.linalg.norm(template)  # Normalize
        
        # Cross-correlation
        correlation = correlate(signal, template, mode='valid')
        peak_idx = np.argmax(correlation)
        peak_time = (peak_idx + template_len // 2) / self.sfreq
        
        return peak_time, correlation
        
    def plot_results(self, times, signal, segment_centers=None, segment_power=None, 
                    envelope=None, onset_time=None, peak_time=None):
        """Plot analysis results."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Original signal with stimulation period
        axes[0].plot(times, signal, 'k-', label='EEG Signal')
        if self.stim_start_time and self.stim_end_time:
            axes[0].axvspan(self.stim_start_time, self.stim_end_time, 
                           color='green', alpha=0.3, label='Stimulation Period')
        if onset_time:
            axes[0].axvline(onset_time, color='red', linestyle='--', 
                           linewidth=2, label='Hilbert Onset')
        if peak_time:
            axes[0].axvline(peak_time, color='blue', linestyle='--', 
                           linewidth=2, label='Template Match')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (µV)')
        axes[0].set_title('EEG Signal with Stimulation Detection')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Power analysis
        if segment_centers is not None and segment_power is not None:
            axes[1].plot(segment_centers, segment_power, 'b-', linewidth=2)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Relative Power (%)')
            axes[1].set_title(f'Stimulation Frequency Power ({self.stim_freq:.1f} Hz)')
            axes[1].grid(True, alpha=0.3)
            
        # Plot 3: Hilbert envelope
        if envelope is not None:
            axes[2].plot(times, envelope, 'purple', linewidth=2)
            if onset_time:
                axes[2].axvline(onset_time, color='red', linestyle='--', 
                               linewidth=2, label='Onset Detection')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Envelope Amplitude')
            axes[2].set_title('Hilbert Envelope')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def run_analysis(self, channel_idx=None, filepath=None):
        """Run complete stimulation artifact analysis."""
        if filepath:
            self.filepath = filepath
            
        # Load data
        if self.raw is None:
            self.load_data()
            
        # Auto-select channel if not specified (use channel with highest variance)
        if channel_idx is None:
            channel_vars = np.var(self.data, axis=1)
            channel_idx = np.argmax(channel_vars)
            print(f"Auto-selected channel {channel_idx} (highest variance)")
            
        self.selected_channel = channel_idx
        
        # Find stimulation frequency
        print("Computing PSD for stimulation frequency detection...")
        psd, freqs = self.compute_channel_psd(channel_idx)
        self.stim_freq = self.find_stimulation_frequency(psd, freqs)
        print(f"Detected stimulation frequency: {self.stim_freq:.2f} Hz")
        
        # Detect stimulation period
        print("Detecting stimulation period...")
        times, signal, seg_centers, seg_power = self.detect_stimulation_period(channel_idx)
        
        # Hilbert analysis
        print("Computing Hilbert envelope...")
        envelope, filtered_signal = self.compute_hilbert_envelope(signal)
        
        # Detect onset
        print("Detecting stimulation onset...")
        onset_idx = self.detect_stimulation_onset(envelope)
        onset_time = times[onset_idx] if onset_idx is not None else None
        
        if onset_time:
            print(f"Stimulation onset detected at: {onset_time:.3f} s")
        else:
            print("No sustained stimulation onset detected")
            
        # Template matching
        print("Performing template matching...")
        peak_time, correlation = self.template_matching(signal)
        print(f"Template match peak at: {peak_time:.6f} s")
        
        # Plot results
        self.plot_results(times, signal, seg_centers, seg_power, 
                         envelope, onset_time, peak_time)
        
        return {
            'stim_frequency': self.stim_freq,
            'stim_start_time': self.stim_start_time,
            'stim_end_time': self.stim_end_time,
            'onset_time': onset_time,
            'template_peak_time': peak_time,
            'selected_channel': channel_idx
        }


def main():
    """Main function to run the analysis."""
    # Example usage - modify path as needed
    detector = StimArtifactDetector()
    
    # You can specify a file path directly or use GUI selection
    # detector = StimArtifactDetector('/path/to/your/file.fif')
    
    try:
        results = detector.run_analysis()
        print("\nAnalysis Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()