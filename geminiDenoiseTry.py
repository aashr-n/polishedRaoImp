# Enhanced Stimulation Pulse Artifact Detection
# Building on Kristin Sellers' 2018 artifact rejection protocol
# Adds comprehensive individual pulse detection and template matching

import mne
import scipy
import os
import numpy as np
from scipy.signal import welch, find_peaks, correlate
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

from scipy.interpolate import CubicSpline

# GUI imports
import tkinter as tk
from tkinter import filedialog
import tkinter.simpledialog as simpledialog
from mne.time_frequency import psd_array_multitaper

# --- Utility functions ---

def select_fif_file():
    """Select FIF file using GUI dialog"""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select EEG FIF file",
        filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
    )
    root.destroy()
    if not path:
        raise SystemExit("No file selected.")
    return path

def compute_mean_psd(data, sfreq, bandwidth=0.5):
    """Compute mean PSD across channels"""
    psd_list = []
    for idx in range(data.shape[0]):
        total_ch = data.shape[0]
        print(f"Channel {idx+1}/{total_ch}'s PSD is being calculated")
        psd_ch, freqs = psd_array_multitaper(
            data[idx:idx+1], sfreq=sfreq, fmin=1.0, fmax=sfreq/2,
            bandwidth=bandwidth, adaptive=False, low_bias=True,
            normalization='full', verbose=False
        )
        psd_list.append(psd_ch[0])
    psds = np.vstack(psd_list)
    return psds.mean(axis=0), freqs

def find_stim_frequency(mean_psd, freqs, prominence=20, min_freq=0):
    """Find stimulation frequency from PSD peaks"""
    # Find peaks with absolute prominence threshold
    peaks, props = find_peaks(mean_psd, prominence=prominence)
    if len(peaks) == 0:
        raise ValueError(f"No peaks found with prominence ≥ {prominence}")

    # Get peak frequencies and prominences
    pfreqs = freqs[peaks]
    proms = props['prominences']
    heights = mean_psd[peaks]

    # Compute relative prominence (prominence normalized by peak height)
    rel_proms = proms / heights

    # Exclude peaks below the minimum frequency
    mask_freq = pfreqs >= min_freq
    pfreqs = pfreqs[mask_freq]
    proms = proms[mask_freq]
    rel_proms = rel_proms[mask_freq]
    
    if len(pfreqs) == 0:
        raise ValueError(f"No peaks found above {min_freq} Hz with prominence ≥ {prominence}")

    # Print each peak with absolute and relative prominence
    print(f"Peaks ≥ {min_freq} Hz with abs prom ≥ {prominence}:")
    for f, p, rp in zip(pfreqs, proms, rel_proms):
        print(f"  {f:.2f} Hz → abs prom {p:.4f}, rel prom {rp:.4f}")

    # Select the lowest-frequency peak with relative prominence ≥ 0.5
    mask_rel = rel_proms >= 0.5
    if not np.any(mask_rel):
        # Fallback: use all peaks if none exceed threshold
        mask_rel = np.ones_like(rel_proms, dtype=bool)
    
    valid_freqs = pfreqs[mask_rel]
    stim_freq = np.min(valid_freqs)
    return stim_freq

# --- NEW FUNCTIONS for template detection and pulse finding ---

def detect_stim_epochs(signal, sfreq, stim_freq):
    """
    Detect stimulation epochs using sliding window PSD analysis
    Returns: stim_start_time, stim_end_time
    """
    # Dynamic window sizing based on sample rate
    win_sec = max(0.2, round(sfreq/5000, 1))  # Minimum 0.2s window
    step_sec = win_sec / 2
    nperseg = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)
    
    segment_centers = []
    segment_power = []

    # Slide window through the signal
    for start in range(0, signal.size - nperseg + 1, step_samps):
        stop = start + nperseg
        segment = signal[start:stop]
        
        # Use Welch's method for more robust PSD estimation
        freqs_w, psd_w = welch(segment, fs=sfreq, nperseg=min(nperseg, len(segment)))
        
        # Find power at the nearest frequency bin to stim_freq
        idx = np.argmin(np.abs(freqs_w - stim_freq))
        segment_centers.append((start + stop) / 2 / sfreq)
        
        # Normalize stim-frequency power by total power in this window
        total_power = np.sum(psd_w)
        if total_power > 0:
            rel_power = (psd_w[idx] / total_power) * 100
        else:
            rel_power = 0
        segment_power.append(rel_power)

    segment_power = np.array(segment_power)
    segment_centers = np.array(segment_centers)

    if len(segment_power) == 0:
        raise ValueError("No segments could be processed")

    # Find maximum prominence value
    max_prom_val = segment_power.max()
    if max_prom_val == 0:
        raise ValueError("No stimulation activity detected")
        
    max_idxs = np.where(segment_power == max_prom_val)[0]
    first_max, last_max = max_idxs.min(), max_idxs.max()

    # Define a drop fraction to detect sharp drop-off
    drop_frac = 0.1
    drop_thresh = drop_frac * max_prom_val

    # Expand left from first_max until relative prominence falls below drop_thresh
    start_idx = first_max
    while start_idx > 0 and segment_power[start_idx] >= drop_thresh:
        start_idx -= 1
    stim_start_time = segment_centers[start_idx]

    # Expand right from last_max until relative prominence falls below drop_thresh
    end_idx = last_max
    while end_idx < len(segment_power) - 1 and segment_power[end_idx] >= drop_thresh:
        end_idx += 1
    stim_end_time = segment_centers[end_idx]

    return stim_start_time, stim_end_time

def find_template_pulse(signal, sfreq, stim_freq, stim_start_time, stim_end_time):
    """
    Find the best template pulse within the stimulation period
    Returns: template, template_start_idx, template_duration
    """
    # Convert times to sample indices
    start_idx = int(stim_start_time * sfreq)
    end_idx = int(stim_end_time * sfreq)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    
    if start_idx >= end_idx:
        raise ValueError("Invalid stimulation period indices")
        
    stim_signal = signal[start_idx:end_idx]
    
    # Expected period in samples
    period_samples = int(sfreq / stim_freq)
    
    # Template duration: make it about 80% of the period to capture the pulse
    template_duration = int(0.8 * period_samples)
    
    if template_duration >= len(stim_signal):
        template_duration = len(stim_signal) // 2
    
    # Find the strongest artifact in the first few periods
    search_length = min(5 * period_samples, len(stim_signal) - template_duration)
    
    if search_length <= 0:
        raise ValueError("Stimulation signal too short for template extraction")
    
    best_template = None
    best_score = 0
    best_start = 0
    
    # Search for the template with highest variance (artifacts have high variance)
    step_size = max(1, period_samples // 4)  # Step by quarter periods
    for i in range(0, search_length, step_size):
        if i + template_duration >= len(stim_signal):
            break
            
        candidate = stim_signal[i:i + template_duration]
        
        # Score the candidate (your existing scoring)
        if len(candidate) > 0:
            variance_score = np.var(candidate)
            pp_score = np.ptp(candidate)
            combined_score = variance_score * pp_score
            
            if combined_score > best_score:
                # NEW: Find the main peak in this candidate
                peak_idx = np.argmax(np.abs(candidate))
                
                # NEW: Center template around this peak
                half_template = template_duration // 2
                peak_global_idx = i + peak_idx
                
                centered_start = max(0, peak_global_idx - half_template)
                centered_end = min(len(stim_signal), peak_global_idx + half_template)
                
                # Make sure we have enough data
                if centered_end - centered_start >= template_duration * 0.8:
                    best_score = combined_score
                    best_template = stim_signal[centered_start:centered_end].copy()
                    best_start = centered_start  # Update to use centered position

    
    if best_template is None:
        raise ValueError("Could not find suitable template pulse")
    
    template_start_idx = start_idx + best_start
    
    print(f"Template found at sample {template_start_idx} ({template_start_idx/sfreq:.3f}s)")
    print(f"Template duration: {template_duration} samples ({template_duration/sfreq:.3f}s)")
    
    return best_template, template_start_idx, template_duration

def cross_correlate_pulses(signal, template, sfreq, stim_freq, stim_start_time, stim_end_time):
    """
    Use cross-correlation to find all pulses similar to the template
    Returns: pulse_starts, pulse_ends, correlation_scores
    """
    # Convert times to sample indices
    start_idx = int(stim_start_time * sfreq)
    end_idx = int(stim_end_time * sfreq)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    
    stim_signal = signal[start_idx:end_idx]
    
    if len(stim_signal) == 0 or len(template) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Normalize template for better correlation
    template_std = np.std(template)
    if template_std > 0:
        template_norm = (template - np.mean(template)) / template_std
    else:
        template_norm = template - np.mean(template)
    
    # Cross-correlation
    correlation = correlate(stim_signal, template_norm, mode='valid')
    
    if len(correlation) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Expected period and minimum distance between pulses
    period_samples = int(sfreq / stim_freq)
    min_distance = max(1, int(0.7 * period_samples))  # Minimum 70% of period between peaks
    
    # Find peaks in correlation with minimum distance constraint
    correlation_threshold = np.percentile(correlation, 75)  # Top 25% of correlations
    peaks, properties = find_peaks(correlation, 
                                 height=correlation_threshold,
                                 distance=min_distance)
    
    # Convert correlation peak indices to signal indices
    pulse_starts = peaks + start_idx  # Add offset back to original signal indexing
    pulse_ends = pulse_starts + len(template)
    correlation_scores = correlation[peaks]
    
    print(f"Found {len(pulse_starts)} pulse artifacts via cross-correlation")
    
    return pulse_starts, pulse_ends, correlation_scores

def refine_pulse_boundaries(signal, pulse_starts, pulse_ends, sfreq):
    """
    Refine pulse start and end times using gradient-based detection
    Returns: refined_starts, refined_ends
    """
    if len(pulse_starts) == 0:
        return np.array([]), np.array([])
        
    refined_starts = []
    refined_ends = []
    
    for start, end in zip(pulse_starts, pulse_ends):
        # Extract pulse region with some padding
        padding = int(0.01 * sfreq)  # 10ms padding
        region_start = max(0, start - padding)
        region_end = min(len(signal), end + padding)
        region = signal[region_start:region_end]
        
        if len(region) < 3:  # Need at least 3 points for gradient
            refined_starts.append(start)
            refined_ends.append(end)
            continue
        
        # Calculate gradient to find sharp transitions
        gradient = np.gradient(region)
        abs_gradient = np.abs(gradient)
        
        # Smooth the gradient to reduce noise
        if len(abs_gradient) > 2:
            smoothed_gradient = gaussian_filter1d(abs_gradient, sigma=2)
        else:
            smoothed_gradient = abs_gradient
        
        # Find gradient peaks (sharp transitions)
        if len(smoothed_gradient) > 0:
            threshold = np.percentile(smoothed_gradient, 70)
            grad_peaks, _ = find_peaks(smoothed_gradient, height=threshold)
            
            if len(grad_peaks) >= 2:
                # First significant gradient peak as start, last as end
                pulse_start_refined = region_start + grad_peaks[0]
                pulse_end_refined = region_start + grad_peaks[-1]
            else:
                # Fallback to original boundaries
                pulse_start_refined = start
                pulse_end_refined = end
        else:
            pulse_start_refined = start
            pulse_end_refined = end
        
        refined_starts.append(pulse_start_refined)
        refined_ends.append(pulse_end_refined)
    
    return np.array(refined_starts), np.array(refined_ends)

def visualize_pulse_detection(signal, sfreq, template, template_start_idx, 
                            pulse_starts, pulse_ends, stim_start_time, stim_end_time):
    """
    Create comprehensive visualization of pulse detection results
    """
    times = np.arange(len(signal)) / sfreq
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 1. Full signal overview with stimulation period
    axes[0].plot(times, signal, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axvspan(stim_start_time, stim_end_time, color='red', alpha=0.2, label='Stim Period')
    axes[0].set_title('Full Signal with Stimulation Period')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Template pulse
    template_times = np.arange(len(template)) / sfreq
    axes[1].plot(template_times, template, 'g-', linewidth=2)
    axes[1].set_title(f'Template Pulse (from {template_start_idx/sfreq:.3f}s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Stimulation period with detected pulses
    stim_start_idx = int(stim_start_time * sfreq)
    stim_end_idx = int(stim_end_time * sfreq)
    stim_times = times[stim_start_idx:stim_end_idx]
    stim_signal = signal[stim_start_idx:stim_end_idx]
    
    axes[2].plot(stim_times, stim_signal, 'b-', alpha=0.7, linewidth=0.8)
    
    # Mark all detected pulses
    if len(pulse_starts) > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(pulse_starts)))
        for i, (start, end, color) in enumerate(zip(pulse_starts, pulse_ends, colors)):
            start_time = start / sfreq
            end_time = end / sfreq
            axes[2].axvspan(start_time, end_time, color=color, alpha=0.3)
            # Add pulse number
            mid_time = (start_time + end_time) / 2
            if len(stim_signal) > 0:
                axes[2].text(mid_time, np.max(stim_signal) * 0.9, str(i+1), 
                            ha='center', va='center', fontsize=8, 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    axes[2].set_title(f'Detected Pulses in Stimulation Period ({len(pulse_starts)} pulses)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Pulse timing analysis
    if len(pulse_starts) > 1:
        pulse_intervals = np.diff(pulse_starts) / sfreq
        pulse_times = pulse_starts[:-1] / sfreq
        
        axes[3].plot(pulse_times, pulse_intervals, 'ro-', markersize=4)
        axes[3].axhline(y=np.mean(pulse_intervals), color='g', linestyle='--', 
                       label=f'Mean: {np.mean(pulse_intervals):.4f}s')
        axes[3].set_title('Inter-Pulse Intervals')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Interval (s)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'Insufficient pulses for interval analysis', 
                    ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Inter-Pulse Intervals - Not Available')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== PULSE DETECTION SUMMARY ===")
    print(f"Total pulses detected: {len(pulse_starts)}")
    print(f"Stimulation period: {stim_start_time:.3f}s to {stim_end_time:.3f}s ({stim_end_time-stim_start_time:.3f}s)")
    
    if len(pulse_starts) > 1:
        pulse_intervals = np.diff(pulse_starts) / sfreq
        print(f"Mean inter-pulse interval: {np.mean(pulse_intervals):.4f}s ± {np.std(pulse_intervals):.4f}s")
        stim_freq = 1 / np.mean(pulse_intervals) if np.mean(pulse_intervals) > 0 else 0
        print(f"Estimated frequency from intervals: {stim_freq:.2f} Hz")
        
    if len(pulse_starts) > 0:
        pulse_durations = (pulse_ends - pulse_starts) / sfreq
        print(f"Pulse duration range: {np.min(pulse_durations):.4f}s to {np.max(pulse_durations):.4f}s")




def spline_artifact_extended_anchors(data, artifact_starts, artifact_ends, sfreq, buffer_ms=5.0):
    """
    Replaces identified artifacts using cubic spline interpolation,
    attempting to anchor the spline on data *outside* the artifact for
    better smoothing. Uses a standard cubic fit between anchor points.

    Args:
        data (np.ndarray): Can be 1D (single channel: samples) 
                           or 2D (channels x samples).
        artifact_starts (np.ndarray): 1D array of sample indices where 
                                      artifacts start (0-indexed).
        artifact_ends (np.ndarray): 1D array of sample indices of where 
                                    artifacts end (0-indexed). Must be the 
                                    same length as artifact_starts.
        sfreq (float): Sampling frequency of the data in Hz.
        buffer_ms (float): Duration in milliseconds to look before artifact start
                           and after artifact end for spline anchor points.
    
    Returns:
        np.ndarray: Data with artifact segments replaced by spline interpolation.
                    Shape will match input data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input 'data' must be a NumPy array.")
    # ... (add other input validation as in previous versions if needed) ...
    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")

    is_1d_input = False
    if data.ndim == 1:
        is_1d_input = True
        spline_data = data.reshape(1, -1).copy()
        original_data_for_check = data.reshape(1, -1) # For NaN check
    elif data.ndim == 2:
        spline_data = data.copy()
        original_data_for_check = data
    else:
        raise ValueError("Input 'data' must be a 1D or 2D NumPy array.")

    num_channels, num_samples = spline_data.shape
    buffer_samples = int(buffer_ms / 1000.0 * sfreq)
    print(f"Using buffer_samples: {buffer_samples} (from {buffer_ms}ms at {sfreq}Hz)")


    for i_chan in range(num_channels):
        print(f"Spline interpolation (extended anchors): Channel {i_chan + 1}/{num_channels}")

        if num_samples > 0 and np.isnan(original_data_for_check[i_chan, 0]):
            print(f"  Channel {i_chan + 1}: First sample is NaN, skipping.")
            continue
        if len(artifact_starts) == 0:
            continue
            
        for i_stim in range(len(artifact_starts)):
            start_sample = int(artifact_starts[i_stim])
            end_sample = int(artifact_ends[i_stim])

            if not (0 <= start_sample < num_samples and \
                    0 <= end_sample < num_samples and \
                    start_sample <= end_sample):
                print(f"  Warning: Art {i_stim+1} (Ch {i_chan+1}) invalid bounds ({start_sample}-{end_sample}). Skipping.")
                continue
            
            if start_sample == end_sample: 
                continue 

            anchor1_idx_ideal = start_sample - buffer_samples
            anchor2_idx_ideal = end_sample + buffer_samples
            anchor1_idx_clipped = max(0, anchor1_idx_ideal)
            anchor2_idx_clipped = min(num_samples - 1, anchor2_idx_ideal)
            
            use_extended_anchors = False
            if buffer_samples > 0:
                if (anchor1_idx_clipped < start_sample and \
                    anchor2_idx_clipped > end_sample and \
                    anchor1_idx_clipped < anchor2_idx_clipped): # Key conditions
                    use_extended_anchors = True
            
            if use_extended_anchors:
                x_known = np.array([anchor1_idx_clipped, anchor2_idx_clipped])
            else:
                x_known = np.array([start_sample, end_sample])
                if buffer_samples > 0: 
                    print(
                        f"  Info: Art {i_stim+1} (Ch {i_chan+1}, {start_sample}-{end_sample}): "
                        f"Fallback to artifact boundaries. Cannot use extended anchors (buf={buffer_samples}samp, "
                        f"ideal: {anchor1_idx_ideal}-{anchor2_idx_ideal}, "
                        f"clipped: {anchor1_idx_clipped}-{anchor2_idx_clipped})."
                    )
            
            if x_known[0] >= x_known[1]:
                 print(f"  Warning: Art {i_stim+1} (Ch {i_chan+1}): Anchors ({x_known[0]},{x_known[1]}) not increasing. Skipping.")
                 continue
            
            y_known = spline_data[i_chan, x_known]

            if np.any(np.isnan(y_known)):
                print(f"  Warning: Art {i_stim+1} (Ch {i_chan+1}) NaN y_known ({y_known}) at x_known={x_known}. Skipping.")
                continue

            x_query = np.arange(start_sample, end_sample + 1) 
            if len(x_query) == 0: continue

            try:
                # Default bc_type is 'not-a-knot', which fits a cubic polynomial
                cs = CubicSpline(x_known, y_known) 
                interpolated_values = cs(x_query)
                spline_data[i_chan, start_sample : end_sample + 1] = interpolated_values
            except ValueError as e:
                print(
                    f"  Error: Art {i_stim+1} (Ch {i_chan+1}) spline error ({start_sample}-{end_sample}): {e}. "
                    f"x_known={x_known}, y_known={y_known}. Skipping."
                )
                continue

    if is_1d_input:
        return spline_data.squeeze()
    else:
        return spline_data


def main():
    
    '''# 1) Load data
    #path = select_fif_file()
    path = select_fif_file()
    raw = mne.io.read_raw_fif(path, preload=True)
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    avg_sig = data.mean(axis=0)

    
    # 2) PSD & stim frequency
    #mean_psd, freqs = compute_mean_psd(channel_data, sfreq)
    mean_REAL_psd, freqs = compute_mean_psd(data, sfreq)
    stim_freq = find_stim_frequency(mean_REAL_psd, freqs)
    print(f"Estimated stim frequency: {stim_freq:.2f} Hz")

    # 3) Artifact detection
    initial_starts = detect_artifacts(avg_sig, sfreq, stim_freq)
    starts, ends, template, mf_out = template_match_starts(
        avg_sig, initial_starts, sfreq, stim_freq
    )
    print(f"Detected {len(starts)} artifact pulses")

    # 2) Choose channel with largest artifact response (after artifact detection)
    # Compute sum of absolute raw amplitudes at artifact starts for each channel
    responses = [np.sum(np.abs(data[ch_idx, starts])) for ch_idx in range(data.shape[0])]
    best_idx = int(np.argmax(responses))
    best_ch = raw.ch_names[best_idx]
    print(f"Selected channel for plotting: {best_ch}. Index: {best_idx}")

    # Selected channel for plotting: POL R ACC1-Ref. Index: 8
    '''
    """ 
    Main pipeline for comprehensive stimulation pulse detection
    """
    try:
        # 1) Load data - UPDATE THIS PATH
        path = '/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw.fif'
        # Uncomment the line below to use file selection dialog
        # path = select_fif_file()
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            print("Using file selection dialog...")
            path = select_fif_file()
        
        raw = mne.io.read_raw_fif(path, preload=True)
        sfreq = raw.info['sfreq']
        print(f"Sample frequency: {sfreq} Hz")

        data = raw.get_data()
        
        # Use specific channel (modify as needed)
        if data.shape[0] <= 8:
            channel_idx = 0
            print(f"Using channel 0 (only {data.shape[0]} channels available)")
        else:
            channel_idx = 8
            print(f"Using channel {channel_idx}")
            
        channel_data = data[channel_idx:channel_idx+1, :]
        signal = channel_data[0]  # Extract 1D signal

        # 2) Compute PSD and find stimulation frequency
        clearest_psd, freqs = compute_mean_psd(channel_data, sfreq)
        stim_freq = find_stim_frequency(clearest_psd, freqs, prominence=5, min_freq=1)
        print(f"Estimated stimulation frequency: {stim_freq:.2f} Hz")

        # 3) Detect stimulation epochs
        stim_start_time, stim_end_time = detect_stim_epochs(signal, sfreq, stim_freq)
        print(f"Stimulation period: {stim_start_time:.3f}s to {stim_end_time:.3f}s")

        # 4) Find template pulse
        template, template_start_idx, template_duration = find_template_pulse(
            signal, sfreq, stim_freq, stim_start_time, stim_end_time)

        # 5) Find all pulses using cross-correlation
        pulse_starts, pulse_ends, correlation_scores = cross_correlate_pulses(
            signal, template, sfreq, stim_freq, stim_start_time, stim_end_time)

        # 6) Refine pulse boundarie
        pulse_starts_refined, pulse_ends_refined = refine_pulse_boundaries(
            signal, pulse_starts, pulse_ends, sfreq)

        # 7) Comprehensive visualization
        visualize_pulse_detection(signal, sfreq, template, template_start_idx,
                                pulse_starts_refined, pulse_ends_refined, 
                                stim_start_time, stim_end_time)
        
        # ... (previous code in main, including visualize_pulse_detection)

        # 8) Apply Cubic Spline Interpolation to all channels
        print(f"\nApplying Cubic Spline Interpolation...")
        print(f"Using refined pulse boundaries from channel {channel_idx} for all channels.")
        print(f"Number of pulses to interpolate: {len(pulse_starts_refined)}")

        # The 'data' variable holds all channel data (channels x samples)
        # pulse_starts_refined and pulse_ends_refined are from the single channel analysis ('signal')
        # These artifact times will be applied to all channels in 'data' by the spline function.
        # You can adjust buffer_ms as needed.
        corrected_data_all_channels = spline_artifact_extended_anchors(
            data,  # Full data array (n_channels, n_samples)
            pulse_starts_refined,
            pulse_ends_refined,
            sfreq,
            buffer_ms=5.0  # Buffer in milliseconds for anchor points
        )
        print("Cubic spline interpolation complete.")

        # Store the corrected data in the results dictionary if you're using it
        if 'results' not in locals(): # or however you manage your results dictionary
            results = {}
        results['corrected_data_all_channels'] = corrected_data_all_channels
        if data.ndim > 1 and channel_idx < data.shape[0]:
            results['corrected_data_channel_selected'] = corrected_data_all_channels[channel_idx]
        elif data.ndim == 1:
             results['corrected_data_channel_selected'] = corrected_data_all_channels



        # 9) Visualize the effect of spline interpolation on the selected channel
        print(f"Visualizing spline interpolation result for channel {channel_idx}...")

        if data.ndim > 1:
            original_signal_selected_channel = data[channel_idx]
            # Ensure corrected_data_all_channels is not None and has the channel
            if corrected_data_all_channels is not None and channel_idx < corrected_data_all_channels.shape[0]:
                 corrected_signal_selected_channel = corrected_data_all_channels[channel_idx]
            else:
                print(f"Warning: Could not retrieve corrected data for channel {channel_idx}. Skipping visualization.")
                corrected_signal_selected_channel = None # or handle error
        elif data.ndim == 1: # If data was 1D to begin with
            original_signal_selected_channel = data
            corrected_signal_selected_channel = corrected_data_all_channels
        else:
            print("Error: Data has unexpected dimensions. Skipping visualization.")
            original_signal_selected_channel = None
            corrected_signal_selected_channel = None

        if original_signal_selected_channel is not None and corrected_signal_selected_channel is not None:
            num_samples_viz = len(original_signal_selected_channel)
            times_viz = np.arange(num_samples_viz) / sfreq

            plt.figure(figsize=(18, 8)) # Adjusted for potentially more detail

            # Define a window for plotting, e.g., around the stimulation period
            # Use a slightly wider window than just stim_start_time to stim_end_time for context
            # Handle cases where stim_start_time or stim_end_time might be None
            plot_window_padding = 0.5 # seconds
            if stim_start_time is not None and stim_end_time is not None :
                vis_start_time = stim_start_time - plot_window_padding
                vis_end_time = stim_end_time + plot_window_padding
            elif len(pulse_starts_refined) > 0: # Fallback to first/last pulse
                vis_start_time = (pulse_starts_refined[0] / sfreq) - plot_window_padding
                vis_end_time = (pulse_ends_refined[-1] / sfreq) + plot_window_padding
            else: # Fallback to full signal if no other info
                vis_start_time = times_viz[0]
                vis_end_time = times_viz[-1]


            plot_start_idx = max(0, int(vis_start_time * sfreq))
            plot_end_idx = min(num_samples_viz, int(vis_end_time * sfreq))

            if plot_start_idx >= plot_end_idx and num_samples_viz > 0: # If range is invalid, plot a default portion
                plot_start_idx = 0
                plot_end_idx = min(num_samples_viz, int(sfreq * 10)) # e.g., first 10 seconds or whole signal

            # Plot original signal in the window
            plt.plot(times_viz[plot_start_idx:plot_end_idx],
                     original_signal_selected_channel[plot_start_idx:plot_end_idx],
                     label=f'Original Channel {channel_idx} (Ch Index {raw.ch_names[channel_idx] if isinstance(raw, mne.io.BaseRaw) and channel_idx < len(raw.ch_names) else channel_idx})',
                     color='gray', alpha=0.6, linewidth=1.0)

            # Plot corrected signal in the window
            plt.plot(times_viz[plot_start_idx:plot_end_idx],
                     corrected_signal_selected_channel[plot_start_idx:plot_end_idx],
                     label=f'Corrected Channel {channel_idx} (Spline)', color='blue', linewidth=1.2, alpha=0.8)

            # Highlight the interpolated segments on the corrected signal
            print(f"Highlighting {len(pulse_starts_refined)} interpolated segments for channel {channel_idx} in visualization.")
            first_interp_label = True
            for i in range(len(pulse_starts_refined)):
                start_sample = pulse_starts_refined[i]
                end_sample = pulse_ends_refined[i] # This is the last sample *of* the artifact

                # Determine the part of the pulse that is within the current plot window
                seg_display_start = max(plot_start_idx, start_sample)
                seg_display_end = min(plot_end_idx, end_sample + 1) # +1 for Python slicing up to end_sample

                if seg_display_start < seg_display_end: # If the segment is visible in the plot window
                    times_segment = times_viz[seg_display_start:seg_display_end]
                    data_segment = corrected_signal_selected_channel[seg_display_start:seg_display_end]
                    
                    if len(times_segment) > 0 and len(data_segment) > 0:
                         plt.plot(times_segment, data_segment, color='red', linewidth=1.5,
                                 label='Interpolated Segment' if first_interp_label else "",
                                 zorder=5) # zorder to draw on top
                         first_interp_label = False


            plt.title(f'Cubic Spline Interpolation for Channel {channel_idx} ({raw.ch_names[channel_idx] if isinstance(raw, mne.io.BaseRaw) and channel_idx < len(raw.ch_names) else ""})', fontsize=14)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            if plot_start_idx < plot_end_idx and len(times_viz) > max(plot_start_idx, plot_end_idx-1) :
                plt.xlim(times_viz[plot_start_idx], times_viz[plot_end_idx-1])
                plt.legend(loc='upper right')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Skipping visualization for channel {channel_idx} due to missing data.")

            # ... (rest of your main function, e.g., updating and returning the results dictionary)
            if 'results' in locals() and original_signal_selected_channel is not None:
                results['original_signal_selected_channel'] = original_signal_selected_channel
            # The corrected data for selected channel and all channels are already added to results earlier

        correctedData = spline_artifact_extended_anchors(data, pulse_starts_refined, pulse_ends_refined, sfreq)
        # Assuming 'correctedData' is your (n_channels, n_samples) array
        # and 'channel_idx' is the index of the channel you want to plot.
        # 'sfreq' is your sampling frequency.

        # Extract the single channel data as a 1D array
        single_channel_waveform = correctedData[channel_idx] # or correctedData[channel_idx, :]

        # Create a time vector (optional, but good for correct x-axis)
        # num_samples = single_channel_waveform.shape[0]
        # times = np.arange(num_samples) / sfreq

        plt.figure(figsize=(12, 4)) # Optional: adjust figure size

        # Use plt.plot() for a line plot
        plt.plot(single_channel_waveform)
        # If you have 'times':
        # plt.plot(times, single_channel_waveform)

        plt.title(f"Corrected Waveform for Channel {channel_idx}")
        plt.xlabel("Sample Number") # Or "Time (s)" if using the 'times' vector
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        # 8) Return results for further analysis
        results = {
            'signal': signal,
            'sfreq': sfreq,
            'stim_freq': stim_freq,
            'stim_start_time': stim_start_time,
            'stim_end_time': stim_end_time,
            'template': template,
            'pulse_starts': pulse_starts_refined,
            'pulse_ends': pulse_ends_refined,
            'pulse_times': pulse_starts_refined / sfreq,
            'pulse_durations': (pulse_ends_refined - pulse_starts_refined) / sfreq,
            'correlation_scores': correlation_scores
        }
        
        return results
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    
    if results is not None:
        # Additional analysis can be performed here with the results dictionary
        print(f"\nFirst 10 pulse start times: {results['pulse_times'][:10]}")
        print(f"First 10 pulse durations: {results['pulse_durations'][:10]}")
    else:
        print("Pipeline failed to complete successfully.")