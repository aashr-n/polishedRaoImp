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

def main():
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

        # 6) Refine pulse boundaries
        pulse_starts_refined, pulse_ends_refined = refine_pulse_boundaries(
            signal, pulse_starts, pulse_ends, sfreq)

        # 7) Comprehensive visualization
        visualize_pulse_detection(signal, sfreq, template, template_start_idx,
                                pulse_starts_refined, pulse_ends_refined, 
                                stim_start_time, stim_end_time)
        
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