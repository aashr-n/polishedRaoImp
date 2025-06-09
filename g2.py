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
    start_idx_epoch = first_max # Use a different variable name to avoid confusion with sample indices
    while start_idx_epoch > 0 and segment_power[start_idx_epoch] >= drop_thresh:
        start_idx_epoch -= 1
    stim_start_time = segment_centers[start_idx_epoch]

    # Expand right from last_max until relative prominence falls below drop_thresh
    end_idx_epoch = last_max
    while end_idx_epoch < len(segment_power) - 1 and segment_power[end_idx_epoch] >= drop_thresh:
        end_idx_epoch += 1
    stim_end_time = segment_centers[end_idx_epoch]

    return stim_start_time, stim_end_time

def find_template_pulse(signal, sfreq, stim_freq, stim_start_time, stim_end_time):
    """
    Find the best template pulse within the stimulation period.
    The template width will be exactly 1/stim_freq.
    Returns: template, template_start_idx (first sample of template), template_length (in samples)
    """
    # Convert times to sample indices
    start_idx_stim_period = int(stim_start_time * sfreq)
    end_idx_stim_period = int(stim_end_time * sfreq)

    # Ensure indices are within bounds
    start_idx_stim_period = max(0, start_idx_stim_period)
    end_idx_stim_period = min(len(signal), end_idx_stim_period)

    if start_idx_stim_period >= end_idx_stim_period:
        raise ValueError("Invalid stimulation period indices")

    stim_signal = signal[start_idx_stim_period:end_idx_stim_period]

    # Expected period in samples (this is the desired template duration)
    period_samples = int(sfreq / stim_freq)
    if period_samples <= 0:
        raise ValueError(f"period_samples must be positive. Got {period_samples} from sfreq={sfreq}, stim_freq={stim_freq}")

    template_duration_samples = period_samples #MODIFIED

    if template_duration_samples > len(stim_signal): #MODIFIED
        # This case should ideally not happen if stim_period is long enough
        # If it does, it means stim_signal is shorter than one period.
        # We might need to take the whole stim_signal or error out.
        print(f"Warning: stim_signal length ({len(stim_signal)}) is shorter than template_duration_samples ({template_duration_samples}). Using full stim_signal as template.")
        template_duration_samples = len(stim_signal)
        if template_duration_samples == 0:
            raise ValueError("Stimulation signal is empty, cannot extract template.")


    # Find the strongest artifact in the first few periods of the stim_signal
    # Search length should be enough to find a representative pulse, but not exceed stim_signal length
    search_length = min(5 * period_samples, len(stim_signal) - template_duration_samples + 1) # Ensure we can extract a full template

    if search_length <= 0:
        # This implies len(stim_signal) < template_duration_samples
        # which should be handled by the previous check. If not, it's an issue.
        if len(stim_signal) >= template_duration_samples: # If somehow it gets here
             best_template = stim_signal[:template_duration_samples].copy()
             best_start_in_stim_signal = 0
             print("Warning: search_length for template was <=0, but stim_signal was long enough. Using first part.")
        else: # stim_signal is too short.
            raise ValueError(f"Stimulation signal (length {len(stim_signal)}) is too short for template extraction of duration {template_duration_samples}.")


    best_template = None
    best_score = -np.inf # Initialize with a very small number for scoring
    best_start_in_stim_signal = 0

    # Search for the template with highest variance (artifacts have high variance)
    step_size = max(1, period_samples // 10 if period_samples > 10 else 1)  # Smaller step for finer search

    for i in range(0, search_length, step_size):
        # Ensure the candidate template does not exceed stim_signal bounds
        if i + template_duration_samples > len(stim_signal):
            break

        candidate = stim_signal[i : i + template_duration_samples]

        if len(candidate) > 0:
            # Score based on variance and peak-to-peak amplitude
            variance_score = np.var(candidate)
            pp_score = np.ptp(candidate) # Peak-to-peak
            combined_score = variance_score * pp_score # Example metric, could be tuned

            if combined_score > best_score:
                best_score = combined_score
                # The centering logic could be kept if desired, but the overall template
                # duration is fixed. For simplicity with fixed duration, we can take the found segment.
                # If centering is used, ensure it respects template_duration_samples.
                # For now, let's use the segment that gave the best score directly.
                best_template = candidate.copy()
                best_start_in_stim_signal = i

    if best_template is None:
        # Fallback: if no template found by scoring (e.g., search_length was too small or signal too flat)
        # and stim_signal is long enough, take the first possible template.
        if len(stim_signal) >= template_duration_samples:
            print("Warning: Could not find suitable template pulse by scoring, using first segment of stim_signal.")
            best_template = stim_signal[0:template_duration_samples].copy()
            best_start_in_stim_signal = 0
        else:
            raise ValueError("Could not find suitable template pulse.")

    # template_start_idx is relative to the original full signal
    template_start_idx_abs = start_idx_stim_period + best_start_in_stim_signal

    print(f"Template found starting at sample {template_start_idx_abs} ({template_start_idx_abs/sfreq:.3f}s)")
    print(f"Template duration: {template_duration_samples} samples ({template_duration_samples/sfreq:.3f}s)")

    return best_template, template_start_idx_abs, template_duration_samples


def cross_correlate_pulses(signal, template, sfreq, stim_freq, stim_start_time, stim_end_time):
    """
    Use cross-correlation to find all pulses similar to the template.
    pulse_ends will be the inclusive last sample index of the pulse.
    Returns: pulse_starts, pulse_ends, correlation_scores
    """
    # Convert times to sample indices
    start_idx_corr = int(stim_start_time * sfreq) # Search within the detected stimulation period
    end_idx_corr = int(stim_end_time * sfreq)

    # Ensure indices are within bounds
    start_idx_corr = max(0, start_idx_corr)
    end_idx_corr = min(len(signal), end_idx_corr)

    if start_idx_corr >= end_idx_corr or (end_idx_corr - start_idx_corr) < len(template):
        print("Warning: Stimulation signal segment for correlation is too short or invalid.")
        return np.array([]), np.array([]), np.array([])

    stim_signal_corr = signal[start_idx_corr:end_idx_corr]

    if len(stim_signal_corr) == 0 or len(template) == 0:
        return np.array([]), np.array([]), np.array([])

    # Normalize template for robust correlation
    template_std = np.std(template)
    if template_std > 1e-9: # Avoid division by zero or very small std
        template_norm = (template - np.mean(template)) / template_std
    else:
        template_norm = template - np.mean(template) # Only mean center if std is ~0

    # Cross-correlation
    # Mode 'valid' means correlation is computed only where signals fully overlap
    correlation = correlate(stim_signal_corr, template_norm, mode='valid')

    if len(correlation) == 0:
        return np.array([]), np.array([]), np.array([])

    # Expected period and minimum distance between pulse starts
    period_samples = int(sfreq / stim_freq)
    min_distance_peaks = max(1, int(0.7 * period_samples))  # Min 70% of period between peak starts

    # Find peaks in correlation signal
    # Threshold can be tuned. Using a percentile is more adaptive.
    correlation_threshold = np.percentile(correlation, 85) # E.g., top 15% of correlation values
    peaks_in_corr, properties = find_peaks(correlation,
                                 height=correlation_threshold,
                                 distance=min_distance_peaks)

    # Convert correlation peak indices back to original signal indices
    # The correlation result is shorter than stim_signal_corr by len(template) - 1
    # A peak at index `p` in `correlation` corresponds to the template starting at `p` in `stim_signal_corr`.
    pulse_starts_abs = peaks_in_corr + start_idx_corr  # Add original offset of stim_signal_corr

    # Define pulse_ends as inclusive last sample index
    # len(template) is period_samples
    pulse_ends_abs = pulse_starts_abs + len(template) - 1 # MODIFIED: inclusive end

    correlation_scores = correlation[peaks_in_corr]

    print(f"Found {len(pulse_starts_abs)} pulse artifacts via cross-correlation")

    return pulse_starts_abs, pulse_ends_abs, correlation_scores


def refine_pulse_boundaries(signal, pulse_starts, initial_pulse_ends, sfreq, stim_freq):
    """
    Refine pulse start times using gradient-based detection.
    Pulse end times are then set to ensure each pulse has a duration of period_samples
    (i.e., width 1/stim_freq).
    'pulse_ends' and 'initial_pulse_ends' refer to the *last sample index* of the pulse.
    Returns: refined_starts, refined_ends (both inclusive indices)
    """
    if len(pulse_starts) == 0:
        return np.array([]), np.array([])

    refined_starts_list = []
    period_samples = int(sfreq / stim_freq) # Number of samples in one pulse period
    
    if period_samples <= 0:
        print(f"Warning: period_samples is {period_samples}. Check sfreq ({sfreq}) and stim_freq ({stim_freq}). Cannot refine boundaries.")
        return pulse_starts, initial_pulse_ends # Fallback

    for i in range(len(pulse_starts)):
        current_start_estimate = pulse_starts[i]
        
        # Define a search window for the gradient around the current_start_estimate.
        # Search from 20% of period before to 20% of period after the estimate.
        search_padding = int(0.2 * period_samples)
        
        region_start_for_grad_search = max(0, current_start_estimate - search_padding)
        # Ensure the search region for the start doesn't extend too far into where the peak might be
        region_end_for_grad_search = min(len(signal), current_start_estimate + search_padding) # exclusive end for slice

        refined_start_candidate = current_start_estimate # Default to original estimate

        if region_start_for_grad_search < region_end_for_grad_search:
            region = signal[region_start_for_grad_search:region_end_for_grad_search] # Slice
            
            if len(region) >= 3: # Need at least 3 points for np.gradient
                gradient = np.gradient(region)
                abs_gradient = np.abs(gradient) # Magnitude of change

                # Smooth the gradient to reduce noise sensitivity
                sigma_gradient_smooth = max(1, int(0.02 * period_samples)) # Smooth based on period
                # Ensure gaussian_filter1d has enough data points relative to sigma
                if len(abs_gradient) > 2 * sigma_gradient_smooth + 1 :
                    smoothed_gradient = gaussian_filter1d(abs_gradient, sigma=sigma_gradient_smooth)
                else:
                    smoothed_gradient = abs_gradient # Not enough points to smooth reliably
                
                if len(smoothed_gradient) > 0:
                    # Assume pulse start is characterized by the steepest slope (max gradient)
                    # in this local window.
                    peak_grad_idx_in_region = np.argmax(smoothed_gradient)
                    refined_start_candidate = region_start_for_grad_search + peak_grad_idx_in_region
        
        refined_starts_list.append(refined_start_candidate)

    final_refined_starts = np.array(refined_starts_list, dtype=int)
    
    # Calculate refined_ends (inclusive last sample index) based on these refined_starts
    # and the fixed period_samples.
    # s_last = s_first + N_samples - 1
    final_refined_ends = final_refined_starts + period_samples - 1 # MODIFIED

    # Ensure pulse boundaries are within signal limits
    # Start indices must be less than signal length
    final_refined_starts = np.clip(final_refined_starts, 0, len(signal) - 1)
    # End indices must also be less than signal length and should be >= start indices
    final_refined_ends = np.clip(final_refined_ends, 0, len(signal) - 1)

    # Filter out invalid pulses:
    # - where start index is after end index (can happen due to clipping or extreme refinement)
    # - where start or end indices are at the very edge making the pulse too short (implicitly handled by start <= end)
    valid_indices = (final_refined_starts <= final_refined_ends) & \
                    (final_refined_starts < len(signal)-1) # Ensure start is not the last sample if period > 1
    
    # Additionally, ensure that the duration is at least 1 sample after clipping.
    # (final_refined_ends - final_refined_starts + 1) should ideally be period_samples.
    # Clipping might shorten it. For now, start <= end is the main check.

    final_refined_starts = final_refined_starts[valid_indices]
    final_refined_ends = final_refined_ends[valid_indices]
    
    return final_refined_starts, final_refined_ends


def visualize_pulse_detection(signal, sfreq, stim_freq, template, template_start_idx,
                            pulse_starts, pulse_ends, stim_start_time, stim_end_time): # Added stim_freq
    """
    Create comprehensive visualization of pulse detection results.
    pulse_starts and pulse_ends are inclusive sample indices.
    """
    times = np.arange(len(signal)) / sfreq
    period_samples = int(sfreq / stim_freq) # Nominal period in samples

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=False) # Allow different x-axes

    # 1. Full signal overview with stimulation period
    axes[0].plot(times, signal, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axvspan(stim_start_time, stim_end_time, color='red', alpha=0.2, label=f'Stim Period ({stim_start_time:.2f}s - {stim_end_time:.2f}s)')
    axes[0].set_title('Full Signal with Stimulation Period')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    # Zoom to a relevant portion if signal is too long
    if stim_end_time > 0 and (stim_end_time - stim_start_time) < 0.3 * times[-1] : # If stim period is a smaller portion
        plot_start_time_ax0 = max(0, stim_start_time - 0.1*(stim_end_time - stim_start_time))
        plot_end_time_ax0 = min(times[-1], stim_end_time + 0.1*(stim_end_time - stim_start_time))
        axes[0].set_xlim(plot_start_time_ax0, plot_end_time_ax0)


    # 2. Template pulse
    template_times = np.arange(len(template)) / sfreq
    axes[1].plot(template_times, template, 'g-', linewidth=2)
    axes[1].set_title(f'Template Pulse (from {template_start_idx/sfreq:.3f}s, duration {len(template)/sfreq:.4f}s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # 3. Stimulation period with detected pulses
    stim_start_idx_viz = int(stim_start_time * sfreq)
    stim_end_idx_viz = int(stim_end_time * sfreq)
    # Add some padding for visualization if possible
    padding_viz_samples = int(0.1 * (stim_end_idx_viz - stim_start_idx_viz))
    stim_start_idx_viz = max(0, stim_start_idx_viz - padding_viz_samples)
    stim_end_idx_viz = min(len(signal), stim_end_idx_viz + padding_viz_samples)

    if stim_start_idx_viz < stim_end_idx_viz:
        stim_times_viz = times[stim_start_idx_viz:stim_end_idx_viz]
        stim_signal_viz = signal[stim_start_idx_viz:stim_end_idx_viz]
        axes[2].plot(stim_times_viz, stim_signal_viz, 'b-', alpha=0.7, linewidth=0.8)
    else: # Fallback if stim period is too short or invalid
        axes[2].plot(times, signal, 'b-', alpha=0.7, linewidth=0.8) # Plot full signal
        stim_signal_viz = signal # for text y-pos

    # Mark all detected pulses
    if len(pulse_starts) > 0 and len(pulse_ends) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(pulse_starts))) # Changed colormap
        for i in range(len(pulse_starts)):
            start_sample = pulse_starts[i]
            end_sample = pulse_ends[i] # Inclusive end sample

            # For axvspan, xmax is exclusive end time
            start_time_vspan = start_sample / sfreq
            end_time_vspan = (end_sample + 1) / sfreq # MODIFIED for inclusive end_sample

            axes[2].axvspan(start_time_vspan, end_time_vspan, color=colors[i], alpha=0.35)
            # Add pulse number
            mid_time_text = (start_sample + 0.5 * (end_sample - start_sample)) / sfreq
            # Position text: find y-value in the original signal at mid_time_text
            # Ensure index is valid
            text_y_idx = int(mid_time_text * sfreq)
            if 0 <= text_y_idx < len(signal):
                text_y_val = signal[text_y_idx]
            elif len(stim_signal_viz) > 0 and 0 <= (text_y_idx - stim_start_idx_viz) < len(stim_signal_viz): # If within zoomed stim_signal_viz
                 text_y_val = stim_signal_viz[text_y_idx - stim_start_idx_viz]
            elif len(stim_signal_viz) > 0 : # Fallback to max of visible segment
                text_y_val = np.max(stim_signal_viz) * 0.9
            else: # Absolute fallback
                text_y_val = 0

            axes[2].text(mid_time_text, text_y_val, str(i+1),
                        ha='center', va='bottom', fontsize=7, color='black',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, lw=0.5))

    axes[2].set_title(f'Detected Pulses in Stimulation Period ({len(pulse_starts)} pulses)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    if stim_start_idx_viz < stim_end_idx_viz: # Set xlim only if viz segment is valid
        axes[2].set_xlim(stim_times_viz[0], stim_times_viz[-1])
    axes[2].grid(True, alpha=0.3)

    # 4. Pulse timing analysis
    if len(pulse_starts) > 1:
        pulse_start_times_sec = pulse_starts / sfreq
        inter_pulse_intervals_sec = np.diff(pulse_start_times_sec) # Time between starts of consecutive pulses

        axes[3].plot(pulse_start_times_sec[:-1], inter_pulse_intervals_sec, 'ro-', markersize=4, linewidth=1)
        mean_interval = np.mean(inter_pulse_intervals_sec)
        axes[3].axhline(y=mean_interval, color='g', linestyle='--',
                       label=f'Mean Interval: {mean_interval:.4f}s')
        axes[3].axhline(y=1/stim_freq, color='purple', linestyle=':',
                       label=f'Nominal Interval (1/stim_freq): {1/stim_freq:.4f}s')
        axes[3].set_title('Inter-Pulse Intervals (Start-to-Start)')
        axes[3].set_xlabel('Time of First Pulse in Pair (s)')
        axes[3].set_ylabel('Interval (s)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        if stim_start_idx_viz < stim_end_idx_viz: # Align x-axis with plot above if possible
             axes[3].set_xlim(stim_times_viz[0], stim_times_viz[-1])

    else:
        axes[3].text(0.5, 0.5, 'Insufficient pulses for interval analysis (<2)',
                    ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Inter-Pulse Intervals - Not Available')

    plt.tight_layout(pad=1.5) # Add some padding
    plt.show()

    # Print summary statistics
    print("\n=== PULSE DETECTION SUMMARY ===")
    print(f"Total pulses detected: {len(pulse_starts)}")
    print(f"Stimulation period: {stim_start_time:.3f}s to {stim_end_time:.3f}s (Duration: {stim_end_time-stim_start_time:.3f}s)")
    print(f"Nominal pulse width (1/stim_freq): {1/stim_freq:.4f}s ({period_samples} samples at {sfreq} Hz)")

    if len(pulse_starts) > 1:
        # Inter-pulse intervals already calculated for plot
        print(f"Mean inter-pulse interval (start-to-start): {mean_interval:.4f}s ± {np.std(inter_pulse_intervals_sec):.4f}s")
        if mean_interval > 0:
            print(f"Estimated frequency from intervals: {1 / mean_interval:.2f} Hz (compare to input stim_freq: {stim_freq:.2f} Hz)")

    if len(pulse_starts) > 0 and len(pulse_ends) > 0:
        # pulse_ends is inclusive last sample index
        actual_pulse_durations_samples = (pulse_ends - pulse_starts) + 1
        actual_pulse_durations_sec = actual_pulse_durations_samples / sfreq
        print(f"Effective pulse duration range (after potential clipping at signal ends): {np.min(actual_pulse_durations_sec):.4f}s to {np.max(actual_pulse_durations_sec):.4f}s")
        # Check how many differ from nominal (period_samples)
        num_non_nominal_duration = np.sum(actual_pulse_durations_samples != period_samples)
        if num_non_nominal_duration > 0:
            print(f"  ({num_non_nominal_duration} pulses have durations not exactly {period_samples} samples, likely due to signal boundary clipping)")


def spline_artifact_extended_anchors(data, artifact_starts, artifact_ends, sfreq, buffer_ms=5.0):
    """
    Replaces identified artifacts using cubic spline interpolation,
    attempting to anchor the spline on data *outside* the artifact for
    better smoothing. Uses a standard cubic fit between anchor points.
    'artifact_starts' and 'artifact_ends' are inclusive sample indices.

    Args:
        data (np.ndarray): Can be 1D (single channel: samples)
                           or 2D (channels x samples).
        artifact_starts (np.ndarray): 1D array of sample indices where
                                      artifacts start (0-indexed, inclusive).
        artifact_ends (np.ndarray): 1D array of sample indices of where
                                    artifacts end (0-indexed, inclusive). Must be the
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
    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")
    if not (isinstance(artifact_starts, np.ndarray) and isinstance(artifact_ends, np.ndarray)):
        raise TypeError("'artifact_starts' and 'artifact_ends' must be NumPy arrays.")
    if artifact_starts.shape != artifact_ends.shape:
        raise ValueError("'artifact_starts' and 'artifact_ends' must have the same shape.")
    if artifact_starts.ndim != 1:
        raise ValueError("'artifact_starts' and 'artifact_ends' must be 1D arrays.")


    is_1d_input = False
    if data.ndim == 1:
        is_1d_input = True
        spline_data = data.reshape(1, -1).copy()
    elif data.ndim == 2:
        spline_data = data.copy()
    else:
        raise ValueError("Input 'data' must be a 1D or 2D NumPy array.")

    num_channels, num_samples = spline_data.shape
    buffer_samples = int(buffer_ms / 1000.0 * sfreq)
    print(f"Spline: Using buffer_samples: {buffer_samples} (from {buffer_ms}ms at {sfreq}Hz)")


    for i_chan in range(num_channels):
        print(f"Spline interpolation (extended anchors): Channel {i_chan + 1}/{num_channels}")

        if num_samples > 0 and np.isnan(spline_data[i_chan, 0]): # Check on spline_data (the copy)
            print(f"  Channel {i_chan + 1}: First sample is NaN, skipping (potential issue with data loading).")
            continue
        if len(artifact_starts) == 0:
            continue # No artifacts to interpolate

        for i_stim in range(len(artifact_starts)):
            # artifact_starts and artifact_ends are INCLUSIVE indices
            start_sample_artifact = int(artifact_starts[i_stim])
            end_sample_artifact = int(artifact_ends[i_stim])


            if not (0 <= start_sample_artifact < num_samples and \
                    0 <= end_sample_artifact < num_samples and \
                    start_sample_artifact <= end_sample_artifact): # end_sample can be num_samples-1
                print(f"  Warning: Art {i_stim+1} (Ch {i_chan+1}) invalid bounds ({start_sample_artifact}-{end_sample_artifact}). Skipping.")
                continue

            # Define anchor points for the spline
            # Anchor 1: before artifact_start. Anchor 2: after artifact_end.
            anchor1_idx_ideal = start_sample_artifact - buffer_samples
            anchor2_idx_ideal = end_sample_artifact + buffer_samples

            anchor1_idx_clipped = max(0, anchor1_idx_ideal)
            anchor2_idx_clipped = min(num_samples - 1, anchor2_idx_ideal) # num_samples-1 is last valid index

            use_extended_anchors = False
            if buffer_samples > 0:
                # Conditions for using extended anchors:
                # 1. Clipped anchor1 is actually *before* the artifact starts.
                # 2. Clipped anchor2 is actually *after* the artifact ends.
                # 3. Clipped anchor1 is before clipped anchor2 (ensures distinct points).
                if (anchor1_idx_clipped < start_sample_artifact and \
                    anchor2_idx_clipped > end_sample_artifact and \
                    anchor1_idx_clipped < anchor2_idx_clipped):
                    use_extended_anchors = True

            if use_extended_anchors:
                x_known = np.array([anchor1_idx_clipped, anchor2_idx_clipped])
            else:
                # Fallback to using the artifact boundaries themselves as anchors.
                # This happens if buffer_samples is 0, or if artifact is too close to signal ends.
                # Ensure start_sample_artifact and end_sample_artifact are not the same if used as anchors for a segment.
                # If they are the same (1-sample artifact), CubicSpline needs distinct points.
                # However, CubicSpline can handle x_known with two identical x if y are also identical (constant).
                # More robust: if start == end, maybe just copy value or skip interpolation for that point?
                # For a 1-sample artifact, interpolation means replacing that single point.
                # If start_sample_artifact == end_sample_artifact, x_known could be [start-1, end+1] if available.
                # Current logic: x_query is start_sample_artifact to end_sample_artifact+1.
                # If start=end, x_query has 1 point. CS requires at least 2 for non-trivial.
                # Let's ensure x_known has at least two distinct points if possible for CubicSpline.
                # The simplest fallback is artifact boundaries, but if start=end, CS might struggle.
                # However, the user wants to replace the segment from start_sample_artifact to end_sample_artifact.
                # If start_sample_artifact == end_sample_artifact, we are replacing one point.
                # CubicSpline needs at least 2 points. If buffer_ms=0, then anchors are start/end.
                # If start=end, then x_known = [start_sample_artifact, start_sample_artifact].
                # This will likely cause CubicSpline to fail if y values are different (which they won't be here).
                # It's better to ensure x_known has distinct points.
                
                # Safest fallback if extended anchors cannot be used:
                # Use points immediately outside the artifact, if available.
                # If not, use artifact boundaries but this is less ideal for interpolation shape.
                x_anchor_start = max(0, start_sample_artifact - 1)
                x_anchor_end = min(num_samples - 1, end_sample_artifact + 1)

                if x_anchor_start < start_sample_artifact and x_anchor_end > end_sample_artifact and x_anchor_start < x_anchor_end:
                    x_known = np.array([x_anchor_start, x_anchor_end])
                    if not use_extended_anchors and buffer_samples > 0: # Only print if extended was attempted
                         print(
                            f"  Info: Art {i_stim+1} (Ch {i_chan+1}, {start_sample_artifact}-{end_sample_artifact}): "
                            f"Fallback to immediate pre/post artifact for anchors."
                        )
                else: # True fallback to artifact boundaries, mostly for very short signals or artifacts at edges
                    x_known = np.array([start_sample_artifact, end_sample_artifact])
                    if not use_extended_anchors and buffer_samples > 0:
                        print(
                            f"  Info: Art {i_stim+1} (Ch {i_chan+1}, {start_sample_artifact}-{end_sample_artifact}): "
                            f"Fallback to artifact boundaries for anchors. Cannot use extended or immediate pre/post."
                        )


            if x_known[0] >= x_known[1] and not (x_known[0] == x_known[1] and start_sample_artifact == end_sample_artifact): # Allow x_known[0]==x_known[1] only if it's a single point artifact
                 print(f"  Warning: Art {i_stim+1} (Ch {i_chan+1}): Anchors ({x_known[0]},{x_known[1]}) not distinct or not increasing. Skipping.")
                 continue

            y_known = spline_data[i_chan, x_known]

            if np.any(np.isnan(y_known)):
                print(f"  Warning: Art {i_stim+1} (Ch {i_chan+1}) NaN in y_known ({y_known}) at x_known={x_known}. Skipping.")
                continue

            # x_query are the sample indices to be interpolated (inclusive start and end)
            if start_sample_artifact > end_sample_artifact : # Should have been caught earlier
                continue
            x_query = np.arange(start_sample_artifact, end_sample_artifact + 1) # +1 because arange is exclusive end

            if len(x_query) == 0: continue # Should not happen if start_sample_artifact <= end_sample_artifact

            try:
                # If x_known has only one unique point (e.g. x_known=[A,A]), CubicSpline will fail if y_known are different.
                # If x_known=[A,A] and y_known=[yA,yA], cs(A) = yA.
                # If len(x_query) is 1 (single point artifact at start_sample_artifact),
                # and x_known = [start_sample_artifact-1, start_sample_artifact+1], this is fine.
                if x_known[0] == x_known[1] and len(x_query) > 0 : # e.g. artifact at signal edge, only one anchor point possible
                    # Linear interpolation for this specific edge case or just copy anchor value.
                    # If only one anchor point, we can't do cubic. Fill with that anchor value.
                    # This happens if the artifact is at the very start/end and buffer makes anchors identical.
                    print(f"  Info: Art {i_stim+1} (Ch {i_chan+1}) single unique anchor {x_known[0]}. Filling with anchor value.")
                    interpolated_values = np.full(len(x_query), y_known[0])
                else:
                    # bc_type='not-a-knot' is default and generally good.
                    # Consider 'clamped' if derivative estimates are available, or 'natural'.
                    cs = CubicSpline(x_known, y_known, bc_type='not-a-knot')
                    interpolated_values = cs(x_query)

                # Replace data in the spline_data array
                # Slice is [start_inclusive : end_inclusive + 1] for assignment
                spline_data[i_chan, start_sample_artifact : end_sample_artifact + 1] = interpolated_values

            except ValueError as e:
                print(
                    f"  Error: Art {i_stim+1} (Ch {i_chan+1}) spline error ({start_sample_artifact}-{end_sample_artifact}): {e}. "
                    f"x_known={x_known}, y_known={y_known}, x_query_len={len(x_query)}. Skipping."
                )
                continue

    if is_1d_input:
        return spline_data.squeeze()
    else:
        return spline_data


def main():
    """
    Main pipeline for comprehensive stimulation pulse detection and artifact removal.
    """
    results = {} # Initialize results dictionary
    try:
        # 1) Load data
        default_path = '/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw.fif' # User specific path
        path = ''
        if os.path.exists(default_path):
             # Simple dialog to confirm or change path
            root_tk = tk.Tk()
            root_tk.withdraw() # Hide main window
            use_default = tk.messagebox.askyesno("Confirm File Path", f"Use default file path?\n{default_path}")
            if use_default:
                path = default_path
            else:
                path = select_fif_file()
            root_tk.destroy()
        else:
            print(f"Default file not found: {default_path}")
            path = select_fif_file()

        if not path:
            print("No file selected. Exiting.")
            return None

        raw = mne.io.read_raw_fif(path, preload=True)
        sfreq = raw.info['sfreq']
        print(f"Sample frequency: {sfreq} Hz")
        results['sfreq'] = sfreq

        data = raw.get_data() # All channels

        # Channel selection for analysis (e.g., PSD, template finding)
        # This can be made more sophisticated or user-selectable.
        # For now, using a fixed channel or asking the user.
        # Example: Ask user for channel index
        # channel_idx_str = simpledialog.askstring("Input", f"Enter channel index for analysis (0-{data.shape[0]-1}):", parent=tk.Tk().withdraw())
        # try:
        #     channel_idx = int(channel_idx_str)
        #     if not (0 <= channel_idx < data.shape[0]):
        #         raise ValueError("Channel index out of bounds.")
        # except (ValueError, TypeError):
        #     print("Invalid channel index input, defaulting to 0.")
        #     channel_idx = 0

        if data.shape[0] == 0:
            raise ValueError("No data channels found in the file.")
        
        channel_idx = 8 # Defaulting to channel 8 as in original snippet if available
        if channel_idx >= data.shape[0]:
            print(f"Warning: Channel index {channel_idx} is out of bounds for {data.shape[0]} channels. Using channel 0.")
            channel_idx = 0
        
        print(f"Using channel {channel_idx} ({raw.ch_names[channel_idx]}) for initial artifact detection.")
        signal_for_detection = data[channel_idx].copy() # Use a copy of the single channel data

        # 2) Compute PSD and find stimulation frequency using the selected channel
        # Using single channel for PSD to find stim_freq might be cleaner if that channel has strong artifact
        # Or compute mean PSD across all channels as before. Let's use the selected channel's data.
        # compute_mean_psd expects 2D array, so reshape:
        psd_selected_ch, freqs = compute_mean_psd(signal_for_detection.reshape(1, -1), sfreq)
        # prominence and min_freq might need tuning based on data characteristics
        stim_freq = find_stim_frequency(psd_selected_ch, freqs, prominence=10, min_freq=10) # Adjusted prominence, min_freq
        print(f"Estimated stimulation frequency: {stim_freq:.2f} Hz")
        results['stim_freq'] = stim_freq

        # 3) Detect stimulation epochs on the selected channel's signal
        stim_start_time, stim_end_time = detect_stim_epochs(signal_for_detection, sfreq, stim_freq)
        print(f"Stimulation period detected: {stim_start_time:.3f}s to {stim_end_time:.3f}s")
        results['stim_start_time'] = stim_start_time
        results['stim_end_time'] = stim_end_time


        # 4) Find template pulse from the selected channel's signal
        template, template_start_idx, template_duration_samples = find_template_pulse(
            signal_for_detection, sfreq, stim_freq, stim_start_time, stim_end_time)
        results['template'] = template
        results['template_start_idx'] = template_start_idx
        results['template_duration_samples'] = template_duration_samples


        # 5) Find all pulses using cross-correlation on the selected channel's signal
        # pulse_starts/ends are inclusive indices
        pulse_starts, pulse_ends, correlation_scores = cross_correlate_pulses(
            signal_for_detection, template, sfreq, stim_freq, stim_start_time, stim_end_time)
        results['correlation_scores'] = correlation_scores

        if len(pulse_starts) == 0:
            print("No pulses found by cross-correlation. Cannot proceed with refinement or interpolation.")
            return results # Return partial results

        # 6) Refine pulse boundaries (still on the selected channel's signal)
        # Pass stim_freq. pulse_starts/ends are inclusive.
        pulse_starts_refined, pulse_ends_refined = refine_pulse_boundaries(
            signal_for_detection, pulse_starts, pulse_ends, sfreq, stim_freq)
        
        if len(pulse_starts_refined) == 0:
            print("No pulses remained after refinement. Using unrefined pulses for interpolation if any.")
            # Fallback to unrefined pulses if refinement removed all
            if len(pulse_starts)>0:
                 pulse_starts_refined, pulse_ends_refined = pulse_starts, pulse_ends
            else:
                 print("No unrefined pulses either. Interpolation cannot proceed.")
                 return results


        results['pulse_starts_refined'] = pulse_starts_refined
        results['pulse_ends_refined'] = pulse_ends_refined # Inclusive end
        # Durations in seconds (inclusive start to inclusive end)
        results['pulse_durations_sec'] = (pulse_ends_refined - pulse_starts_refined + 1) / sfreq
        results['pulse_times_sec'] = pulse_starts_refined / sfreq


        # 7) Comprehensive visualization (using the selected channel's signal)
        visualize_pulse_detection(signal_for_detection, sfreq, stim_freq, template, template_start_idx,
                                pulse_starts_refined, pulse_ends_refined,
                                stim_start_time, stim_end_time)

        # 8) Apply Cubic Spline Interpolation to all channels using boundaries from selected channel
        print(f"\nApplying Cubic Spline Interpolation to ALL {data.shape[0]} channels...")
        print(f"Using refined pulse boundaries from channel {channel_idx} ({raw.ch_names[channel_idx]}) for all channels.")
        print(f"Number of pulses to interpolate: {len(pulse_starts_refined)}")

        # The 'data' variable holds all channel data (channels x samples)
        # pulse_starts_refined and pulse_ends_refined are inclusive indices from single channel analysis.
        corrected_data_all_channels = spline_artifact_extended_anchors(
            data,  # Full data array (n_channels, n_samples)
            pulse_starts_refined, # artifact_starts (inclusive)
            pulse_ends_refined,   # artifact_ends (inclusive)
            sfreq,
            buffer_ms=5.0  # Buffer in milliseconds for spline anchor points
        )
        print("Cubic spline interpolation complete for all channels.")
        results['corrected_data_all_channels'] = corrected_data_all_channels


        # 9) Visualize the effect of spline interpolation on the selected channel
        print(f"Visualizing spline interpolation result for selected channel {channel_idx} ({raw.ch_names[channel_idx]})...")

        original_signal_selected_channel = data[channel_idx] # Original data for the analyzed channel
        if corrected_data_all_channels is not None and corrected_data_all_channels.ndim == 2 and channel_idx < corrected_data_all_channels.shape[0]:
            corrected_signal_selected_channel = corrected_data_all_channels[channel_idx]
        elif corrected_data_all_channels is not None and corrected_data_all_channels.ndim == 1 and channel_idx == 0: # If data was 1D and corrected
            corrected_signal_selected_channel = corrected_data_all_channels
        else:
            print(f"Warning: Could not retrieve corrected data for channel {channel_idx}. Skipping spline visualization.")
            corrected_signal_selected_channel = None

        if corrected_signal_selected_channel is not None:
            times_viz = np.arange(len(original_signal_selected_channel)) / sfreq
            plt.figure(figsize=(18, 7))

            # Define a window for plotting, e.g., around the stimulation period or first few pulses
            plot_window_padding_sec = 0.2 # seconds
            if stim_start_time is not None and stim_end_time is not None and stim_end_time > stim_start_time :
                vis_start_time_spline = max(0, stim_start_time - plot_window_padding_sec)
                vis_end_time_spline = min(times_viz[-1], stim_end_time + plot_window_padding_sec)
            elif len(pulse_starts_refined) > 0: # Fallback to first/last pulse if stim period is weird
                vis_start_time_spline = max(0, (pulse_starts_refined[0] / sfreq) - plot_window_padding_sec)
                # Show a few pulses or a fixed duration
                vis_end_time_spline = min(times_viz[-1], (pulse_starts_refined[min(len(pulse_starts_refined)-1, 10)] / sfreq) + plot_window_padding_sec + (1/stim_freq if stim_freq >0 else 0.1) )
            else: # Fallback to a short segment from start if no pulses/stim_period
                vis_start_time_spline = times_viz[0]
                vis_end_time_spline = min(times_viz[-1], times_viz[0] + 5.0) # Show 5 seconds


            plot_start_idx_spline = max(0, int(vis_start_time_spline * sfreq))
            plot_end_idx_spline = min(len(original_signal_selected_channel), int(vis_end_time_spline * sfreq))

            if plot_start_idx_spline >= plot_end_idx_spline and len(original_signal_selected_channel) > 0 : # If range is invalid, plot a default portion
                plot_start_idx_spline = 0
                plot_end_idx_spline = min(len(original_signal_selected_channel), int(sfreq * 2)) # e.g., first 2 seconds

            # Plot original signal in the window
            plt.plot(times_viz[plot_start_idx_spline:plot_end_idx_spline],
                     original_signal_selected_channel[plot_start_idx_spline:plot_end_idx_spline],
                     label=f'Original Ch {channel_idx} ({raw.ch_names[channel_idx]})',
                     color='dimgray', alpha=0.7, linewidth=1.0)

            # Plot corrected signal in the window
            plt.plot(times_viz[plot_start_idx_spline:plot_end_idx_spline],
                     corrected_signal_selected_channel[plot_start_idx_spline:plot_end_idx_spline],
                     label=f'Corrected Ch {channel_idx} (Spline)', color='dodgerblue', linewidth=1.2, alpha=0.9)

            # Highlight the interpolated segments on the corrected signal plot
            first_interp_label = True
            for i in range(len(pulse_starts_refined)):
                start_sample_interp = pulse_starts_refined[i] # inclusive
                end_sample_interp = pulse_ends_refined[i]   # inclusive

                # Determine the part of the pulse that is within the current plot window
                seg_display_start_idx = max(plot_start_idx_spline, start_sample_interp)
                seg_display_end_idx = min(plot_end_idx_spline, end_sample_interp + 1) # +1 for Python slicing up to end_sample

                if seg_display_start_idx < seg_display_end_idx: # If the segment is visible
                    times_segment = times_viz[seg_display_start_idx:seg_display_end_idx]
                    data_segment = corrected_signal_selected_channel[seg_display_start_idx:seg_display_end_idx]

                    if len(times_segment) > 0 and len(data_segment) > 0:
                         plt.plot(times_segment, data_segment, color='red', linewidth=1.8, linestyle='-',
                                 label='Interpolated Segment' if first_interp_label else "",
                                 zorder=5)
                         first_interp_label = False

            plt.title(f'Effect of Cubic Spline Interpolation on Channel {channel_idx} ({raw.ch_names[channel_idx]})', fontsize=14)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)

            if plot_start_idx_spline < plot_end_idx_spline : # Ensure valid plot range
                plt.xlim(times_viz[plot_start_idx_spline], times_viz[plot_end_idx_spline-1 if plot_end_idx_spline > 0 else 0])
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()

        return results

    except ValueError as ve: # Catch specific expected errors first
        print(f"ValueError in main pipeline: {str(ve)}")
        import traceback
        traceback.print_exc()
        return results # Return any partial results
    except FileNotFoundError as fe:
        print(f"FileNotFoundError in main pipeline: {str(fe)}")
        import traceback
        traceback.print_exc()
        return results
    except Exception as e:
        print(f"Unexpected error in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return results # Return any partial results

if __name__ == "__main__":
    analysis_results = main()

    if analysis_results and analysis_results.get('pulse_times_sec') is not None and len(analysis_results['pulse_times_sec']) > 0:
        print("\n--- Analysis Summary from Results ---")
        print(f"Stimulation Frequency: {analysis_results.get('stim_freq', 'N/A'):.2f} Hz")
        print(f"Number of pulses detected and refined: {len(analysis_results['pulse_starts_refined'])}")
        print(f"First 5 pulse start times (s): {np.round(analysis_results['pulse_times_sec'][:5], 4)}")
        print(f"First 5 pulse durations (s): {np.round(analysis_results['pulse_durations_sec'][:5], 4)}")
        if 'corrected_data_all_channels' in analysis_results and analysis_results['corrected_data_all_channels'] is not None:
            print(f"Corrected data shape: {analysis_results['corrected_data_all_channels'].shape}")
        print("Pipeline completed.")
    elif analysis_results:
        print("\nPipeline completed, but no pulses were fully processed or some results are missing.")
        if 'stim_freq' in analysis_results:
             print(f"  Stimulation Frequency Found: {analysis_results['stim_freq']:.2f} Hz")
        if 'stim_start_time' in analysis_results:
             print(f"  Stimulation Epoch: {analysis_results['stim_start_time']:.2f}s - {analysis_results['stim_end_time']:.2f}s")
    else:
        print("Pipeline failed to complete or was exited early.")