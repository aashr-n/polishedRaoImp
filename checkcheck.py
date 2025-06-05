# Enhanced Stimulation Pulse Artifact Detection
# Building on Kristin Sellers' 2018 artifact rejection protocol
# Adds comprehensive individual pulse detection and template matching

import mne
import scipy
import os
import numpy as np
from scipy.signal import welch, find_peaks, correlate
import matplotlib.pyplot as plt
from scipy.signal import square # Not used, but kept from original
from scipy.stats import zscore # Not used, but kept from original
from scipy.ndimage import gaussian_filter1d

from scipy.interpolate import CubicSpline

# GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox # Added messagebox
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
        # raise SystemExit("No file selected.") # Changed to return None
        print("No file selected by user.")
        return None
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
    peaks, props = find_peaks(mean_psd, prominence=prominence)
    if len(peaks) == 0:
        raise ValueError(f"No peaks found with prominence ≥ {prominence}")

    pfreqs = freqs[peaks]
    proms = props['prominences']
    heights = mean_psd[peaks]
    rel_proms = proms / heights

    mask_freq = pfreqs >= min_freq
    if not np.any(mask_freq): # Check if any peaks remain after frequency filter
        raise ValueError(f"No peaks found above {min_freq} Hz with prominence ≥ {prominence} (initial count: {len(pfreqs)})")

    pfreqs = pfreqs[mask_freq]
    proms = proms[mask_freq]
    rel_proms = rel_proms[mask_freq]
    heights = heights[mask_freq]


    print(f"Peaks ≥ {min_freq} Hz with abs prom ≥ {prominence}:")
    for f, p, rp, h in zip(pfreqs, proms, rel_proms, heights):
        print(f"  {f:.2f} Hz → abs prom {p:.4f}, rel prom {rp:.4f}, height {h:.4f}")

    # Select the lowest-frequency peak with relative prominence ≥ 0.5 (or highest if none meet criteria)
    mask_rel = rel_proms >= 0.5
    if not np.any(mask_rel):
        print("No peaks met relative prominence threshold of 0.5. Considering all valid peaks.")
        # Fallback: consider all peaks that passed frequency and absolute prominence filters
        # Could choose the one with max absolute prominence, or lowest frequency still.
        # For now, let's stick to lowest frequency among the initially filtered ones.
        stim_freq = np.min(pfreqs) # pfreqs is already filtered by min_freq and prominence
    else:
        valid_freqs_for_stim = pfreqs[mask_rel]
        stim_freq = np.min(valid_freqs_for_stim)

    return stim_freq

# --- NEW FUNCTIONS for template detection and pulse finding ---

def detect_stim_epochs(signal, sfreq, stim_freq):
    """
    Detect stimulation epochs using sliding window PSD analysis
    Returns: stim_start_time, stim_end_time
    """
    win_sec = max(0.2, round(sfreq/5000, 1))
    step_sec = win_sec / 2
    nperseg = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)

    segment_centers, segment_power = [], []
    for start in range(0, signal.size - nperseg + 1, step_samps):
        stop = start + nperseg
        segment = signal[start:stop]
        freqs_w, psd_w = welch(segment, fs=sfreq, nperseg=min(nperseg, len(segment)))
        idx = np.argmin(np.abs(freqs_w - stim_freq))
        segment_centers.append((start + stop) / 2 / sfreq)
        total_power = np.sum(psd_w)
        rel_power = (psd_w[idx] / total_power) * 100 if total_power > 0 else 0
        segment_power.append(rel_power)

    segment_power = np.array(segment_power)
    segment_centers = np.array(segment_centers)

    if len(segment_power) == 0: raise ValueError("No segments processed for epoch detection")
    max_prom_val = segment_power.max()
    if max_prom_val == 0: raise ValueError("No stimulation activity detected (max prominence is 0)")

    max_idxs = np.where(segment_power == max_prom_val)[0]
    first_max, last_max = max_idxs.min(), max_idxs.max()
    drop_thresh = 0.1 * max_prom_val

    start_idx_epoch = first_max
    while start_idx_epoch > 0 and segment_power[start_idx_epoch] >= drop_thresh: start_idx_epoch -= 1
    stim_start_time = segment_centers[start_idx_epoch]

    end_idx_epoch = last_max
    while end_idx_epoch < len(segment_power) - 1 and segment_power[end_idx_epoch] >= drop_thresh: end_idx_epoch += 1
    stim_end_time = segment_centers[end_idx_epoch]
    return stim_start_time, stim_end_time


def find_template_pulse(signal, sfreq, stim_freq, stim_start_time, stim_end_time):
    """
    Find the best template pulse within the stimulation period.
    The template is centered around its peak and has width 1/stim_freq.
    Returns: template, template_start_idx (abs first sample), template_length (samples)
    """
    start_idx_stim_period = int(stim_start_time * sfreq)
    end_idx_stim_period = int(stim_end_time * sfreq)
    start_idx_stim_period = max(0, start_idx_stim_period)
    end_idx_stim_period = min(len(signal), end_idx_stim_period)

    if start_idx_stim_period >= end_idx_stim_period:
        raise ValueError("Invalid stimulation period indices for template search.")

    stim_signal = signal[start_idx_stim_period:end_idx_stim_period]
    period_samples = int(sfreq / stim_freq)
    if period_samples <= 0:
        raise ValueError(f"Period_samples must be positive. Got {period_samples}.")

    template_duration_samples = period_samples

    if template_duration_samples > len(stim_signal):
        print(f"Warning: stim_signal (len {len(stim_signal)}) shorter than template_duration ({template_duration_samples}). Using full stim_signal.")
        template_duration_samples = len(stim_signal)
        if template_duration_samples == 0: raise ValueError("Stim_signal empty, cannot make template.")

    best_template = None
    best_score = -np.inf
    best_template_start_abs_idx = -1

    # Search for a good segment to find a peak, then center template.
    # Limit search to a reasonable portion of stim_signal
    search_limit_in_stim_signal = min(len(stim_signal) - template_duration_samples + 1, 10 * period_samples)
    step = max(1, period_samples // 4)

    for i in range(0, search_limit_in_stim_signal, step):
        # This is the window where we look for a peak
        segment_for_peak_search = stim_signal[i : i + template_duration_samples]
        if len(segment_for_peak_search) < template_duration_samples : continue # Should not happen with loop range

        peak_idx_in_segment = np.argmax(np.abs(segment_for_peak_search))
        peak_idx_relative_to_stim_signal = i + peak_idx_in_segment

        # Define template centered around this peak
        current_template_start_relative = peak_idx_relative_to_stim_signal - (template_duration_samples // 2)
        current_template_end_relative = current_template_start_relative + template_duration_samples # Exclusive end for slice

        if current_template_start_relative >= 0 and current_template_end_relative <= len(stim_signal):
            actual_template_candidate = stim_signal[current_template_start_relative : current_template_end_relative]

            if len(actual_template_candidate) == template_duration_samples: # Ensure full length
                variance_score = np.var(actual_template_candidate)
                pp_score = np.ptp(actual_template_candidate)
                combined_score = variance_score * pp_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_template = actual_template_candidate.copy()
                    best_template_start_abs_idx = start_idx_stim_period + current_template_start_relative

    if best_template is None:
        if len(stim_signal) >= template_duration_samples:
            print("Warning: Template by peak centering failed. Taking first valid segment of stim_signal as template.")
            # Center this first segment around its peak as a fallback
            first_segment = stim_signal[0:template_duration_samples]
            peak_idx_in_first_segment = np.argmax(np.abs(first_segment))
            # peak_idx_relative_to_stim_signal is peak_idx_in_first_segment for i=0
            
            current_template_start_relative = peak_idx_in_first_segment - (template_duration_samples // 2)
            current_template_end_relative = current_template_start_relative + template_duration_samples

            if current_template_start_relative >=0 and current_template_end_relative <= len(stim_signal):
                 best_template = stim_signal[current_template_start_relative:current_template_end_relative].copy()
                 best_template_start_abs_idx = start_idx_stim_period + current_template_start_relative
            else: # True fallback if centering pushes out of bounds
                 best_template = first_segment.copy()
                 best_template_start_abs_idx = start_idx_stim_period

            if best_template is None: # Should not happen if len(stim_signal) >= template_duration_samples
                 raise ValueError("Fallback template creation failed.")
        else:
            raise ValueError("Could not find suitable template pulse. Stimulation signal may be too short or flat.")

    print(f"Template found starting at sample {best_template_start_abs_idx} ({best_template_start_abs_idx/sfreq:.3f}s)")
    print(f"Template duration: {len(best_template)} samples ({len(best_template)/sfreq:.3f}s)")
    return best_template, best_template_start_abs_idx, len(best_template)


def cross_correlate_pulses(signal, template, sfreq, stim_freq, stim_start_time, stim_end_time):
    start_idx_corr = int(stim_start_time * sfreq)
    end_idx_corr = int(stim_end_time * sfreq)
    start_idx_corr = max(0, start_idx_corr)
    end_idx_corr = min(len(signal), end_idx_corr)

    if start_idx_corr >= end_idx_corr or (end_idx_corr - start_idx_corr) < len(template):
        print("Warning: Stim segment for correlation too short. No pulses from cross-correlation.")
        return np.array([]), np.array([]), np.array([])
    stim_signal_corr = signal[start_idx_corr:end_idx_corr]

    template_std = np.std(template)
    template_norm = (template - np.mean(template)) / template_std if template_std > 1e-9 else template - np.mean(template)
    correlation = correlate(stim_signal_corr, template_norm, mode='valid')
    if len(correlation) == 0: return np.array([]), np.array([]), np.array([])

    period_samples = int(sfreq / stim_freq)
    min_distance_peaks = max(1, int(0.7 * period_samples))
    correlation_threshold = np.percentile(correlation, 85) # Consider values in the top 15%
    peaks_in_corr, _ = find_peaks(correlation, height=correlation_threshold, distance=min_distance_peaks)

    pulse_starts_abs = peaks_in_corr + start_idx_corr
    pulse_ends_abs = pulse_starts_abs + len(template) - 1 # Inclusive end
    correlation_scores = correlation[peaks_in_corr]
    print(f"Found {len(pulse_starts_abs)} pulse artifacts via cross-correlation")
    return pulse_starts_abs, pulse_ends_abs, correlation_scores


def refine_pulse_boundaries(signal, pulse_starts_initial, initial_pulse_ends_unused, sfreq, stim_freq):
    """
    Refines pulse boundaries to ensure the peak of each stimulation artifact is
    centered within a window of width 1/stim_freq.
    'initial_pulse_ends_unused' is kept for signature but not used.
    Returns: refined_starts, refined_ends (both inclusive indices)
    """
    if len(pulse_starts_initial) == 0:
        return np.array([]), np.array([])

    refined_starts_list = []
    refined_ends_list = []
    period_samples = int(sfreq / stim_freq)

    if period_samples <= 0:
        print(f"Warning: period_samples is {period_samples}. Cannot refine boundaries.")
        # Return original starts and attempt to derive ends based on period_samples if possible
        # This path means stim_freq or sfreq is problematic.
        if len(pulse_starts_initial)>0 and period_samples > 0:
             ends = pulse_starts_initial + period_samples - 1
             return np.clip(pulse_starts_initial,0,len(signal)-1), np.clip(ends,0,len(signal)-1)
        return pulse_starts_initial, np.array([]) # Fallback, likely problematic downstream


    half_period_floor = period_samples // 2

    for i in range(len(pulse_starts_initial)):
        s_initial = pulse_starts_initial[i]

        # Define search window for the peak: from initial start to one period length
        search_win_start = s_initial
        search_win_end = s_initial + period_samples # Exclusive end for slicing

        search_win_start_clipped = max(0, search_win_start)
        search_win_end_clipped = min(len(signal), search_win_end)

        if search_win_start_clipped >= search_win_end_clipped:
            print(f"Info: Pulse {i} at {s_initial}s: initial search window [{search_win_start_clipped}-{search_win_end_clipped}] too small or invalid. Trying a broader window.")
            # Try a broader window around s_initial if the default is bad (e.g. s_initial too close to end)
            search_win_start_clipped = max(0, s_initial - half_period_floor)
            search_win_end_clipped = min(len(signal), s_initial + period_samples - half_period_floor)
            if search_win_start_clipped >= search_win_end_clipped:
                 print(f"  Broadened search window also invalid. Skipping refinement for this pulse.")
                 continue


        pulse_segment = signal[search_win_start_clipped : search_win_end_clipped]

        if len(pulse_segment) == 0:
            continue

        peak_idx_in_segment = np.argmax(np.abs(pulse_segment))
        global_peak_idx = search_win_start_clipped + peak_idx_in_segment

        new_start = global_peak_idx - half_period_floor
        new_end = new_start + period_samples - 1 # Inclusive end

        refined_starts_list.append(new_start)
        refined_ends_list.append(new_end)

    if not refined_starts_list: # If all pulses were skipped
        print("Warning: No pulses could be refined with peak centering.")
        return np.array([]), np.array([])

    final_refined_starts = np.array(refined_starts_list, dtype=int)
    final_refined_ends = np.array(refined_ends_list, dtype=int)

    # Clip to ensure boundaries are within signal limits
    final_refined_starts = np.clip(final_refined_starts, 0, len(signal) - 1)
    final_refined_ends = np.clip(final_refined_ends, 0, len(signal) - 1)

    valid_indices = (final_refined_starts <= final_refined_ends) & \
                    (final_refined_starts < len(signal) - (1 if period_samples > 1 else 0) ) # Start shouldn't be the very last sample if pulse is longer
    
    final_refined_starts = final_refined_starts[valid_indices]
    final_refined_ends = final_refined_ends[valid_indices]
    
    if len(final_refined_starts) < len(pulse_starts_initial):
        print(f"Info: Peak-centering refinement resulted in {len(pulse_starts_initial) - len(final_refined_starts)} pulses being dropped (due to boundary issues or invalid windows).")

    return final_refined_starts, final_refined_ends


def visualize_pulse_detection(signal, sfreq, stim_freq, template, template_start_idx,
                            pulse_starts, pulse_ends, stim_start_time, stim_end_time):
    times = np.arange(len(signal)) / sfreq
    period_samples = int(sfreq / stim_freq)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=False)

    axes[0].plot(times, signal, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axvspan(stim_start_time, stim_end_time, color='red', alpha=0.2, label=f'Stim Period ({stim_start_time:.2f}s - {stim_end_time:.2f}s)')
    axes[0].set_title('Full Signal with Stimulation Period')
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right'); axes[0].grid(True, alpha=0.3)
    if stim_end_time > 0 and (stim_end_time - stim_start_time) < 0.3 * (times[-1] if len(times)>0 else 0) :
        plot_start_time_ax0 = max(0, stim_start_time - 0.1*(stim_end_time - stim_start_time))
        plot_end_time_ax0 = min(times[-1] if len(times)>0 else stim_end_time, stim_end_time + 0.1*(stim_end_time - stim_start_time))
        if plot_start_time_ax0 < plot_end_time_ax0: axes[0].set_xlim(plot_start_time_ax0, plot_end_time_ax0)

    template_times = np.arange(len(template)) / sfreq
    axes[1].plot(template_times, template, 'g-', linewidth=2)
    axes[1].set_title(f'Template Pulse (from {template_start_idx/sfreq:.3f}s, duration {len(template)/sfreq:.4f}s)')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Amplitude'); axes[1].grid(True, alpha=0.3)

    stim_start_idx_viz = int(stim_start_time * sfreq)
    stim_end_idx_viz = int(stim_end_time * sfreq)
    padding_viz_samples = int(0.1 * max(0, (stim_end_idx_viz - stim_start_idx_viz)))
    stim_start_idx_viz = max(0, stim_start_idx_viz - padding_viz_samples)
    stim_end_idx_viz = min(len(signal), stim_end_idx_viz + padding_viz_samples)

    stim_signal_viz_slice = slice(stim_start_idx_viz, stim_end_idx_viz)
    if stim_start_idx_viz < stim_end_idx_viz and len(signal[stim_signal_viz_slice]) > 0:
        stim_times_viz = times[stim_signal_viz_slice]
        stim_signal_for_plot = signal[stim_signal_viz_slice]
        axes[2].plot(stim_times_viz, stim_signal_for_plot, 'b-', alpha=0.7, linewidth=0.8)
        axes[2].set_xlim(stim_times_viz[0], stim_times_viz[-1])
    else:
        axes[2].plot(times, signal, 'b-', alpha=0.7, linewidth=0.8)
        stim_signal_for_plot = signal # for text y-pos

    if len(pulse_starts) > 0 and len(pulse_ends) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(pulse_starts)))
        for i in range(len(pulse_starts)):
            start_sample, end_sample = pulse_starts[i], pulse_ends[i]
            start_time_vspan, end_time_vspan = start_sample / sfreq, (end_sample + 1) / sfreq
            axes[2].axvspan(start_time_vspan, end_time_vspan, color=colors[i], alpha=0.35)
            mid_time_text = (start_sample + 0.5 * (end_sample - start_sample)) / sfreq
            text_y_idx = int(mid_time_text * sfreq)
            text_y_val = signal[text_y_idx] if 0 <= text_y_idx < len(signal) else (np.max(stim_signal_for_plot) * 0.9 if len(stim_signal_for_plot)>0 else 0)
            axes[2].text(mid_time_text, text_y_val, str(i+1), ha='center', va='bottom', fontsize=7, color='black', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, lw=0.5))
    axes[2].set_title(f'Detected Pulses in Stimulation Period ({len(pulse_starts)} pulses)')
    axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Amplitude'); axes[2].grid(True, alpha=0.3)

    if len(pulse_starts) > 1:
        pulse_start_times_sec = pulse_starts / sfreq
        inter_pulse_intervals_sec = np.diff(pulse_start_times_sec)
        mean_interval = np.mean(inter_pulse_intervals_sec)
        axes[3].plot(pulse_start_times_sec[:-1], inter_pulse_intervals_sec, 'ro-', markersize=4, linewidth=1)
        axes[3].axhline(y=mean_interval, color='g', linestyle='--', label=f'Mean Interval: {mean_interval:.4f}s')
        axes[3].axhline(y=1/stim_freq, color='purple', linestyle=':', label=f'Nominal (1/stim_freq): {1/stim_freq:.4f}s')
        axes[3].set_title('Inter-Pulse Intervals (Start-to-Start)'); axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('Interval (s)')
        axes[3].legend(loc='upper right'); axes[3].grid(True, alpha=0.3)
        if stim_start_idx_viz < stim_end_idx_viz and len(signal[stim_signal_viz_slice]) > 0: axes[3].set_xlim(stim_times_viz[0], stim_times_viz[-1])
    else:
        axes[3].text(0.5, 0.5, 'Insufficient pulses for interval analysis (<2)', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Inter-Pulse Intervals - Not Available')
    plt.tight_layout(pad=1.5); plt.show()

    print("\n=== PULSE DETECTION SUMMARY ===")
    print(f"Total pulses detected: {len(pulse_starts)}")
    print(f"Stimulation period: {stim_start_time:.3f}s to {stim_end_time:.3f}s (Duration: {stim_end_time-stim_start_time:.3f}s)")
    print(f"Nominal pulse width (1/stim_freq): {1/stim_freq:.4f}s ({period_samples} samples at {sfreq} Hz)")
    if len(pulse_starts) > 1:
        print(f"Mean inter-pulse interval (start-to-start): {mean_interval:.4f}s ± {np.std(inter_pulse_intervals_sec):.4f}s")
        if mean_interval > 0: print(f"  Est. freq from intervals: {1 / mean_interval:.2f} Hz (cf. input: {stim_freq:.2f} Hz)")
    if len(pulse_starts) > 0 and len(pulse_ends) > 0:
        actual_dur_samp = (pulse_ends - pulse_starts) + 1
        actual_dur_sec = actual_dur_samp / sfreq
        print(f"Effective pulse duration range (after centering & clipping): {np.min(actual_dur_sec):.4f}s to {np.max(actual_dur_sec):.4f}s")
        num_non_nominal = np.sum(actual_dur_samp != period_samples)
        if num_non_nominal > 0: print(f"  ({num_non_nominal} pulses have durations not exactly {period_samples} samples, likely due to boundary clipping)")


def spline_artifact_extended_anchors(data, artifact_starts, artifact_ends, sfreq, buffer_ms=5.0):
    if not isinstance(data, np.ndarray): raise TypeError("Input 'data' must be a NumPy array.")
    if sfreq <= 0: raise ValueError("sfreq must be positive.")
    if not (isinstance(artifact_starts, np.ndarray) and isinstance(artifact_ends, np.ndarray)):
        raise TypeError("'artifact_starts' and 'artifact_ends' must be NumPy arrays.")
    if artifact_starts.shape != artifact_ends.shape: raise ValueError("Shapes of artifact_starts and _ends must match.")
    if artifact_starts.ndim != 1: raise ValueError("artifact_starts and _ends must be 1D.")

    is_1d_input = data.ndim == 1
    spline_data = data.reshape(1, -1).copy() if is_1d_input else data.copy()
    num_channels, num_samples = spline_data.shape
    buffer_samples = int(buffer_ms / 1000.0 * sfreq)
    print(f"Spline: Buffer {buffer_samples} samples ({buffer_ms}ms at {sfreq}Hz)")

    for i_chan in range(num_channels):
        print(f"Spline Chan {i_chan + 1}/{num_channels}")
        if num_samples > 0 and np.isnan(spline_data[i_chan, 0]):
            print(f"  Chan {i_chan + 1}: First sample NaN, skipping."); continue
        if len(artifact_starts) == 0: continue

        for i_stim in range(len(artifact_starts)):
            start_artifact, end_artifact = int(artifact_starts[i_stim]), int(artifact_ends[i_stim])
            if not (0 <= start_artifact < num_samples and 0 <= end_artifact < num_samples and start_artifact <= end_artifact):
                print(f"  Warn: Art {i_stim+1} invalid bounds ({start_artifact}-{end_artifact}). Skip."); continue

            anchor1_ideal, anchor2_ideal = start_artifact - buffer_samples, end_artifact + buffer_samples
            anchor1_clip, anchor2_clip = max(0, anchor1_ideal), min(num_samples - 1, anchor2_ideal)
            use_extended = buffer_samples > 0 and anchor1_clip < start_artifact and anchor2_clip > end_artifact and anchor1_clip < anchor2_clip

            if use_extended:
                x_known = np.array([anchor1_clip, anchor2_clip])
            else:
                x_anchor_start = max(0, start_artifact - 1)
                x_anchor_end = min(num_samples - 1, end_artifact + 1)
                if x_anchor_start < start_artifact and x_anchor_end > end_artifact and x_anchor_start < x_anchor_end:
                    x_known = np.array([x_anchor_start, x_anchor_end])
                    if not use_extended and buffer_samples > 0: print(f"  Info: Art {i_stim+1} fallback to immediate pre/post anchors.")
                else:
                    x_known = np.array([start_artifact, end_artifact]) # True fallback
                    if not use_extended and buffer_samples > 0: print(f"  Info: Art {i_stim+1} fallback to artifact boundary anchors.")
            
            if x_known[0] >= x_known[1] and not (x_known[0] == x_known[1] and start_artifact == end_artifact):
                 print(f"  Warn: Art {i_stim+1} anchors ({x_known[0]},{x_known[1]}) not distinct/increasing. Skip."); continue
            
            y_known = spline_data[i_chan, x_known]
            if np.any(np.isnan(y_known)): print(f"  Warn: Art {i_stim+1} NaN in y_known. Skip."); continue
            
            x_query = np.arange(start_artifact, end_artifact + 1)
            if len(x_query) == 0: continue

            try:
                if x_known[0] == x_known[1] and len(x_query) > 0 : # Single unique anchor point
                    print(f"  Info: Art {i_stim+1} single unique anchor. Fill with anchor value.")
                    interpolated_values = np.full(len(x_query), y_known[0])
                else:
                    cs = CubicSpline(x_known, y_known, bc_type='not-a-knot')
                    interpolated_values = cs(x_query)
                spline_data[i_chan, start_artifact : end_artifact + 1] = interpolated_values
            except ValueError as e:
                print(f"  Error: Art {i_stim+1} spline ({start_artifact}-{end_artifact}): {e}. Skip."); continue
    return spline_data.squeeze() if is_1d_input else spline_data


def main():
    results = {}
    try:
        default_path = '' # Provide a default path if desired, e.g. from previous run
        # path = default_path # Uncomment to use default without asking if it exists
        # For first time use or if you always want to select:
        path = select_fif_file()
        if not path: print("No file selected. Exiting."); return None
        
        # Example: Ask user for channel index
        # temp_root = tk.Tk(); temp_root.withdraw() # Create and hide Tk root for dialog
        # channel_idx_str = simpledialog.askstring("Input", f"Enter channel index for analysis (e.g., 0 to N-1):", parent=temp_root)
        # temp_root.destroy()
        # try:
        #     channel_idx_for_analysis = int(channel_idx_str)
        # except (ValueError, TypeError):
        #     print("Invalid channel index input, defaulting to 0.")
        #     channel_idx_for_analysis = 0 # Fallback channel index

        channel_idx_for_analysis = 8 # Defaulting to channel 8

        raw = mne.io.read_raw_fif(path, preload=True)
        sfreq = raw.info['sfreq']; results['sfreq'] = sfreq
        data = raw.get_data()
        if data.shape[0] == 0: raise ValueError("No data channels in file.")
        
        if not (0 <= channel_idx_for_analysis < data.shape[0]):
            print(f"Warn: Chan {channel_idx_for_analysis} out of bounds ({data.shape[0]} chans). Using 0.")
            channel_idx_for_analysis = 0
        print(f"Using channel {channel_idx_for_analysis} ({raw.ch_names[channel_idx_for_analysis]}) for initial detection.")
        signal_for_detection = data[channel_idx_for_analysis].copy()

        psd_ch, freqs = compute_mean_psd(signal_for_detection.reshape(1, -1), sfreq)
        stim_freq = find_stim_frequency(psd_ch, freqs, prominence=10, min_freq=10) # Tune these
        results['stim_freq'] = stim_freq; print(f"Est. stim_freq: {stim_freq:.2f} Hz")

        stim_start_t, stim_end_t = detect_stim_epochs(signal_for_detection, sfreq, stim_freq)
        results['stim_start_time'] = stim_start_t; results['stim_end_time'] = stim_end_t
        print(f"Stim period: {stim_start_t:.3f}s to {stim_end_t:.3f}s")

        template, template_start_idx, template_len = find_template_pulse(
            signal_for_detection, sfreq, stim_freq, stim_start_t, stim_end_t)
        results['template'] = template; results['template_start_idx'] = template_start_idx
        results['template_duration_samples'] = template_len

        p_starts, p_ends, corr_scores = cross_correlate_pulses(
            signal_for_detection, template, sfreq, stim_freq, stim_start_t, stim_end_t)
        results['correlation_scores'] = corr_scores
        if len(p_starts) == 0: print("No pulses from cross-correlation. Cannot proceed."); return results

        p_starts_ref, p_ends_ref = refine_pulse_boundaries(
            signal_for_detection, p_starts, p_ends, sfreq, stim_freq) # p_ends is not used by new logic
        if len(p_starts_ref) == 0:
            print("No pulses after refinement. Using unrefined if available.")
            p_starts_ref, p_ends_ref = (p_starts, p_ends) if len(p_starts)>0 else (np.array([]), np.array([]))
        
        results['pulse_starts_refined'] = p_starts_ref # inclusive
        results['pulse_ends_refined'] = p_ends_ref     # inclusive
        if len(p_starts_ref)>0:
            results['pulse_durations_sec'] = (p_ends_ref - p_starts_ref + 1) / sfreq
            results['pulse_times_sec'] = p_starts_ref / sfreq
        else:
            results['pulse_durations_sec'] = np.array([])
            results['pulse_times_sec'] = np.array([])


        visualize_pulse_detection(signal_for_detection, sfreq, stim_freq, template, template_start_idx,
                                p_starts_ref, p_ends_ref, stim_start_t, stim_end_t)

        print(f"\nApplying Spline Interpolation to ALL {data.shape[0]} channels...")
        corrected_data_all = spline_artifact_extended_anchors(
            data, p_starts_ref, p_ends_ref, sfreq, buffer_ms=5.0)
        results['corrected_data_all_channels'] = corrected_data_all
        print("Spline interpolation complete for all channels.")

        print(f"Visualizing spline result for channel {channel_idx_for_analysis}...")
        original_sig_selected = data[channel_idx_for_analysis]
        corrected_sig_selected = corrected_data_all[channel_idx_for_analysis] if corrected_data_all.ndim == 2 else corrected_data_all

        times_v = np.arange(len(original_sig_selected)) / sfreq
        plt.figure(figsize=(18, 7))
        # Determine plot window for spline viz
        if len(p_starts_ref) > 0:
            vis_start_spline_s = max(0, (p_starts_ref[0] / sfreq) - 0.1)
            # Show a few pulses or a fixed duration
            num_pulses_to_show = min(len(p_starts_ref), 5)
            vis_end_spline_s = min(times_v[-1], (p_ends_ref[num_pulses_to_show-1] / sfreq) + 0.1)
        else: # Fallback if no pulses
            vis_start_spline_s = stim_start_t - 0.1 if stim_start_t is not None else 0
            vis_end_spline_s = stim_end_t + 0.1 if stim_end_t is not None else min(times_v[-1] if len(times_v)>0 else 2, 2.0) # Show 2s

        plot_start_idx_s = max(0, int(vis_start_spline_s * sfreq))
        plot_end_idx_s = min(len(original_sig_selected), int(vis_end_spline_s * sfreq))
        
        if plot_start_idx_s >= plot_end_idx_s and len(original_sig_selected) > 0 : # Default if range invalid
            plot_start_idx_s = 0; plot_end_idx_s = min(len(original_sig_selected), int(sfreq * 2))


        plt.plot(times_v[plot_start_idx_s:plot_end_idx_s],
                 original_sig_selected[plot_start_idx_s:plot_end_idx_s],
                 label=f'Original Ch {channel_idx_for_analysis}', color='dimgray', alpha=0.7, lw=1.0)
        plt.plot(times_v[plot_start_idx_s:plot_end_idx_s],
                 corrected_sig_selected[plot_start_idx_s:plot_end_idx_s],
                 label=f'Corrected Ch {channel_idx_for_analysis}', color='dodgerblue', lw=1.2, alpha=0.9)

        first_label = True
        for i in range(len(p_starts_ref)):
            s, e = p_starts_ref[i], p_ends_ref[i]
            seg_disp_start = max(plot_start_idx_s, s)
            seg_disp_end = min(plot_end_idx_s, e + 1)
            if seg_disp_start < seg_disp_end:
                ts, ds = times_v[seg_disp_start:seg_disp_end], corrected_sig_selected[seg_disp_start:seg_disp_end]
                if len(ts) > 0:
                     plt.plot(ts, ds, color='red', lw=1.8, label='Interpolated' if first_label else "", zorder=5)
                     first_label = False
        plt.title(f'Spline Interpolation Effect on Channel {channel_idx_for_analysis}', fontsize=14)
        plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
        if plot_start_idx_s < plot_end_idx_s and len(times_v)>max(plot_start_idx_s, plot_end_idx_s-1):
            plt.xlim(times_v[plot_start_idx_s], times_v[plot_end_idx_s-1 if plot_end_idx_s > 0 else 0])
        plt.legend(loc='upper right'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.show()
        return results

    except Exception as e:
        print(f"Error in main: {str(e)}"); import traceback; traceback.print_exc(); return results

if __name__ == "__main__":
    analysis_results = main()
    if analysis_results and analysis_results.get('pulse_times_sec', np.array([])).size > 0:
        print("\n--- Analysis Summary ---")
        print(f"Stim Freq: {analysis_results.get('stim_freq', 'N/A'):.2f} Hz")
        print(f"# Pulses refined: {len(analysis_results['pulse_starts_refined'])}")
        print(f"First 5 pulse starts (s): {np.round(analysis_results['pulse_times_sec'][:5], 4)}")
        print(f"First 5 pulse durations (s): {np.round(analysis_results['pulse_durations_sec'][:5], 4)}")
        if 'corrected_data_all_channels' in analysis_results and analysis_results['corrected_data_all_channels'] is not None:
            print(f"Corrected data shape: {analysis_results['corrected_data_all_channels'].shape}")
        print("Pipeline completed.")
    elif analysis_results: print("\nPipeline completed with partial or no pulse results.")
    else: print("Pipeline failed or exited early.")