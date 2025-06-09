# Enhanced Stimulation Pulse Artifact Detection
# Building on Kristin Sellers' 2018 artifact rejection protocol
# Adds comprehensive individual pulse detection and template matching

import mne
import scipy
import os
import numpy as np
from scipy.signal import welch, find_peaks, correlate
import matplotlib.pyplot as plt
# from scipy.signal import square # Not used, but kept from original
# from scipy.stats import zscore # Not used, but kept from original
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
        print("No file selected by user.")
        return None
    return path

def compute_mean_psd(data_ch, sfreq, bandwidth=0.5): # data_ch is expected to be (1, n_samples)
    """Compute PSD for a single channel"""
    # Expects data_ch to be 2D array like (1, n_samples)
    if data_ch.ndim == 1:
        data_ch = data_ch.reshape(1, -1)
    
    psd_val, freqs = psd_array_multitaper(
        data_ch, sfreq=sfreq, fmin=1.0, fmax=sfreq/2, # fmax can be higher if needed
        bandwidth=bandwidth, adaptive=False, low_bias=True,
        normalization='full', verbose=False
    )
    return psd_val[0], freqs # Return 1D PSD and freqs

def find_best_stim_peak_from_psd(psd_data, freqs, abs_prominence_thresh, min_freq_hz, rel_prom_thresh):
    """
    Finds the best candidate stimulation peak from a PSD.
    Selection criteria:
    1. Filters peaks by absolute prominence and minimum frequency.
    2. Filters remaining peaks by relative prominence.
    3. From these candidates, selects the one with the *highest relative prominence*.
       If tied, selects the one with lowest frequency among them.
    Returns: (frequency, relative_prominence, absolute_prominence) or (None, -1, -1)
    """
    peaks_indices, props = find_peaks(psd_data, prominence=abs_prominence_thresh)

    if len(peaks_indices) == 0:
        return None, -1.0, -1.0 # Ensure float for prominence

    candidate_freqs = freqs[peaks_indices]
    candidate_abs_proms = props['prominences']
    candidate_heights = psd_data[peaks_indices]
    
    # Ensure heights are positive to avoid division by zero or NaN for relative prominence
    candidate_heights[candidate_heights <= 1e-9] = 1e-9 # A small positive floor
    candidate_rel_proms = candidate_abs_proms / candidate_heights

    # 1. Filter by min_freq_hz
    mask_min_freq = candidate_freqs >= min_freq_hz
    if not np.any(mask_min_freq): return None, -1.0, -1.0
    
    candidate_freqs = candidate_freqs[mask_min_freq]
    candidate_abs_proms = candidate_abs_proms[mask_min_freq]
    candidate_rel_proms = candidate_rel_proms[mask_min_freq]

    # 2. Filter by rel_prom_thresh
    mask_rel_prom = candidate_rel_proms >= rel_prom_thresh
    if not np.any(mask_rel_prom): return None, -1.0, -1.0

    final_candidate_freqs = candidate_freqs[mask_rel_prom]
    final_candidate_abs_proms = candidate_abs_proms[mask_rel_prom]
    final_candidate_rel_proms = candidate_rel_proms[mask_rel_prom]

    # 3. Select best from final candidates
    if len(final_candidate_freqs) == 0: # Should be caught by np.any(mask_rel_prom)
        return None, -1.0, -1.0
    elif len(final_candidate_freqs) == 1:
        return final_candidate_freqs[0], final_candidate_rel_proms[0], final_candidate_abs_proms[0]
    else:
        # Max relative prominence
        max_rel_prom_val = np.max(final_candidate_rel_proms)
        indices_max_rel_prom = np.where(final_candidate_rel_proms == max_rel_prom_val)[0]

        if len(indices_max_rel_prom) == 1:
            idx = indices_max_rel_prom[0]
            return final_candidate_freqs[idx], final_candidate_rel_proms[idx], final_candidate_abs_proms[idx]
        else:
            # Tie in relative prominence, pick lowest frequency among them
            tied_freqs = final_candidate_freqs[indices_max_rel_prom]
            tied_rel_proms = final_candidate_rel_proms[indices_max_rel_prom]
            tied_abs_proms = final_candidate_abs_proms[indices_max_rel_prom]
            
            idx_lowest_freq_in_tie = np.argmin(tied_freqs)
            return tied_freqs[idx_lowest_freq_in_tie], tied_rel_proms[idx_lowest_freq_in_tie], tied_abs_proms[idx_lowest_freq_in_tie]


# --- Functions for template detection and pulse finding (largely unchanged from previous version) ---
def detect_stim_epochs(signal, sfreq, stim_freq):
    win_sec = max(0.2, round(sfreq/5000, 1))
    step_sec = win_sec / 2
    nperseg = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)

    segment_centers, segment_power = [], []
    for start in range(0, signal.size - nperseg + 1, step_samps):
        stop = start + nperseg
        segment = signal[start:stop]
        if len(segment) < nperseg and len(segment)>0: # Handle segments shorter than nperseg at the end
             freqs_w, psd_w = welch(segment, fs=sfreq, nperseg=len(segment))
        elif len(segment) == 0:
            continue
        else:
            freqs_w, psd_w = welch(segment, fs=sfreq, nperseg=nperseg)

        idx = np.argmin(np.abs(freqs_w - stim_freq))
        segment_centers.append((start + stop) / 2 / sfreq)
        total_power = np.sum(psd_w)
        rel_power = (psd_w[idx] / total_power) * 100 if total_power > 0 else 0
        segment_power.append(rel_power)

    segment_power = np.array(segment_power)
    segment_centers = np.array(segment_centers)

    if len(segment_power) == 0: raise ValueError("No segments processed for epoch detection")
    max_prom_val = segment_power.max()
    if max_prom_val <= 1e-9: # Check against small threshold instead of exactly 0
        #This means no clear power modulation at stim_freq was found across segments
        print("Warning: No significant stimulation activity detected by power modulation. Using full signal duration for analysis.")
        return 0, len(signal)/sfreq # Fallback to full duration

    max_idxs = np.where(segment_power == max_prom_val)[0]
    first_max, last_max = max_idxs.min(), max_idxs.max()
    drop_thresh = 0.1 * max_prom_val # 10% drop from max prominence

    start_idx_epoch = first_max
    # Ensure start_idx_epoch is within bounds before accessing segment_power
    while start_idx_epoch > 0 and segment_power[start_idx_epoch] >= drop_thresh: start_idx_epoch -= 1
    stim_start_time = segment_centers[start_idx_epoch]

    end_idx_epoch = last_max
    while end_idx_epoch < len(segment_power) - 1 and segment_power[end_idx_epoch] >= drop_thresh: end_idx_epoch += 1
    stim_end_time = segment_centers[end_idx_epoch]
    return stim_start_time, stim_end_time


def find_template_pulse(signal, sfreq, stim_freq, stim_start_time, stim_end_time):
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

    search_limit_in_stim_signal = min(len(stim_signal) - template_duration_samples + 1, 10 * period_samples)
    step = max(1, period_samples // 4 if period_samples > 0 else 1)


    for i in range(0, search_limit_in_stim_signal, step):
        segment_for_peak_search = stim_signal[i : i + template_duration_samples]
        if len(segment_for_peak_search) < template_duration_samples : continue

        peak_idx_in_segment = np.argmax(np.abs(segment_for_peak_search))
        peak_idx_relative_to_stim_signal = i + peak_idx_in_segment

        current_template_start_relative = peak_idx_relative_to_stim_signal - (template_duration_samples // 2)
        current_template_end_relative = current_template_start_relative + template_duration_samples

        if current_template_start_relative >= 0 and current_template_end_relative <= len(stim_signal):
            actual_template_candidate = stim_signal[current_template_start_relative : current_template_end_relative]

            if len(actual_template_candidate) == template_duration_samples:
                variance_score = np.var(actual_template_candidate)
                pp_score = np.ptp(actual_template_candidate)
                combined_score = variance_score * pp_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_template = actual_template_candidate.copy()
                    best_template_start_abs_idx = start_idx_stim_period + current_template_start_relative

    if best_template is None:
        if len(stim_signal) >= template_duration_samples:
            print("Warning: Template by peak centering failed. Taking first valid segment of stim_signal as template, attempting to center it.")
            first_segment = stim_signal[0:template_duration_samples]
            if len(first_segment)>0: # Ensure segment is not empty
                peak_idx_in_first_segment = np.argmax(np.abs(first_segment))
                current_template_start_relative = peak_idx_in_first_segment - (template_duration_samples // 2)
                current_template_end_relative = current_template_start_relative + template_duration_samples

                if current_template_start_relative >=0 and current_template_end_relative <= len(stim_signal):
                     best_template = stim_signal[current_template_start_relative:current_template_end_relative].copy()
                     best_template_start_abs_idx = start_idx_stim_period + current_template_start_relative
                else:
                     best_template = first_segment.copy()
                     best_template_start_abs_idx = start_idx_stim_period
            else: # Fallback if first_segment is empty for some reason
                 raise ValueError("Fallback template creation: first segment is empty.")

            if best_template is None or len(best_template)==0:
                 raise ValueError("Fallback template creation failed to produce a valid template.")
        else:
            raise ValueError("Could not find suitable template. Stimulation signal may be too short or flat.")

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

    if len(template)==0:
        print("Error: Empty template provided to cross_correlate_pulses.")
        return np.array([]), np.array([]), np.array([])

    template_std = np.std(template)
    template_norm = (template - np.mean(template)) / template_std if template_std > 1e-9 else template - np.mean(template)
    correlation = correlate(stim_signal_corr, template_norm, mode='valid')
    if len(correlation) == 0: return np.array([]), np.array([]), np.array([])

    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else len(template) # Fallback for period_samples
    min_distance_peaks = max(1, int(0.7 * period_samples)) if period_samples > 0 else 1
    correlation_threshold = np.percentile(correlation, 85) 
    peaks_in_corr, _ = find_peaks(correlation, height=correlation_threshold, distance=min_distance_peaks)

    pulse_starts_abs = peaks_in_corr + start_idx_corr
    pulse_ends_abs = pulse_starts_abs + len(template) - 1 
    correlation_scores = correlation[peaks_in_corr]
    print(f"Found {len(pulse_starts_abs)} pulse artifacts via cross-correlation")
    return pulse_starts_abs, pulse_ends_abs, correlation_scores

def refine_pulse_boundaries(signal, pulse_starts_initial, initial_pulse_ends_unused, sfreq, stim_freq):
    if len(pulse_starts_initial) == 0:
        return np.array([]), np.array([])

    refined_starts_list, refined_ends_list = [], []
    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else 0

    if period_samples <= 0:
        print(f"Warning: period_samples is {period_samples}. Cannot refine boundaries with peak centering based on this period.")
        if len(pulse_starts_initial) > 0 and initial_pulse_ends_unused is not None and len(initial_pulse_ends_unused) == len(pulse_starts_initial):
            return pulse_starts_initial, initial_pulse_ends_unused # Fallback to unrefined
        return np.array([]),np.array([])


    half_period_floor = period_samples // 2

    for i in range(len(pulse_starts_initial)):
        s_initial = pulse_starts_initial[i]
        search_win_start = s_initial
        search_win_end = s_initial + period_samples

        search_win_start_clipped = max(0, search_win_start)
        search_win_end_clipped = min(len(signal), search_win_end)

        if search_win_start_clipped >= search_win_end_clipped:
            search_win_start_clipped = max(0, s_initial - half_period_floor)
            search_win_end_clipped = min(len(signal), s_initial + period_samples - half_period_floor) # Centered around s_initial
            if search_win_start_clipped >= search_win_end_clipped:
                 print(f"  Skipping refinement for pulse at {s_initial}: invalid search window even after broadening.")
                 continue
        
        pulse_segment = signal[search_win_start_clipped : search_win_end_clipped]
        if len(pulse_segment) == 0: continue

        peak_idx_in_segment = np.argmax(np.abs(pulse_segment))
        global_peak_idx = search_win_start_clipped + peak_idx_in_segment

        new_start = global_peak_idx - half_period_floor
        new_end = new_start + period_samples - 1

        refined_starts_list.append(new_start)
        refined_ends_list.append(new_end)

    if not refined_starts_list:
        print("Warning: No pulses could be refined with peak centering.")
        return np.array([]), np.array([])

    final_refined_starts = np.array(refined_starts_list, dtype=int)
    final_refined_ends = np.array(refined_ends_list, dtype=int)

    final_refined_starts = np.clip(final_refined_starts, 0, len(signal) - 1)
    final_refined_ends = np.clip(final_refined_ends, 0, len(signal) - 1)

    valid_indices = (final_refined_starts <= final_refined_ends) & \
                    (final_refined_starts < (len(signal) - (1 if period_samples > 1 else 0)) )
    
    final_refined_starts = final_refined_starts[valid_indices]
    final_refined_ends = final_refined_ends[valid_indices]
    
    if len(final_refined_starts) < len(pulse_starts_initial):
        print(f"Info: Peak-centering resulted in {len(pulse_starts_initial) - len(final_refined_starts)} pulses being dropped.")
    return final_refined_starts, final_refined_ends

def visualize_pulse_detection(signal, sfreq, stim_freq, template, template_start_idx,
                            pulse_starts, pulse_ends, stim_start_time, stim_end_time):
    # This function remains largely the same as in the previous version,
    # ensuring it correctly uses the provided pulse_starts and pulse_ends (inclusive indices)
    # and stim_freq for annotations.
    times = np.arange(len(signal)) / sfreq
    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else (len(template) if len(template)>0 else 0)


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

    if len(template)>0:
        template_times = np.arange(len(template)) / sfreq
        axes[1].plot(template_times, template, 'g-', linewidth=2)
        axes[1].set_title(f'Template Pulse (from {template_start_idx/sfreq:.3f}s, duration {len(template)/sfreq:.4f}s)')
    else:
        axes[1].set_title('Template Pulse (Not Available)')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Amplitude'); axes[1].grid(True, alpha=0.3)


    stim_start_idx_viz = int(stim_start_time * sfreq)
    stim_end_idx_viz = int(stim_end_time * sfreq)
    padding_viz_samples = int(0.1 * max(0, (stim_end_idx_viz - stim_start_idx_viz)))
    stim_start_idx_viz = max(0, stim_start_idx_viz - padding_viz_samples)
    stim_end_idx_viz = min(len(signal), stim_end_idx_viz + padding_viz_samples)

    stim_signal_viz_slice = slice(stim_start_idx_viz, stim_end_idx_viz)
    stim_signal_for_plot = signal # Default to full signal
    if stim_start_idx_viz < stim_end_idx_viz and len(signal[stim_signal_viz_slice]) > 0:
        stim_times_viz = times[stim_signal_viz_slice]
        stim_signal_for_plot = signal[stim_signal_viz_slice]
        axes[2].plot(stim_times_viz, stim_signal_for_plot, 'b-', alpha=0.7, linewidth=0.8)
        axes[2].set_xlim(stim_times_viz[0], stim_times_viz[-1])
    else:
        axes[2].plot(times, signal, 'b-', alpha=0.7, linewidth=0.8)
        

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

    if len(pulse_starts) > 1 and stim_freq > 0 : # Check stim_freq for nominal interval
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
        axes[3].text(0.5, 0.5, 'Insufficient pulses or stim_freq for interval analysis', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Inter-Pulse Intervals - Not Available')
    plt.tight_layout(pad=1.5); plt.show()

    print("\n=== PULSE DETECTION SUMMARY ===")
    print(f"Total pulses detected: {len(pulse_starts)}")
    print(f"Stimulation period: {stim_start_time:.3f}s to {stim_end_time:.3f}s (Duration: {stim_end_time-stim_start_time:.3f}s)")
    if stim_freq > 0 and period_samples > 0:
        print(f"Nominal pulse width (1/stim_freq): {1/stim_freq:.4f}s ({period_samples} samples at {sfreq} Hz)")
    if len(pulse_starts) > 1 and mean_interval > 0: # mean_interval defined if len(pulse_starts)>1
        print(f"Mean inter-pulse interval (start-to-start): {mean_interval:.4f}s ± {np.std(inter_pulse_intervals_sec):.4f}s")
        if stim_freq > 0: print(f"  Est. freq from intervals: {1 / mean_interval:.2f} Hz (cf. input: {stim_freq:.2f} Hz)")
    if len(pulse_starts) > 0 and len(pulse_ends) > 0:
        actual_dur_samp = (pulse_ends - pulse_starts) + 1
        actual_dur_sec = actual_dur_samp / sfreq
        print(f"Effective pulse duration range (after centering & clipping): {np.min(actual_dur_sec):.4f}s to {np.max(actual_dur_sec):.4f}s")
        if period_samples > 0:
            num_non_nominal = np.sum(actual_dur_samp != period_samples)
            if num_non_nominal > 0: print(f"  ({num_non_nominal} pulses have durations not exactly {period_samples} samples, likely due to boundary clipping)")


def spline_artifact_extended_anchors(data, artifact_starts, artifact_ends, sfreq, buffer_ms=5.0):
    # This function remains largely the same as in the previous version.
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
            else: # Fallback anchor strategy
                x_anchor_start = max(0, start_artifact - 1)
                x_anchor_end = min(num_samples - 1, end_artifact + 1)
                if x_anchor_start < start_artifact and x_anchor_end > end_artifact and x_anchor_start < x_anchor_end:
                    x_known = np.array([x_anchor_start, x_anchor_end])
                    if not use_extended and buffer_samples > 0: print(f"  Info: Art {i_stim+1} fallback to immediate pre/post anchors.")
                else:
                    x_known = np.array([start_artifact, end_artifact]) 
                    if not use_extended and buffer_samples > 0: print(f"  Info: Art {i_stim+1} fallback to artifact boundary anchors.")
            
            if x_known[0] >= x_known[1] and not (x_known[0] == x_known[1] and start_artifact == end_artifact):
                 print(f"  Warn: Art {i_stim+1} anchors ({x_known[0]},{x_known[1]}) not distinct/increasing. Skip."); continue
            
            y_known = spline_data[i_chan, x_known]
            if np.any(np.isnan(y_known)): print(f"  Warn: Art {i_stim+1} NaN in y_known. Skip."); continue
            
            x_query = np.arange(start_artifact, end_artifact + 1)
            if len(x_query) == 0: continue

            try:
                if x_known[0] == x_known[1] and len(x_query) > 0 : 
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
        path = select_fif_file()
        if not path: print("No file selected. Exiting."); return None
        
        raw = mne.io.read_raw_fif(path, preload=True)
        sfreq = raw.info['sfreq']; results['sfreq'] = sfreq
        data = raw.get_data() # All channels data (channels x samples)
        ch_names = raw.ch_names
        if data.shape[0] == 0: raise ValueError("No data channels in file.")

        print(f"Loaded data with {data.shape[0]} channels. Sample frequency: {sfreq} Hz.")

        # --- Determine best channel and stimulation frequency ---
        best_channel_idx = -1
        overall_best_stim_freq = None
        max_rel_prominence_overall = -1.0

        # Parameters for finding stim peak in each channel's PSD
        # These might need tuning based on expected artifact characteristics
        psd_abs_prom_thresh = 5  # Absolute prominence for peak detection in PSD
        psd_min_freq_hz = 10     # Minimum frequency for a peak to be considered stimulation
        psd_rel_prom_thresh = 0.2 # Relative prominence threshold for a peak

        print(f"\nSearching for channel with clearest stimulation artifact (RelProm > {psd_rel_prom_thresh}):")
        for i_ch in range(data.shape[0]):
            print(f"  Analyzing Channel {i_ch} ({ch_names[i_ch]}):")
            channel_signal_temp = data[i_ch, :]
            
            # Ensure channel_signal_temp is not all zeros or constant
            if np.all(channel_signal_temp == channel_signal_temp[0]):
                print(f"    Channel {i_ch} is constant or all zeros. Skipping PSD analysis.")
                continue

            try:
                psd_ch, freqs_ch = compute_mean_psd(channel_signal_temp, sfreq) # Pass 1D array
                current_freq, current_rel_prom, current_abs_prom = find_best_stim_peak_from_psd(
                    psd_ch, freqs_ch, psd_abs_prom_thresh, psd_min_freq_hz, psd_rel_prom_thresh
                )
                if current_freq is not None:
                    print(f"    Found potential stim peak: {current_freq:.2f} Hz, RelProm: {current_rel_prom:.2f}, AbsProm: {current_abs_prom:.2f}")
                    if current_rel_prom > max_rel_prominence_overall:
                        max_rel_prominence_overall = current_rel_prom
                        overall_best_stim_freq = current_freq
                        best_channel_idx = i_ch
                else:
                    print(f"    No suitable stimulation peak found meeting criteria.")
            except Exception as e_psd:
                print(f"    Error during PSD analysis for channel {i_ch}: {e_psd}")
                continue
        
        if best_channel_idx == -1 or overall_best_stim_freq is None:
            # Fallback: ask user for channel index and stim frequency
            print("\nCould not automatically determine the best channel and stimulation frequency.")
            print("Please provide these values manually.")
            temp_root_dialog = tk.Tk(); temp_root_dialog.withdraw()
            try:
                best_channel_idx_str = simpledialog.askstring("Input", f"Enter channel index for analysis (0-{data.shape[0]-1}):", parent=temp_root_dialog)
                best_channel_idx = int(best_channel_idx_str) if best_channel_idx_str else 0
                if not (0 <= best_channel_idx < data.shape[0]):
                    print(f"Invalid channel index {best_channel_idx}, using 0."); best_channel_idx = 0

                overall_best_stim_freq_str = simpledialog.askstring("Input", "Enter estimated stimulation frequency (Hz):", parent=temp_root_dialog)
                overall_best_stim_freq = float(overall_best_stim_freq_str) if overall_best_stim_freq_str else 0
                if overall_best_stim_freq <=0 :
                     raise ValueError("Stimulation frequency must be positive.")
            except (ValueError, TypeError) as e_manual:
                 temp_root_dialog.destroy()
                 raise ValueError(f"Invalid manual input for channel or frequency: {e_manual}")
            temp_root_dialog.destroy()
            print(f"Using manually entered Channel: {best_channel_idx}, Stim Freq: {overall_best_stim_freq:.2f} Hz")


        print(f"\n--- Using Channel {best_channel_idx} ({ch_names[best_channel_idx]}) with Stored Stim Freq: {overall_best_stim_freq:.2f} Hz for artifact processing ---")
        signal_for_detection = data[best_channel_idx].copy()
        stim_freq = overall_best_stim_freq # Use the determined frequency
        results['stim_freq'] = stim_freq
        results['best_channel_idx_for_detection'] = best_channel_idx
        results['best_channel_name'] = ch_names[best_channel_idx]


        stim_start_t, stim_end_t = detect_stim_epochs(signal_for_detection, sfreq, stim_freq)
        results['stim_start_time'] = stim_start_t; results['stim_end_time'] = stim_end_t
        print(f"Stim period detected: {stim_start_t:.3f}s to {stim_end_t:.3f}s")

        template, template_start_idx, template_len = find_template_pulse(
            signal_for_detection, sfreq, stim_freq, stim_start_t, stim_end_t)
        results['template'] = template; results['template_start_idx'] = template_start_idx
        results['template_duration_samples'] = template_len

        p_starts, p_ends, corr_scores = cross_correlate_pulses(
            signal_for_detection, template, sfreq, stim_freq, stim_start_t, stim_end_t)
        results['correlation_scores'] = corr_scores
        if len(p_starts) == 0: print("No pulses from cross-correlation. Processing may be limited."); # Allow to continue if possible
        
        # Pass p_ends to refine_pulse_boundaries (though current logic might not use it)
        p_starts_ref, p_ends_ref = refine_pulse_boundaries(
            signal_for_detection, p_starts, p_ends if len(p_starts)>0 else np.array([]), sfreq, stim_freq)
        
        if len(p_starts_ref) == 0 and len(p_starts)>0: # If refinement removed all, consider using unrefined
            print("Warning: Refinement (peak-centering) removed all pulses. Consider using pre-refinement boundaries if appropriate for your data.")
            # Decide if fallback to p_starts, p_ends is desired here. For now, proceed with empty if refinement fails.
            # p_starts_ref, p_ends_ref = p_starts, p_ends # Optional fallback
        
        results['pulse_starts_refined'] = p_starts_ref 
        results['pulse_ends_refined'] = p_ends_ref     
        if len(p_starts_ref)>0:
            results['pulse_durations_sec'] = (p_ends_ref - p_starts_ref + 1) / sfreq
            results['pulse_times_sec'] = p_starts_ref / sfreq
        else:
            results['pulse_durations_sec'] = np.array([])
            results['pulse_times_sec'] = np.array([])


        visualize_pulse_detection(signal_for_detection, sfreq, stim_freq, template, template_start_idx,
                                p_starts_ref, p_ends_ref, stim_start_t, stim_end_t)

        if len(p_starts_ref) > 0: # Only spline if there are defined artifact boundaries
            print(f"\nApplying Spline Interpolation to ALL {data.shape[0]} channels based on {len(p_starts_ref)} pulses...")
            corrected_data_all = spline_artifact_extended_anchors(
                data, p_starts_ref, p_ends_ref, sfreq, buffer_ms=5.0) # Tunable buffer
            results['corrected_data_all_channels'] = corrected_data_all
            print("Spline interpolation complete for all channels.")

            print(f"Visualizing spline result for the analysis channel {best_channel_idx} ({ch_names[best_channel_idx]})...")
            original_sig_selected = data[best_channel_idx]
            # Handle 1D case if corrected_data_all is 1D (e.g. single channel input file)
            corrected_sig_selected = corrected_data_all[best_channel_idx] if corrected_data_all.ndim == 2 else corrected_data_all

            times_v = np.arange(len(original_sig_selected)) / sfreq
            plt.figure(figsize=(18, 7))
            
            vis_start_spline_s = max(0, (p_starts_ref[0] / sfreq) - 0.1) if len(p_starts_ref)>0 else stim_start_t -0.1
            num_pulses_to_show = min(len(p_starts_ref), 5) if len(p_starts_ref)>0 else 0
            vis_end_spline_s = min(times_v[-1] if len(times_v)>0 else (stim_end_t+0.1 if stim_end_t else 2.0), 
                                (p_ends_ref[num_pulses_to_show-1] / sfreq) + 0.1 if num_pulses_to_show > 0 else (stim_end_t+0.1 if stim_end_t else 2.0))


            plot_start_idx_s = max(0, int(vis_start_spline_s * sfreq))
            plot_end_idx_s = min(len(original_sig_selected), int(vis_end_spline_s * sfreq))
            
            if plot_start_idx_s >= plot_end_idx_s and len(original_sig_selected) > 0 :
                plot_start_idx_s = 0; plot_end_idx_s = min(len(original_sig_selected), int(sfreq * 2))


            plt.plot(times_v[plot_start_idx_s:plot_end_idx_s],
                     original_sig_selected[plot_start_idx_s:plot_end_idx_s],
                     label=f'Original Ch {best_channel_idx}', color='dimgray', alpha=0.7, lw=1.0)
            plt.plot(times_v[plot_start_idx_s:plot_end_idx_s],
                     corrected_sig_selected[plot_start_idx_s:plot_end_idx_s],
                     label=f'Corrected Ch {best_channel_idx}', color='dodgerblue', lw=1.2, alpha=0.9)

            first_label = True
            for i in range(len(p_starts_ref)): # Iterate through refined pulses
                s, e = p_starts_ref[i], p_ends_ref[i]
                seg_disp_start = max(plot_start_idx_s, s)
                seg_disp_end = min(plot_end_idx_s, e + 1) # +1 for slice
                if seg_disp_start < seg_disp_end: # If segment is visible
                    ts, ds = times_v[seg_disp_start:seg_disp_end], corrected_sig_selected[seg_disp_start:seg_disp_end]
                    if len(ts) > 0: # Ensure there's something to plot
                         plt.plot(ts, ds, color='red', lw=1.8, label='Interpolated' if first_label else "", zorder=5)
                         first_label = False
            plt.title(f'Spline Interpolation Effect on Channel {best_channel_idx} ({ch_names[best_channel_idx]})', fontsize=14)
            plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
            if plot_start_idx_s < plot_end_idx_s and len(times_v)>max(plot_start_idx_s, plot_end_idx_s-1 if plot_end_idx_s > 0 else 0):
                 plt.xlim(times_v[plot_start_idx_s], times_v[plot_end_idx_s-1 if plot_end_idx_s > 0 else 0])
            plt.legend(loc='upper right'); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.show()
        else:
            print("No refined pulses available to apply or visualize spline interpolation.")
            
        return results

    except Exception as e:
        print(f"Error in main: {str(e)}"); import traceback; traceback.print_exc(); return results

if __name__ == "__main__":
    analysis_results = main()
    if analysis_results and analysis_results.get('pulse_times_sec', np.array([])).size > 0:
        print("\n--- Analysis Summary ---")
        print(f"Analysis based on Channel: {analysis_results.get('best_channel_idx_for_detection', 'N/A')} ({analysis_results.get('best_channel_name', 'N/A')})")
        print(f"Determined Stim Freq: {analysis_results.get('stim_freq', 'N/A'):.2f} Hz")
        print(f"# Pulses refined: {len(analysis_results['pulse_starts_refined'])}")
        if len(analysis_results['pulse_times_sec']) > 0:
            print(f"First 5 pulse starts (s): {np.round(analysis_results['pulse_times_sec'][:5], 4)}")
            print(f"First 5 pulse durations (s): {np.round(analysis_results['pulse_durations_sec'][:5], 4)}")
        if 'corrected_data_all_channels' in analysis_results and analysis_results['corrected_data_all_channels'] is not None:
            print(f"Corrected data shape: {analysis_results['corrected_data_all_channels'].shape}")
        print("Pipeline completed.")
    elif analysis_results: print("\nPipeline completed with partial or no pulse results.")
    else: print("Pipeline failed or exited early.")