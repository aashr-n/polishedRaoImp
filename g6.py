# only use multitaper!!!


# Enhanced Stimulation Pulse Artifact Detection
# (incorporates user feedback for channel selection and stim frequency determination)

import mne
import scipy
import os
import numpy as np
from scipy.signal import welch, find_peaks, correlate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

# GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.simpledialog as simpledialog
from mne.time_frequency import psd_array_multitaper

# --- Utility functions ---

def select_fif_file():
    """Select FIF file using GUI dialog"""
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Select EEG FIF file", filetypes=[("FIF files", "*.fif"), ("All files", "*.*")])
    root.destroy()
    if not path: print("No file selected by user."); return None
    return path

def compute_single_channel_psd(channel_data_1d, sfreq, bandwidth=0.5):
    """Compute PSD for a single channel's 1D data."""
    if channel_data_1d.ndim != 1: raise ValueError("compute_single_channel_psd expects 1D array.")
    data_reshaped = channel_data_1d.reshape(1, -1) # MNE function expects 2D
    psd_val, freqs = psd_array_multitaper(
        data_reshaped, sfreq=sfreq, fmin=1.0, fmax=sfreq/2,
        bandwidth=bandwidth, adaptive=False, low_bias=True,
        normalization='full', verbose=False
    )
    return psd_val[0], freqs # Return 1D PSD

def find_best_stim_peak_from_psd(psd_data, freqs, abs_prominence_thresh, min_freq_hz, rel_prom_thresh):
    """Finds the best candidate stimulation peak from a PSD."""
    peaks_indices, props = find_peaks(psd_data, prominence=abs_prominence_thresh)
    if len(peaks_indices) == 0: return None, -1.0, -1.0

    candidate_freqs = freqs[peaks_indices]
    candidate_abs_proms = props['prominences']
    candidate_heights = psd_data[peaks_indices]
    candidate_heights[candidate_heights <= 1e-9] = 1e-9
    candidate_rel_proms = candidate_abs_proms / candidate_heights

    mask_min_freq = candidate_freqs >= min_freq_hz
    if not np.any(mask_min_freq): return None, -1.0, -1.0
    
    candidate_freqs, candidate_abs_proms, candidate_rel_proms = \
        candidate_freqs[mask_min_freq], candidate_abs_proms[mask_min_freq], candidate_rel_proms[mask_min_freq]

    mask_rel_prom = candidate_rel_proms >= rel_prom_thresh
    if not np.any(mask_rel_prom): return None, -1.0, -1.0

    final_candidate_freqs = candidate_freqs[mask_rel_prom]
    final_candidate_abs_proms = candidate_abs_proms[mask_rel_prom]
    final_candidate_rel_proms = candidate_rel_proms[mask_rel_prom]

    if len(final_candidate_freqs) == 0: return None, -1.0, -1.0
    if len(final_candidate_freqs) == 1:
        return final_candidate_freqs[0], final_candidate_rel_proms[0], final_candidate_abs_proms[0]
    else:
        max_rel_prom_val = np.max(final_candidate_rel_proms)
        indices_max_rel_prom = np.where(final_candidate_rel_proms == max_rel_prom_val)[0]
        if len(indices_max_rel_prom) == 1:
            idx = indices_max_rel_prom[0]
            return final_candidate_freqs[idx], final_candidate_rel_proms[idx], final_candidate_abs_proms[idx]
        else: # Tie in relative prominence, pick lowest frequency
            idx_lowest_freq_in_tie = np.argmin(final_candidate_freqs[indices_max_rel_prom])
            actual_idx = indices_max_rel_prom[idx_lowest_freq_in_tie]
            return final_candidate_freqs[actual_idx], final_candidate_rel_proms[actual_idx], final_candidate_abs_proms[actual_idx]

def get_peak_properties_at_freq(psd_data, freqs, target_freq, freq_tolerance, min_abs_prominence):
    """
    Finds a peak in psd_data near target_freq and returns its properties,
    prioritizing highest relative prominence if multiple peaks are in the window.
    """
    peaks_indices, props = find_peaks(psd_data, prominence=min_abs_prominence)
    if len(peaks_indices) == 0: return -1.0, None, -1.0

    found_peak_freqs = freqs[peaks_indices]
    found_peak_abs_proms = props['prominences']
    found_peak_heights = psd_data[peaks_indices]
    found_peak_heights[found_peak_heights <= 1e-9] = 1e-9
    found_peak_rel_proms = found_peak_abs_proms / found_peak_heights

    freq_diff = np.abs(found_peak_freqs - target_freq)
    peaks_in_window_mask = freq_diff <= freq_tolerance
    
    if not np.any(peaks_in_window_mask): return -1.0, None, -1.0

    freqs_in_window = found_peak_freqs[peaks_in_window_mask]
    rel_proms_in_window = found_peak_rel_proms[peaks_in_window_mask]
    abs_proms_in_window = found_peak_abs_proms[peaks_in_window_mask]
    
    best_idx_in_window = np.argmax(rel_proms_in_window) # Highest relative prominence in window
    
    return rel_proms_in_window[best_idx_in_window], freqs_in_window[best_idx_in_window], abs_proms_in_window[best_idx_in_window]


# --- Functions for template detection, pulse finding, visualization, spline (largely unchanged) ---
def detect_stim_epochs(signal, sfreq, stim_freq):
    win_sec = max(0.2, round(sfreq/5000, 1))
    step_sec = win_sec / 2
    nperseg = int(win_sec * sfreq)
    if nperseg < 3: # Ensure nperseg is valid for PSD computation
        print(f"Warning: nperseg ({nperseg}) too small for welch/multitaper. Adjusting to 3 or sfreq/2 if very low sfreq.")
        nperseg = max(3, int(sfreq / 2)) # Smallest possible nperseg or half sfreq
        win_sec = nperseg / sfreq
        step_sec = win_sec / 2

    step_samps = int(step_sec * sfreq)

    segment_centers, segment_power = [], []
    for start in range(0, signal.size - nperseg + 1, step_samps):
        stop = start + nperseg
        segment = signal[start:stop]
        
        # Use MNE's multitaper PSD for consistency and robustness
        # Reshape segment to (1, n_samples) for psd_array_multitaper
        segment_reshaped = segment.reshape(1, -1)
        psd_w, freqs_w = psd_array_multitaper(
            segment_reshaped, sfreq=sfreq, fmin=1.0, fmax=sfreq/2,
            bandwidth=5.0, adaptive=True, low_bias=True, # Bandwidth of 5Hz, adaptive for robustness
            normalization='full', verbose=False
        )
        psd_w = psd_w[0] # Get 1D PSD array

        idx = np.argmin(np.abs(freqs_w - stim_freq))
        segment_centers.append((start + stop) / 2 / sfreq)
        total_power = np.sum(psd_w)
        rel_power = (psd_w[idx] / total_power) * 100 if total_power > 0 else 0
        segment_power.append(rel_power)

    segment_power = np.array(segment_power)
    segment_centers = np.array(segment_centers)

    if len(segment_power) == 0: raise ValueError("No segments processed for epoch detection")
    max_prom_val = segment_power.max()
    if max_prom_val <= 1e-9: 
        print("Warning: No significant stimulation activity detected by power modulation. Using full signal duration.")
        return 0, len(signal)/sfreq 

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
    start_idx_stim_period = int(stim_start_time * sfreq)
    end_idx_stim_period = int(stim_end_time * sfreq)
    start_idx_stim_period = max(0, start_idx_stim_period)
    end_idx_stim_period = min(len(signal), end_idx_stim_period)

    if start_idx_stim_period >= end_idx_stim_period:
        raise ValueError("Invalid stimulation period indices for template search.")

    stim_signal = signal[start_idx_stim_period:end_idx_stim_period]
    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else 0
    if period_samples <= 0:
        raise ValueError(f"Period_samples must be positive. Got {period_samples} from sfreq={sfreq}, stim_freq={stim_freq}.")

    template_duration_samples = period_samples

    if template_duration_samples > len(stim_signal):
        print(f"Warning: stim_signal (len {len(stim_signal)}) shorter than template_duration ({template_duration_samples}). Attempting to use full stim_signal.")
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
            if len(first_segment)>0: 
                peak_idx_in_first_segment = np.argmax(np.abs(first_segment))
                current_template_start_relative = peak_idx_in_first_segment - (template_duration_samples // 2)
                current_template_end_relative = current_template_start_relative + template_duration_samples
                if current_template_start_relative >=0 and current_template_end_relative <= len(stim_signal):
                     best_template = stim_signal[current_template_start_relative:current_template_end_relative].copy()
                     best_template_start_abs_idx = start_idx_stim_period + current_template_start_relative
                else: # Fallback if centering pushes out of bounds
                     best_template = first_segment.copy()
                     best_template_start_abs_idx = start_idx_stim_period
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

    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else len(template)
    min_distance_peaks = max(1, int(0.7 * period_samples)) if period_samples > 0 else 1
    correlation_threshold = np.percentile(correlation, 85) 
    peaks_in_corr, _ = find_peaks(correlation, height=correlation_threshold, distance=min_distance_peaks)

    pulse_starts_abs = peaks_in_corr + start_idx_corr
    pulse_ends_abs = pulse_starts_abs + len(template) - 1 
    correlation_scores = correlation[peaks_in_corr]
    print(f"Found {len(pulse_starts_abs)} pulse artifacts via cross-correlation")
    return pulse_starts_abs, pulse_ends_abs, correlation_scores

def refine_pulse_boundaries(signal, pulse_starts_initial, initial_pulse_ends_unused, sfreq, stim_freq):
    if len(pulse_starts_initial) == 0: return np.array([]), np.array([])
    refined_starts_list, refined_ends_list = [], []
    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else 0

    if period_samples <= 0:
        print(f"Warning: period_samples is {period_samples}. Cannot refine boundaries with peak centering.")
        if len(pulse_starts_initial)>0 and initial_pulse_ends_unused is not None and len(initial_pulse_ends_unused) == len(pulse_starts_initial):
            return pulse_starts_initial, initial_pulse_ends_unused 
        return np.array([]),np.array([])

    half_period_floor = period_samples // 2
    for i in range(len(pulse_starts_initial)):
        s_initial = pulse_starts_initial[i]
        search_win_start, search_win_end = s_initial, s_initial + period_samples
        search_win_start_clipped, search_win_end_clipped = max(0, search_win_start), min(len(signal), search_win_end)

        if search_win_start_clipped >= search_win_end_clipped:
            search_win_start_clipped = max(0, s_initial - half_period_floor)
            search_win_end_clipped = min(len(signal), s_initial + period_samples - half_period_floor)
            if search_win_start_clipped >= search_win_end_clipped:
                 print(f"  Skipping refinement for pulse at {s_initial}: invalid search window."); continue
        
        pulse_segment = signal[search_win_start_clipped : search_win_end_clipped]
        if len(pulse_segment) == 0: continue

        peak_idx_in_segment = np.argmax(np.abs(pulse_segment))
        global_peak_idx = search_win_start_clipped + peak_idx_in_segment
        new_start, new_end = global_peak_idx - half_period_floor, global_peak_idx - half_period_floor + period_samples - 1
        refined_starts_list.append(new_start); refined_ends_list.append(new_end)

    if not refined_starts_list: print("Warning: No pulses could be refined."); return np.array([]), np.array([])
    final_refined_starts = np.array(refined_starts_list, dtype=int)
    final_refined_ends = np.array(refined_ends_list, dtype=int)
    final_refined_starts = np.clip(final_refined_starts, 0, len(signal) - 1)
    final_refined_ends = np.clip(final_refined_ends, 0, len(signal) - 1)
    valid_indices = (final_refined_starts <= final_refined_ends) & (final_refined_starts < (len(signal) - (1 if period_samples > 1 else 0)) )
    final_refined_starts, final_refined_ends = final_refined_starts[valid_indices], final_refined_ends[valid_indices]
    if len(final_refined_starts) < len(pulse_starts_initial):
        print(f"Info: Peak-centering resulted in {len(pulse_starts_initial) - len(final_refined_starts)} pulses being dropped.")
    return final_refined_starts, final_refined_ends

def visualize_pulse_detection(signal, sfreq, stim_freq, template, template_start_idx,
                            pulse_starts, pulse_ends, stim_start_time, stim_end_time):
    times = np.arange(len(signal)) / sfreq
    period_samples = int(sfreq / stim_freq) if stim_freq > 0 else (len(template) if len(template)>0 else 0)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=False)
    axes[0].plot(times, signal, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axvspan(stim_start_time, stim_end_time, color='red', alpha=0.2, label=f'Stim Period ({stim_start_time:.2f}s - {stim_end_time:.2f}s)')
    axes[0].set_title('Full Signal with Stimulation Period'); axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right'); axes[0].grid(True, alpha=0.3)
    if stim_end_time > 0 and (stim_end_time - stim_start_time) < 0.3 * (times[-1] if len(times)>0 else 0) :
        plot_start_time_ax0 = max(0, stim_start_time - 0.1*(stim_end_time - stim_start_time))
        plot_end_time_ax0 = min(times[-1] if len(times)>0 else stim_end_time, stim_end_time + 0.1*(stim_end_time - stim_start_time))
        if plot_start_time_ax0 < plot_end_time_ax0: axes[0].set_xlim(plot_start_time_ax0, plot_end_time_ax0)

    if len(template)>0:
        template_times = np.arange(len(template)) / sfreq
        axes[1].plot(template_times, template, 'g-', linewidth=2)
        axes[1].set_title(f'Template Pulse (from {template_start_idx/sfreq:.3f}s, dur {len(template)/sfreq:.4f}s)')
    else: axes[1].set_title('Template Pulse (Not Available)')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Amplitude'); axes[1].grid(True, alpha=0.3)

    stim_sidx_viz = int(stim_start_time*sfreq); stim_eidx_viz = int(stim_end_time*sfreq)
    pad_viz = int(0.1*max(0,(stim_eidx_viz-stim_sidx_viz)))
    stim_sidx_viz=max(0,stim_sidx_viz-pad_viz); stim_eidx_viz=min(len(signal),stim_eidx_viz+pad_viz)
    viz_slice = slice(stim_sidx_viz, stim_eidx_viz)
    stim_sig_plot = signal
    if stim_sidx_viz < stim_eidx_viz and len(signal[viz_slice]) > 0:
        stim_t_viz = times[viz_slice]; stim_sig_plot = signal[viz_slice]
        axes[2].plot(stim_t_viz, stim_sig_plot, 'b-', alpha=0.7, lw=0.8); axes[2].set_xlim(stim_t_viz[0], stim_t_viz[-1])
    else: axes[2].plot(times, signal, 'b-', alpha=0.7, lw=0.8)
        
    if len(pulse_starts) > 0 and len(pulse_ends) > 0:
        colors = plt.cm.viridis(np.linspace(0,1,len(pulse_starts)))
        for i in range(len(pulse_starts)):
            s,e=pulse_starts[i],pulse_ends[i]; s_t,e_t_vspan=s/sfreq,(e+1)/sfreq
            axes[2].axvspan(s_t,e_t_vspan,color=colors[i],alpha=0.35)
            mid_t_txt=(s+0.5*(e-s))/sfreq; txt_y_idx=int(mid_t_txt*sfreq)
            txt_y_val=signal[txt_y_idx] if 0<=txt_y_idx<len(signal) else (np.max(stim_sig_plot)*0.9 if len(stim_sig_plot)>0 else 0)
            axes[2].text(mid_t_txt,txt_y_val,str(i+1),ha='center',va='bottom',fontsize=7,color='k',bbox=dict(boxstyle='round,pad=0.15',fc='w',alpha=0.7,lw=0.5))
    axes[2].set_title(f'Detected Pulses ({len(pulse_starts)})'); axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Amplitude'); axes[2].grid(True,alpha=0.3)

    mean_interval = -1
    if len(pulse_starts) > 1 and stim_freq > 0:
        inter_pulse_s = np.diff(pulse_starts/sfreq); mean_interval = np.mean(inter_pulse_s)
        axes[3].plot(pulse_starts[:-1]/sfreq, inter_pulse_s, 'ro-', ms=4, lw=1)
        axes[3].axhline(y=mean_interval, color='g', ls='--', label=f'Mean Interval: {mean_interval:.4f}s')
        axes[3].axhline(y=1/stim_freq, color='purple', ls=':', label=f'Nominal (1/stim_freq): {1/stim_freq:.4f}s')
        axes[3].set_title('Inter-Pulse Intervals'); axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('Interval (s)')
        axes[3].legend(loc='upper right'); axes[3].grid(True,alpha=0.3)
        if stim_sidx_viz < stim_eidx_viz and len(signal[viz_slice]) > 0: axes[3].set_xlim(stim_t_viz[0], stim_t_viz[-1])
    else: axes[3].text(0.5,0.5,'Insufficient data for interval analysis',ha='center',va='center',transform=axes[3].transAxes); axes[3].set_title('Inter-Pulse Intervals - N/A')
    plt.tight_layout(pad=1.5); plt.show()

    print("\n=== PULSE DETECTION SUMMARY ==="); print(f"Total pulses: {len(pulse_starts)}")
    print(f"Stim period: {stim_start_time:.3f}s to {stim_end_time:.3f}s (Dur: {stim_end_time-stim_start_time:.3f}s)")
    if stim_freq>0 and period_samples>0: print(f"Nominal pulse width (1/stim_freq): {1/stim_freq:.4f}s ({period_samples} samples @ {sfreq}Hz)")
    if mean_interval > 0: print(f"Mean inter-pulse interval: {mean_interval:.4f}s ± {np.std(inter_pulse_s):.4f}s"); print(f"  Est. freq from intervals: {1/mean_interval:.2f}Hz (cf. input: {stim_freq:.2f}Hz)")
    if len(pulse_starts)>0 and len(pulse_ends)>0:
        actual_dur_s = ((pulse_ends-pulse_starts)+1)/sfreq
        print(f"Effective pulse duration range: {np.min(actual_dur_s):.4f}s to {np.max(actual_dur_s):.4f}s")
        if period_samples>0: num_non_nom=np.sum(((pulse_ends-pulse_starts)+1)!=period_samples); print(f"  ({num_non_nom} pulses not exactly nominal duration due to clipping)")


def spline_artifact_extended_anchors(data, artifact_starts, artifact_ends, sfreq, buffer_ms=5.0):
    if not isinstance(data, np.ndarray): raise TypeError("Data must be NumPy array.")
    is_1d_input = data.ndim == 1
    spline_data = data.reshape(1, -1).copy() if is_1d_input else data.copy()
    num_ch, num_samp = spline_data.shape; buf_samp = int(buffer_ms/1000.0*sfreq)
    print(f"Spline: Buffer {buf_samp} samples ({buffer_ms}ms @ {sfreq}Hz)")
    for i_ch in range(num_ch):
        print(f"Spline Ch {i_ch+1}/{num_ch}")
        if num_samp>0 and np.isnan(spline_data[i_ch,0]): print(f"  Ch {i_ch+1} NaN start. Skip."); continue
        if len(artifact_starts)==0: continue
        for i_stim in range(len(artifact_starts)):
            s_art,e_art=int(artifact_starts[i_stim]),int(artifact_ends[i_stim])
            if not (0<=s_art<num_samp and 0<=e_art<num_samp and s_art<=e_art): print(f"  Warn: Art {i_stim+1} invalid bounds ({s_art}-{e_art}). Skip."); continue
            a1_id,a2_id=s_art-buf_samp,e_art+buf_samp; a1_cl,a2_cl=max(0,a1_id),min(num_samp-1,a2_id)
            use_ext=buf_samp>0 and a1_cl<s_art and a2_cl>e_art and a1_cl<a2_cl
            if use_ext: x_k=np.array([a1_cl,a2_cl])
            else:
                x_a_s,x_a_e=max(0,s_art-1),min(num_samp-1,e_art+1)
                if x_a_s<s_art and x_a_e>e_art and x_a_s<x_a_e: x_k=np.array([x_a_s,x_a_e])
                else: x_k=np.array([s_art,e_art])
            if x_k[0]>=x_k[1] and not (x_k[0]==x_k[1] and s_art==e_art): print(f"  Warn: Art {i_stim+1} anchors ({x_k[0]},{x_k[1]}) invalid. Skip."); continue
            y_k=spline_data[i_ch,x_k]; x_q=np.arange(s_art,e_art+1)
            if np.any(np.isnan(y_k)): print(f"  Warn: Art {i_stim+1} NaN y_known. Skip."); continue
            if len(x_q)==0: continue
            try:
                if x_k[0]==x_k[1] and len(x_q)>0: vals=np.full(len(x_q),y_k[0])
                else: vals=CubicSpline(x_k,y_k,bc_type='not-a-knot')(x_q)
                spline_data[i_ch,s_art:e_art+1]=vals
            except ValueError as e: print(f"  Error: Art {i_stim+1} spline ({s_art}-{e_art}): {e}. Skip."); continue
    return spline_data.squeeze() if is_1d_input else spline_data

def main():
    results = {}
    # --- Main Pipeline ---
    # 0. Initialize results dictionary.
    # 1. File Selection:
    #    - User selects a .fif file.
    # 2. Load Data:
    #    - Read raw data, sfreq, and channel names from the .fif file.
    # 3. PSD Computation for All Channels:
    #    - Call `_compute_all_channel_psds_and_mean` to get:
    #        - PSD for each channel.
    #        - Mean PSD across all valid channels.
    #        - Frequency axis.
    # 4. Initial Stimulation Frequency Estimation:
    #    - Call `_estimate_initial_stim_freq` using the mean PSD.
    #    - Configuration: `abs_prom`, `min_freq`, `rel_prom`.
    # 5. Find Clearest Channel for Stimulation Artifact:
    #    - If an initial stim frequency was found:
    #        - Call `_find_clearest_channel_for_stim` using all channel PSDs and the initial stim frequency.
    #        - Configuration: `freq_tolerance_hz`, `min_abs_prom_check`.
    #    - Stores `best_ch_idx` and `freq_at_best_ch`.
    # 6. Refine Stimulation Frequency from Best Channel:
    #    - If a best channel was identified:
    #        - Call `_refine_stim_freq_from_best_channel` using the PSD of the best channel and `freq_at_best_ch`.
    #        - Configuration: `abs_prom`, `rel_prom`, `freq_window_hz`.
    #    - Stores `final_stim_freq`.
    # 7. Fallback to Manual Input:
    #    - If `final_stim_freq` or `best_ch_idx` could not be determined automatically:
    #        - Call `_get_manual_stim_input` to prompt the user for channel index and stimulation frequency.
    #        - Uses previous estimates as defaults if available.
    # 8. Prepare for Artifact Processing:
    #    - Select the data from the `best_ch_idx` (`signal_for_detection`).
    #    - Use `final_stim_freq` as `stim_freq_to_use`.
    #    - Store these in the `results` dictionary.
    # 9. Detect Stimulation Epochs, Find Template, Cross-Correlate, Refine Pulses, Visualize, Spline Interpolate.
    try:
        path = select_fif_file()
        if not path: return None
        
        raw = mne.io.read_raw_fif(path, preload=True)
        print(f"Successfully loaded data from {path}")
        sfreq = raw.info['sfreq']; results['sfreq'] = sfreq
        data_all_channels = raw.get_data() 
        ch_names = raw.ch_names
        if data_all_channels.shape[0] == 0: raise ValueError("No data channels in file.")
        print(f"Loaded data: {data_all_channels.shape[0]} channels, {data_all_channels.shape[1]} samples. SFreq: {sfreq} Hz.")

        # --- Step 1 & 2: Calculate all channel PSDs and Mean PSD ---
        all_channel_psds_list = []
        common_freqs_axis = None
        print("\nCalculating PSD for all channels...")
        for i_ch in range(data_all_channels.shape[0]):
            if np.all(data_all_channels[i_ch,:] == data_all_channels[i_ch,0]): # Skip constant channels
                print(f"  Skipping Channel {i_ch} ({ch_names[i_ch]}): data is constant.")
                all_channel_psds_list.append(np.zeros_like(common_freqs_axis) if common_freqs_axis is not None else None) # Placeholder
                continue
            try:
                psd_ch, freqs_ch = compute_single_channel_psd(data_all_channels[i_ch,:], sfreq)
                if common_freqs_axis is None: common_freqs_axis = freqs_ch
                all_channel_psds_list.append(psd_ch)
            except Exception as e_psd_ch:
                 print(f"  Error computing PSD for channel {i_ch} ({ch_names[i_ch]}): {e_psd_ch}")
                 all_channel_psds_list.append(np.zeros_like(common_freqs_axis) if common_freqs_axis is not None else None)

        # Filter out None placeholders before averaging
        valid_psds = [psd for psd in all_channel_psds_list if psd is not None and len(psd) == len(common_freqs_axis)]
        if not valid_psds: raise ValueError("Could not compute PSD for any channel.")
        
        mean_psd_overall = np.mean(np.array(valid_psds), axis=0)

        # --- Step 3: Initial Stim Freq Estimation from Mean PSD ---
        # Tunable parameters for initial estimation from mean PSD
        abs_prom_mean_psd = 5 
        # Note: min_freq_mean_psd is set to 20Hz here, which might be too high for some stimulation frequencies. Consider lowering this if needed.
        min_freq_mean_psd = 20 # Look for stim freqs above 20Hz typically
        rel_prom_mean_psd = 0.3 
        print(f"\nEstimating initial stim frequency from Mean PSD (AbsProm>{abs_prom_mean_psd}, RelProm>{rel_prom_mean_psd}, MinFreq>{min_freq_mean_psd}Hz)...")
        
        initial_stim_freq_estimate, init_rel_prom, init_abs_prom = find_best_stim_peak_from_psd(
            mean_psd_overall, common_freqs_axis, abs_prom_mean_psd, min_freq_mean_psd, rel_prom_mean_psd
        )
        if initial_stim_freq_estimate is None:
            print("Could not estimate initial stimulation frequency from mean PSD. Falling back to manual input.")
            # Fallback handled further down
        else:
            print(f"  Initial Estimated Stim Freq (from Mean PSD): {initial_stim_freq_estimate:.2f} Hz (RelProm: {init_rel_prom:.2f})")

        # --- Step 4 & 5: Find Clearest Channel using Initial Stim Freq Estimate ---
        best_channel_idx_for_final_sf = -1
        max_rel_prom_at_initial_sf = -1.0
        freq_at_max_rel_prom = None

        print("\nSearching for channel with clearest stimulation artifact...")
        if initial_stim_freq_estimate is not None:
            print(f"\nIdentifying clearest channel based on relative prominence around {initial_stim_freq_estimate:.2f} Hz...")
            # Parameters for checking individual channels at the estimated frequency
            freq_tolerance_hz = 5.0 # Hz, how far from initial_stim_freq_estimate to look
            min_abs_prom_for_check = 1 # Lower threshold for this check, as we're targeted

            for i_ch in range(len(all_channel_psds_list)):
                psd_ch_i = all_channel_psds_list[i_ch]
                if psd_ch_i is None : continue # Skip if PSD failed for this channel

                rel_prom_ch, actual_peak_freq_ch, _ = get_peak_properties_at_freq(
                    psd_ch_i, common_freqs_axis, initial_stim_freq_estimate, freq_tolerance_hz, min_abs_prom_for_check
                )
                if actual_peak_freq_ch is not None and rel_prom_ch > max_rel_prom_at_initial_sf:
                    max_rel_prom_at_initial_sf = rel_prom_ch
                    best_channel_idx_for_final_sf = i_ch
                    freq_at_max_rel_prom = actual_peak_freq_ch # Store the actual freq of this best peak
                    print(f"  Channel {i_ch} ({ch_names[i_ch]}): Peak at {actual_peak_freq_ch:.2f}Hz, RelProm={rel_prom_ch:.2f} <-- New Best")
                elif actual_peak_freq_ch is not None:
                     print(f"  Channel {i_ch} ({ch_names[i_ch]}): Peak at {actual_peak_freq_ch:.2f}Hz, RelProm={rel_prom_ch:.2f}")


        # --- Step 6: Recalculate Definitive Stim Freq from Clearest Channel ---
        final_stim_freq = None
        print("\nRefining stimulation frequency estimate...")
        if best_channel_idx_for_final_sf != -1:
            print(f"\nRecalculating stim frequency from clearest channel: {best_channel_idx_for_final_sf} ({ch_names[best_channel_idx_for_final_sf]})")
            clearest_channel_psd = all_channel_psds_list[best_channel_idx_for_final_sf]
            # Parameters for final stim freq determination on the clearest channel
            abs_prom_final = 10 # Can be more stringent
            min_freq_final = max(5, freq_at_max_rel_prom - 10 if freq_at_max_rel_prom else 5) # Focus around found peak
            max_freq_final = freq_at_max_rel_prom + 10 if freq_at_max_rel_prom else sfreq/2
            rel_prom_final = 0.3 # Or higher
            
            # We need find_best_stim_peak_from_psd to also accept a max_freq_hz parameter, or filter frequencies before calling
            # For now, use existing min_freq_final.
            temp_freqs = common_freqs_axis[(common_freqs_axis >= min_freq_final) & (common_freqs_axis <= max_freq_final)]
            temp_psd = clearest_channel_psd[(common_freqs_axis >= min_freq_final) & (common_freqs_axis <= max_freq_final)]

            if len(temp_psd) > 0:
                final_stim_freq, final_rel_prom, _ = find_best_stim_peak_from_psd(
                    temp_psd, temp_freqs, abs_prom_final, 0, rel_prom_final # min_freq is 0 as temp_freqs is already filtered
                )
                if final_stim_freq:
                    print(f"  Definitive Stim Freq (from Ch {best_channel_idx_for_final_sf}): {final_stim_freq:.2f} Hz (RelProm: {final_rel_prom:.2f})")
            else:
                print(f"  Could not extract focused PSD for final stim freq calculation on Ch {best_channel_idx_for_final_sf}.")

        # --- Fallback to Manual Input if any step failed ---
        if final_stim_freq is None or best_channel_idx_for_final_sf == -1:
            print("\nAutomatic stimulation frequency/channel detection failed or yielded no result.")
            print("Please provide these values manually.")
            try:
                with tk.Tk() as temp_root_dialog: temp_root_dialog.withdraw() # Ensure proper Tk context
                best_ch_idx_str = simpledialog.askstring("Input", f"Enter channel index for analysis (0-{data_all_channels.shape[0]-1}):", parent=temp_root_dialog)
                best_channel_idx_for_final_sf = int(best_ch_idx_str) if best_ch_idx_str else 0
                if not (0 <= best_channel_idx_for_final_sf < data_all_channels.shape[0]):
                    print(f"Invalid channel index {best_channel_idx_for_final_sf}, using 0."); best_channel_idx_for_final_sf = 0

                final_stim_freq_str = simpledialog.askstring("Input", "Enter stimulation frequency (Hz):", parent=temp_root_dialog)
                final_stim_freq = float(final_stim_freq_str) if final_stim_freq_str else 0.0
                if final_stim_freq <=0 : raise ValueError("Stimulation frequency must be positive.")
                print(f"Using Manually Entered - Channel: {best_channel_idx_for_final_sf}, Stim Freq: {final_stim_freq:.2f} Hz")
            except Exception as e_manual: # Catch any error from dialog interaction
                 raise ValueError(f"Invalid manual input for channel or frequency: {e_manual}")

        # --- Proceed with pipeline using determined channel and frequency ---
        print(f"\n--- Using Channel {best_channel_idx_for_final_sf} ({ch_names[best_channel_idx_for_final_sf]}) & Stim Freq: {final_stim_freq:.2f} Hz for processing ---")
        signal_for_detection = data_all_channels[best_channel_idx_for_final_sf].copy()
        stim_freq_to_use = final_stim_freq
        
        results['stim_freq'] = stim_freq_to_use
        results['analysis_channel_idx'] = best_channel_idx_for_final_sf
        results['analysis_channel_name'] = ch_names[best_channel_idx_for_final_sf]

        print("\nDetecting stimulation epochs...")
        stim_start_t, stim_end_t = detect_stim_epochs(signal_for_detection, sfreq, stim_freq_to_use)
        results['stim_start_time'] = stim_start_t; results['stim_end_time'] = stim_end_t
        print(f"Stim period detected: {stim_start_t:.3f}s to {stim_end_t:.3f}s")

        template, template_start_idx, template_len = find_template_pulse(
            signal_for_detection, sfreq, stim_freq_to_use, stim_start_t, stim_end_t)
        results['template'] = template; results['template_start_idx'] = template_start_idx
        results['template_duration_samples'] = template_len

        print("\nPerforming cross-correlation to find pulses...")
        p_starts, p_ends, corr_scores = cross_correlate_pulses(
            signal_for_detection, template, sfreq, stim_freq_to_use, stim_start_t, stim_end_t)
        results['correlation_scores'] = corr_scores
        
        print("\nRefining pulse boundaries...")
        p_starts_ref, p_ends_ref = refine_pulse_boundaries(
            signal_for_detection, p_starts, p_ends if len(p_starts)>0 else np.array([]), sfreq, stim_freq_to_use)
        
        if len(p_starts_ref) == 0 and len(p_starts)>0: 
            print("Warning: Refinement removed all pulses. Using unrefined for visualization/spline.")
            p_starts_ref, p_ends_ref = p_starts, p_ends
        
        results['pulse_starts_refined'] = p_starts_ref 
        results['pulse_ends_refined'] = p_ends_ref     
        if len(p_starts_ref)>0:
            results['pulse_durations_sec'] = (p_ends_ref - p_starts_ref + 1) / sfreq
            results['pulse_times_sec'] = p_starts_ref / sfreq
        else: results['pulse_durations_sec'] = np.array([]); results['pulse_times_sec'] = np.array([])

        visualize_pulse_detection(signal_for_detection, sfreq, stim_freq_to_use, template, template_start_idx,
                                # Note: The visualization function expects stim_freq as an argument now.
                                p_starts_ref, p_ends_ref, stim_start_t, stim_end_t)

        if len(p_starts_ref) > 0:
            print(f"\nApplying Spline Interpolation to ALL {data_all_channels.shape[0]} channels based on {len(p_starts_ref)} pulses...")
            corrected_data_all = spline_artifact_extended_anchors(
                data_all_channels, p_starts_ref, p_ends_ref, sfreq, buffer_ms=5.0)
            results['corrected_data_all_channels'] = corrected_data_all
            print("Spline interpolation complete.")
            
            print(f"Visualizing spline result for analysis channel {best_channel_idx_for_final_sf}...")
            original_sig_selected = data_all_channels[best_channel_idx_for_final_sf]
            corrected_sig_selected = corrected_data_all[best_channel_idx_for_final_sf] if corrected_data_all.ndim==2 else corrected_data_all

            times_v = np.arange(len(original_sig_selected)) / sfreq
            plt.figure(figsize=(18, 7))
            vis_s_spline = max(0,(p_starts_ref[0]/sfreq)-0.1) if len(p_starts_ref)>0 else stim_start_t-0.1
            n_pulses_show = min(len(p_starts_ref),5) if len(p_starts_ref)>0 else 0
            vis_e_spline = min(times_v[-1] if len(times_v)>0 else (stim_end_t+0.1 if stim_end_t is not None else 2.0), 
                                (p_ends_ref[n_pulses_show-1]/sfreq)+0.1 if n_pulses_show>0 else (stim_end_t+0.1 if stim_end_t is not None else 2.0))
            
            plot_s_idx, plot_e_idx = max(0,int(vis_s_spline*sfreq)), min(len(original_sig_selected),int(vis_e_spline*sfreq))
            if plot_s_idx>=plot_e_idx and len(original_sig_selected)>0 : plot_s_idx=0; plot_e_idx=min(len(original_sig_selected),int(sfreq*2))

            plt.plot(times_v[plot_s_idx:plot_e_idx],original_sig_selected[plot_s_idx:plot_e_idx],label=f'Original Ch {best_channel_idx_for_final_sf}',color='dimgray',alpha=0.7,lw=1.0)
            plt.plot(times_v[plot_s_idx:plot_e_idx],corrected_sig_selected[plot_s_idx:plot_e_idx],label=f'Corrected Ch {best_channel_idx_for_final_sf}',color='dodgerblue',lw=1.2,alpha=0.9)
            first_lbl=True
            for i in range(len(p_starts_ref)):
                s,e=p_starts_ref[i],p_ends_ref[i]; seg_disp_s,seg_disp_e=max(plot_s_idx,s),min(plot_e_idx,e+1)
                if seg_disp_s<seg_disp_e:
                    ts,ds=times_v[seg_disp_s:seg_disp_e],corrected_sig_selected[seg_disp_s:seg_disp_e]
                    if len(ts)>0: plt.plot(ts,ds,color='red',lw=1.8,label='Interpolated' if first_lbl else "",zorder=5); first_lbl=False
            plt.title(f'Spline Interpolation Effect on Ch {best_channel_idx_for_final_sf}',fontsize=14)
            plt.xlabel('Time (s)');plt.ylabel('Amplitude')
            if plot_s_idx<plot_e_idx and len(times_v)>max(plot_s_idx,plot_e_idx-1 if plot_e_idx>0 else 0): plt.xlim(times_v[plot_s_idx],times_v[plot_e_idx-1 if plot_e_idx>0 else 0])
            plt.legend(loc='upper right');plt.grid(True,ls=':',alpha=0.6);plt.tight_layout();plt.show()
        else: print("No refined pulses to apply spline interpolation.")
        return results
    except Exception as e: print(f"Error in main: {str(e)}"); import traceback; traceback.print_exc(); return results

if __name__ == "__main__":
    analysis_results = main()
    if analysis_results:
        print("\n--- Analysis Summary ---")
        print(f"Analysis based on Channel: {analysis_results.get('analysis_channel_idx', 'N/A')} ({analysis_results.get('analysis_channel_name', 'N/A')})")
        print(f"Determined Stim Freq: {analysis_results.get('stim_freq', 'N/A'):.2f} Hz")
        if analysis_results.get('pulse_times_sec', np.array([])).size > 0:
            print(f"# Pulses refined: {len(analysis_results['pulse_starts_refined'])}")
            print(f"First 5 pulse starts (s): {np.round(analysis_results['pulse_times_sec'][:5], 4)}")
            print(f"First 5 pulse durations (s): {np.round(analysis_results['pulse_durations_sec'][:5], 4)}")
        else:
            print("# Pulses refined: 0")
        if 'corrected_data_all_channels' in analysis_results and analysis_results['corrected_data_all_channels'] is not None:
            print(f"Corrected data shape: {analysis_results['corrected_data_all_channels'].shape}")
        print("Pipeline completed." if analysis_results.get('stim_freq') is not None else "Pipeline completed with issues (e.g. stim_freq not determined).")
    else: print("Pipeline failed or was exited early.")