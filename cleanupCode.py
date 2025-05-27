#
### Intro
# This is a translation of Kristin Sellers' 2018 artifact rejection protocol
#  [as used in Rao et. al:
#       "Direct Electrical Stimulation of Lateral Orbitofrontal Cortex Acutely
#        Improves Mood in Individuals with Symptoms of Depression.]

# One deviation: this only deals with the "clinical" datatype,
    # though it should not be hard to add functionality

# Another deviation: no manual channel removal here,
    #  that must be done beforehand

# Trying to do this analogous to the original

# ctrl + f "section #" to get to the section beginning

'''table of contents:
    Section 1 : calculating stim rate
    Section 2: Convolution + stim matching
    Section 3: spline removal
    
'''

# To do- analyze section two + three

import mne
import scipy

import os
import numpy as np

from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.signal import square

# make all the gui stuff functions at the end to make it tidy

# Prompt user to select a data file
import tkinter as tk
from tkinter import filedialog
import tkinter.simpledialog as simpledialog


import numpy as np
from scipy.signal import find_peaks

from mne.time_frequency import psd_array_multitaper



# --- Modular pipeline functions ---

def select_fif_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select EEG FIF file",
        filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
    )
    if not path:
        raise SystemExit("No file selected.")
    return path

def compute_mean_psd(data, sfreq, bandwidth=0.5):
    from mne.time_frequency import psd_array_multitaper
    import numpy as np
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
    from scipy.signal import find_peaks
    import numpy as np

    # Find peaks with absolute prominence threshold
    peaks, props = find_peaks(mean_psd, prominence=prominence)
    if len(peaks) == 0:
        raise ValueError(f"No peaks found with prominence ≥ {prominence}")

    # Absolute peak frequencies and prominences
    pfreqs = freqs[peaks]
    proms  = props['prominences']
    # Peak heights at those frequencies
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
        # fallback: use all peaks if none exceed threshold
        mask_rel = np.ones_like(rel_proms, dtype=bool)
    valid_freqs = pfreqs[mask_rel]
    stim_freq = np.min(valid_freqs)
    return stim_freq

def detect_artifacts(avg_signal, sfreq, stim_freq):
    import numpy as np
    from scipy.signal import find_peaks

    # 1) thresholded peak detection with refractory filtering
    thresh = np.mean(np.abs(avg_signal)) + 3 * np.std(np.abs(avg_signal))
    min_diff = int((1.0 / stim_freq) * sfreq / 2)
    peaks, props = find_peaks(np.abs(avg_signal), height=thresh, distance=min_diff)
    stim_inds = peaks.tolist()

    # 2) enforce minimum inter-pulse interval (half the period)
    min_diff = int((1.0 / stim_freq) * sfreq / 2)
    i = 0
    while i + 1 < len(stim_inds):
        if stim_inds[i+1] - stim_inds[i] < min_diff:
            del stim_inds[i+1]
        else:
            i += 1
    # Check last pair
    if len(stim_inds) >= 2 and stim_inds[-1] - stim_inds[-2] < min_diff:
        stim_inds.pop()

    return stim_inds

def template_match_starts(avg_signal, starts, sfreq, stim_freq):
    import numpy as np
    avg_signal = np.asarray(avg_signal, dtype=float)
    # Ensure starts indices are integers
    starts = [int(s) for s in starts]
    from scipy.signal import find_peaks
    win_samp = int(5 * sfreq / 1000)
    snippets = []
    for s in starts[:10]:
        seg = avg_signal[max(0, s - win_samp): s + win_samp]
        if len(seg) < 2 * win_samp:
            seg = np.pad(seg, (0, 2 * win_samp - len(seg)))
        snippets.append(seg)
    template = np.mean(snippets, axis=0)
    mf = np.convolve(avg_signal, template[::-1], mode='same')
    thr = np.percentile(mf, 80)
    dist = int(0.5 * (sfreq / stim_freq))
    peaks, _ = find_peaks(mf, height=thr, distance=dist)
    half = len(template) // 2
    ends = [min(len(avg_signal) - 1, p + half) for p in peaks]
    return peaks.tolist(), ends, template, mf

def spline_remove(data, starts, ends):
    from scipy.interpolate import CubicSpline
    import numpy as np
    clean = data.copy()
    for ch in range(clean.shape[0]):
        if np.isnan(clean[ch, 0]):
            continue
        for s, e in zip(starts, ends):
            x = [s - 1, e + 1]
            y = clean[ch, x]
            cs = CubicSpline(x, y)
            xs = np.arange(s, e + 1)
            clean[ch, xs] = cs(xs)
    return clean

def main():
    # i should look through detect_artifacts() and template_match()
    import mne
    import numpy as np

    # 1) Load data
    #path = select_fif_file()
    path = '/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw.fif'
    raw = mne.io.read_raw_fif(path, preload=True)
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    avg_sig = data.mean(axis=0)

    #below is commented out to
    '''
    #maybe we can use the best_ch to get the stim frequency, because it would be clearest
    # Restrict PSD computation to the clearest channel only
    # (best_ch/channel_data will be defined after artifact detection)


    #################

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

    #recalculaet on clearest stim arrticaft vchannel

    # Restrict PSD computation to the clearest channel only
    ch_idx = raw.ch_names.index(best_ch)
    channel_data = data[ch_idx:ch_idx+1, :]'''

    
    channel_data = data[8:9, :]# this is for the shortcut testing when you know best channel!!!!


    avg_sig = channel_data.mean(axis = 0) #recalculate avg_sig for clear chnanel?

    # 2) PSD & stim frequency
    clearest_psd, freqs = compute_mean_psd(channel_data, sfreq)
    stim_freq = find_stim_frequency(clearest_psd, freqs)
    print(f"Estimated stim frequency of clearest channel: {stim_freq:.2f} Hz") 
    #this is what we use from here on out


    '''# 3) Artifact detection
    initial_starts = detect_artifacts(avg_sig, sfreq, stim_freq)
    starts, ends, template, mf_out = template_match_starts(
        avg_sig, initial_starts, sfreq, stim_freq
    )
    print(f"Detected {len(starts)} artifact pulses in clearest channel")

    # --- Refine artifact boundaries by local baseline crossings ---
    signal = channel_data[0]
    baseline = np.median(signal)
    refined_starts = []
    refined_ends   = []
    half_win = int((1.0 / stim_freq) * sfreq / 2)  # half expected period
    for p in starts:
        # scan backward for where signal crosses baseline
        i = p
        while i > 0 and (signal[i] - baseline) * (signal[i-1] - baseline) > 0:
            i -= 1
        refined_starts.append(i)
        # scan forward for crossing
        j = p
        while j < len(signal)-1 and (signal[j] - baseline) * (signal[j+1] - baseline) > 0:
            j += 1
        refined_ends.append(j)
    starts, ends = refined_starts, refined_ends
    print("Artifact boundaries refined by baseline crossings.")

    # 4) Artifact removal on clearest channel!
    cleaned = spline_remove(channel_data, starts, ends)
    print("Spline artifact removal complete.")

    # --- Overlay synthetic square wave for stim-rate validation ---
    import numpy as np
    from scipy.signal import square

    # Align phase to first artifact
    t0 = starts[0] / sfreq
    # Time axis
    times = np.arange(len(avg_sig)) / sfreq
    # Synthetic square wave
    sq = square(2 * np.pi * stim_freq * (times - t0))
    sq *= np.max(np.abs(avg_sig))

    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(times, avg_sig, label='Average Signal', alpha=0.6)
    plt.plot(times, sq,      label=f'Synthetic {stim_freq:.2f} Hz Square', linestyle='--')
    plt.xlim(t0, t0 + 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Real vs. Synthetic Square Wave')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Mark and list artifact start/end times ---
    # Convert sample indices to seconds
    start_times = np.array(starts) / sfreq
    end_times   = np.array(ends)   / sfreq


    # Plot on the average signal
    plt.figure(figsize=(12, 4))
    plt.plot(times, avg_sig, label='Average Signal')
    # Plot artifact starts/ends at the signal amplitude
    y_start = avg_sig[starts]
    y_end   = avg_sig[ends]
    plt.scatter(start_times, y_start, color='green', marker='o', label='Starts')
    plt.scatter(end_times,   y_end,   color='red',   marker='x', label='Ends')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Artifact Start/End Markers')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Display and compare on the clearest stim channel ---
    import matplotlib.pyplot as plt
    import numpy as np

    # 1) Show detected stim frequency
    print(f"Detected stimulation frequency: {stim_freq:.2f} Hz")


    # Extract time series for that channel
    raw_ts = channel_data[0]
    clean_ts = cleaned[0]
    times = np.arange(raw_ts.size) / sfreq'''




if __name__ == "__main__":
    main()



raise SystemExit
###disregard everything below for now
