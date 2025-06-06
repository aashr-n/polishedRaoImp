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


def main():
    # i should look through detect_artifacts() and template_match()
    import mne
    import numpy as np

    # 1) Load data
    #path = select_fif_file()
    path = '/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw.fif'
    raw = mne.io.read_raw_fif(path, preload=True)
    sfreq = raw.info['sfreq']
    print(f"Sample frequency : {sfreq}")

    data = raw.get_data()
    avg_sig = data.mean(axis=0)

    
    channel_data = data[8:9, :]# this is for the shortcut testing when you know best channel!!!!


    avg_sig = channel_data.mean(axis = 0) #recalculate avg_sig for clear chnanel?

    # 2) PSD & stim frequency
    clearest_psd, freqs = compute_mean_psd(channel_data, sfreq)
    stim_freq = find_stim_frequency(clearest_psd, freqs)
    print(f"Estimated stim frequency of clearest channel: {stim_freq:.2f} Hz") 
    #this is what we use from here on out



    ###########################CONVOLUTION######################################

    # --- Step 3: Periodic Prominence-Based Peak Detection ---

    # 3.1) Prepare signal and parameters
    signal = channel_data[0]
    # Convert stim frequency to expected sample period
    period_samples = int(sfreq / stim_freq)
    half_period = period_samples // 2
    # Define tolerance around period (±10%)
    tol = int(0.1 * period_samples)

    # 3.5) Identify time segments where stim frequency power is highest using multitaper
    import numpy as np
    from mne.time_frequency import psd_array_multitaper

    signal = channel_data[0]
    times = np.arange(signal.size)/sfreq
    # trying to dynamically create windows and steps
    win_sec  = round(sfreq/5000, 1) #0.2
    step_sec = win_sec/2 #0.1
    nperseg  = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)
    segment_centers = []
    segment_power   = []

    # Slide window through the signal
    for start in range(0, signal.size - nperseg + 1, step_samps):
        stop = start + nperseg
        # multitaper PSD on this window
        psd_w, freqs_w = psd_array_multitaper(
            signal[start:stop][None, :], sfreq=sfreq,
            fmin=0, fmax=sfreq/2, bandwidth=stim_freq/5,
            adaptive=True, low_bias=True, normalization='full', verbose=False
        )
        psd_w = psd_w[0]  # extract 1D array
        # Find power at the nearest frequency bin to stim_freq
        idx = np.argmin(np.abs(freqs_w - stim_freq))
        segment_centers.append((start + stop) / 2 / sfreq)
        # Normalize stim-frequency power by total power in this window
        rel_power = (psd_w[idx] / np.sum(psd_w)) * 100
        segment_power.append(rel_power)

    segment_power    = np.array(segment_power)
    segment_centers  = np.array(segment_centers)

    # Determine threshold for “high” stim power (e.g., for top 25% of windows set 75)
    thresh_power = np.percentile(segment_power, 0)
    high_idx = segment_power >= thresh_power
    high_times = segment_centers[high_idx]

    print(f"Highlighting {high_idx.sum()} segments with high stim-frequency power")

    # Normalize displayed powers so the highest is 100
    max_high_power = segment_power[high_idx].max() if np.any(high_idx) else 1.0

    # 3.6) Plot the full signal and mark high-power windows
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(times, signal, label='Signal')
    # Plot windows colored by relative power
    import matplotlib.cm as cm

    cmap = cm.get_cmap('Reds')  # 'Reds' and 'plasma' are pretty good
    #for diff colors reference: https://www.analyticsvidhya.com/blog/2020/09/colormaps-matplotlib/
    
    for idx, t0 in enumerate(high_times):
        # normalized power between 0 and 1
        power = segment_power[high_idx][idx]
        norm_power = power / max_high_power
        # choose color from colormap
        color = cmap(norm_power)
        plt.axvspan(t0 - win_sec/2, t0 + win_sec/2, color=color, alpha=0.5)

    # Add a colorbar indicating relative PSD power (0–100%)
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Create a ScalarMappable for the colormap
    norm = Normalize(vmin=0, vmax=100)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Relative Stim-Freq Power (%)')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude / Power')
    plt.title('Signal with High Stim-Frequency Power Segments Highlighted')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Determine true stimulation period by expanding from 100% windows ---
    import numpy as np

    # Find maximum prominence value (should be 100 after normalization)
    segment_prom_rel = segment_power  # Here, segment_power is already relative power (%)
    max_prom_val = segment_prom_rel.max()
    # Locate indices of windows at max prominence
    max_idxs = np.where(segment_prom_rel == max_prom_val)[0]
    first_max, last_max = max_idxs.min(), max_idxs.max()

    # Define a drop fraction (e.g., 80% of max) to detect sharp drop-off
    drop_frac = 0.1 # this seems to work?????
    drop_thresh = drop_frac * max_prom_val

    # Expand left from first_max until relative prominence falls below drop_thresh
    start_idx = first_max
    while start_idx > 0 and segment_prom_rel[start_idx] >= drop_thresh:
        start_idx -= 1
    stim_start_time = segment_centers[start_idx]

    # Expand right from last_max until relative prominence falls below drop_thresh
    end_idx = last_max
    while end_idx < len(segment_prom_rel) - 1 and segment_prom_rel[end_idx] >= drop_thresh:
        end_idx += 1
    stim_end_time = segment_centers[end_idx]

    print(f"Estimated stimulation active from {stim_start_time:.2f}s to {stim_end_time:.2f}s")

    # --- Plot the full trace with the detected stim period highlighted ---
    plt.figure(figsize=(12, 4))
    plt.plot(times, signal, label='Signal')
    plt.axvspan(stim_start_time, stim_end_time, color='green', alpha=0.3, label='Stim Period')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Detected Continuous Stimulation Period')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()



raise SystemExit
###disregard everything below for now
