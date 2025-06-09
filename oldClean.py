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
    thresh = np.mean(np.abs(avg_signal)) + 3 * np.std(np.abs(avg_signal))
    min_dist = int(0.8 * (sfreq / stim_freq))
    peaks, _ = find_peaks(np.abs(avg_signal), height=thresh, distance=min_dist)
    return peaks.tolist()

def template_match_starts(avg_signal, starts, sfreq, stim_freq):
    import numpy as np
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
    import mne
    import numpy as np

    # 1) Load data
    path = select_fif_file()
    raw = mne.io.read_raw_fif(path, preload=True)
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    avg_sig = data.mean(axis=0)



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


    #recalculaet on clearest stim arrticaft vchannel

    # Restrict PSD computation to the clearest channel only
    ch_idx = raw.ch_names.index(best_ch)
    channel_data = data[ch_idx:ch_idx+1, :]

    

    avg_sig = channel_data.mean(axis = 0) #recalculate avg_sig for clear chnanel?

    # 2) PSD & stim frequency
    clearest_psd, freqs = compute_mean_psd(channel_data, sfreq)
    stim_freq = find_stim_frequency(clearest_psd, freqs)
    print(f"Estimated stim frequency of clearest channel: {stim_freq:.2f} Hz")


    # 3) Artifact detection
    initial_starts = detect_artifacts(avg_sig, sfreq, stim_freq)
    starts, ends, template, mf_out = template_match_starts(
        avg_sig, initial_starts, sfreq, stim_freq
    )
    print(f"Detected {len(starts)} artifact pulses in clearest channel")

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

    # --- Display and compare on the clearest stim channel ---
    import matplotlib.pyplot as plt
    import numpy as np

    # 1) Show detected stim frequency
    print(f"Detected stimulation frequency: {stim_freq:.2f} Hz")


    # Extract time series for that channel
    raw_ts = channel_data[0]
    clean_ts = cleaned[0]
    times = np.arange(raw_ts.size) / sfreq




if __name__ == "__main__":
    main()



raise SystemExit
###disregard everything below for now


#hidden root window for all dialogs
tk_root = tk.Tk()
tk_root.withdraw() 

def getFifFile():
    mne.set_log_level('WARNING')  # only show warnings/errors, not info lines

    file_path = filedialog.askopenfilename(
        title="Select EEG FIF file",
        filetypes=[("FIF files", "*.fif"), ("All files", "*.*")]
    )
    if not file_path:
        raise SystemExit("No file selected, exiting.")

    print(f"Selected file: {file_path}")
    print(f"Exists: {os.path.exists(file_path)}")

    #the above lets the user choose a file

    file = mne.io.read_raw_fif(file_path, preload=True)

    return file


### BELOW IS THE GUI FOR SELECTING A .FIF FILE, UNCOMMENT WHEN PUBLISSHING
#### raw = getFifFile()


raw = mne.io.read_raw_fif( '/Users/aashray/Documents/ChangLab/RCS04_tr_103_eeg_raw.fif' , preload = True)


print("File uploaded, fetching stim frequency") #Section 1 starts here




### Parameters first
#  (gonna straight copy from matlab and then move from there)
#   this would be easier to use if you made a GUI for changing these

stimFreq = 100 # in Hz
removeOrReplace = 2 # 1 = replace bad data with zeros; 2 = remove bad data
currentStim = 'OFC_6mA'
# dataType = 1 # 1 = clinical, 2 = TDT, 3 = NeuroOmega . [[THIS PART IS NOT RELEVANT, CLINICAL ONLY]]
rejectStimChans = 1 # 0 = do not reject stim channels; 1 = reject stim channels



sampleRate = raw.info['sfreq']

data = raw.get_data()
n_chan, _ = data.shape # shape (n_chan, n_samples)




#'bandwidth' is W, T is automatically calculated, L is based off T and W
def multitaperMeanPSD(bandwidth):
    # 2) Compute PSD on every channel via multi‐taper
    # --- Compute PSD per channel with progress updates ---
    psd_list = []
    total_ch = len(raw.ch_names)
    for idx, ch in enumerate(raw.ch_names):
        # single-line progress update
        print(f"\rProcessing using bandwidth {bandwidth:.2f} Hz — channel {idx+1}/{total_ch} ({ch})", end='', flush=True)
        psd_ch, freqencyList = psd_array_multitaper(
            data[idx:idx+1],
            sfreq=sampleRate,
            fmin=0,
            fmax=sampleRate / 2,
            bandwidth=bandwidth,
            adaptive=False,
            low_bias=True,
            normalization='full',
            verbose=False
        )
        psd_list.append(psd_ch[0])
    # finalize progress line
    print()
    # Stack them into an array for averaging
    psds = np.vstack(psd_list)
    mean = psds.mean(axis=0)
    return mean, freqencyList

'''
frequencyRes = 0.25 #the divisor defines the
# Increase FFT length for finer frequency resolution (0.01 Hz)
n_fft = int(sampleRate / frequencyRes)

# 2) Compute PSD on every channel via Welch
psd = raw.compute_psd(
    method='welch',
    n_fft=n_fft,
    n_overlap=n_fft // 2,
    n_per_seg=n_fft,
    fmin= frequencyRes,
    fmax= sampleRate / 2 #### lowest 
)
# psd is an instance of PSDArray; extract arrays:
freqs = psd.freqs            # shape (n_freqs,)
psds = psd.get_data()        # shape (n_channels, n_freqs)
'''


# 3) Average across channels
#mean_psd, freqs = multitaperMeanPSD(1)





###There was a graph here of the averaged psd, its at the bottom and commented out now
def findPeak(): #with diagram
    #### Find PEAK
    # 1) find peaks and prominences of spectra
    peaks, props = find_peaks(mean_psd, prominence=10)# check this, play artound with the prominence
    pfreqs = freqs[peaks]
    proms  = props['prominences']


    # Print out each peak frequency and its prominence
    print("Detected PSD peaks and their prominences:")
    for freq_val, prom_val in zip(pfreqs, proms):
        print(f"  {freq_val:.2f} Hz → prominence {prom_val:.4f}")


    # Only use the top 20 most prominent peaks
    N_display = min(20, len(proms))
    top_idx = np.argsort(proms)[::-1][:N_display]
    top_freqs_plot = pfreqs[top_idx]
    top_proms_plot = proms[top_idx]
    top_peaks_plot = peaks[top_idx]

    # After you compute the PSD and get freqs:
    print("First few frequency bins:", freqs[:5])
    res = freqs[1] - freqs[0]
    print(f"Frequency resolution = {res:.4f} Hz (constant spacing)")
    # Smaller marker size
    sizes = top_proms_plot * 100

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, mean_psd, label='Mean PSD')
    plt.scatter(top_freqs_plot, mean_psd[top_peaks_plot], s=sizes, c='red', alpha=0.7)
    for f, p in zip(top_freqs_plot, top_proms_plot):
        psd_val = mean_psd[freqs == f][0]
        plt.text(f, psd_val, f"{p:.2f}", ha='center', va='bottom', fontsize=8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Mean PSD with Peak Prominences')
    # Set axis limits to focus on top 10 peaks with margins
    x_min = 0
    x_max = top_freqs_plot[:10].max() + 20
    y_min = 0
    y_max = mean_psd[top_peaks_plot[:10]].max() * 1.2
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    from matplotlib.lines import Line2D
    # Highlight the stim frequency (lowest freq with prominence > 10)
    stim_mask = top_proms_plot > 10
    if np.any(stim_mask):
        stim_idx = np.where(stim_mask)[0][np.argmin(top_freqs_plot[stim_mask])]
        stim_freq = top_freqs_plot[stim_idx]
        stim_power = mean_psd[freqs == stim_freq][0]
        plt.scatter([stim_freq], [stim_power], s=top_proms_plot[stim_idx] * 100, c='lime', alpha=0.7)
        stim_legend = Line2D([0], [0], marker='o', color='w', label=f'Stim Frequency ({stim_freq:.1f} Hz)',
                            markerfacecolor='lime', markersize=6, alpha=0.7)
        text_legend = Line2D([], [], color='none', marker='', label='Black numbers = prominence of peak')
        # Update legend handles
        plt.legend(handles=[
            Line2D([], [], color='C0', label='Mean PSD'),
            Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=6, alpha=0.7, label='Peaks'),
            stim_legend,
            text_legend
        ])
    else:
        text_legend = Line2D([], [], color='none', marker='', label='Black numbers = prominence of peak')
        plt.legend(handles=[
            Line2D([], [], color='C0', label='Mean PSD'),
            Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=6, alpha=0.7, label='Peaks'),
            text_legend
        ])

    plt.tight_layout()
    plt.show()

    # --- Find peak power near the calculated stim frequency ---
    # Define search window ±0.5 Hz around stim_freq
    # 1) Build a mask for the ±0.5 Hz window
    lower, upper = stim_freq - 0.5, stim_freq + 0.5
    mask = (freqs >= lower) & (freqs <= upper)

    # 2) Slice out the PSD values in that range
    local_psd = mean_psd[mask]
    local_freqs = freqs[mask]

    # 3) Find the index of the maximum power in that slice
    idx = np.argmax(local_psd)

    # 4) Extract the corresponding frequency and power
    best_freq  = local_freqs[idx]
    best_power = local_psd[idx]

    print(f"Highest power within ±0.5 Hz of {stim_freq:.2f} Hz:")
    print(f"  {best_freq:.2f} Hz with power {best_power:.4e}")
    
    return best_freq #this is supposed to eventually just be the stim





#
# --- Compare PSDs for different bandwidths ß
'''bandwidths = [0.012, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_bw = len(bandwidths)
fig, axes = plt.subplots(n_bw, 1, figsize=(8, 3 * n_bw), sharex=True)
for ax, bw in zip(axes, bandwidths):
    print(f"Processing bandwidth = {bw} Hz")
    mean_psd, freqs = multitaperMeanPSD(bw)
    peaks, props = find_peaks(mean_psd, prominence=10)
    pfreqs = freqs[peaks]
    proms  = props['prominences']
    # Plot PSD
    ax.plot(freqs, mean_psd, label=f'BW={bw} Hz')
    # Mark peaks
    sizes = proms * 100
    ax.scatter(pfreqs, mean_psd[peaks], s=sizes, c='red', alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylabel('PSD')
    ax.legend(loc='upper right')
axes[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()'''


mean_psd, freqs = multitaperMeanPSD(0.5)

calcstim = findPeak()


#now calcStim is the final stim rate, treat it as such 

print("Time to start convolution!") #Section 2 starts here!


# Compute average signal across channels (ensure it's defined)
avg_signal = data.mean(axis=0)

# Initial rough detection of artifact starts via thresholding
# Define threshold as mean + 3*std of absolute signal
thresh_init = np.mean(np.abs(avg_signal)) + 3 * np.std(np.abs(avg_signal))
# Minimum distance between pulses in samples (~0.8 × period)
min_dist = int(0.8 * (sampleRate / calcStim))
# Find peaks in absolute average signal
peaks_init, _ = find_peaks(np.abs(avg_signal), height=thresh_init, distance=min_dist)
artifactStarts = peaks_init.tolist()

# --- Template Matching Section ---

# avg_signal: your mean signal across channels (1D array)
# artifactStarts: list/array of initial pulse sample indices
# sampleRate: sampling rate in Hz
# calcStim: your estimated stim frequency

# 1) Build the template by averaging snippets around detected pulses
window_ms = 5  # window half-width in milliseconds
window_samples = int(window_ms * sampleRate / 1000)
snippets = []

for idx in artifactStarts[:10]:  # use first 10 pulses
    start = max(0, idx - window_samples)
    end = min(len(avg_signal), idx + window_samples)
    snippet = avg_signal[start:end]
    # pad/truncate to uniform length
    if len(snippet) < 2*window_samples:
        snippet = np.pad(snippet, (0, 2*window_samples - len(snippet)), mode='constant')
    snippets.append(snippet)

template = np.mean(snippets, axis=0)

# 2) Matched filter: convolve with time-reversed template
mf_kernel = template[::-1]
mf_output = np.convolve(avg_signal, mf_kernel, mode='same')

# 3) Detect peaks in matched-filter output (80th percentile threshold, 0.5× period spacing)
period_samples = sampleRate / calcStim
peak_threshold = np.percentile(mf_output, 80)  # top 20% as threshold
distance = int(0.5 * period_samples)           # allow as close as half a period
peaks_mf, props_mf = find_peaks(
    mf_output,
    height=peak_threshold,
    distance=distance
)

# Use matched-filter peaks as artifact starts
artifactStarts = peaks_mf.tolist()
half_len = len(template) // 2
artifactEnds   = [min(len(avg_signal)-1, start + half_len) for start in artifactStarts]
print(f"Now detected {len(artifactStarts)} artifact pulses (template matching)")

# 4) Plot template and matched-filter response
#    Zoom into one pulse window around the first detected peak
fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)

# Template plot
t_template = (np.arange(len(template)) - len(template)//2) / sampleRate
ax_t.plot(t_template, template, color='C1')
ax_t.set_title("Derived Artifact Template")
ax_t.set_xlabel("Time (s)")
ax_t.set_ylabel("Amplitude")

# Matched-filter output
t_sig = np.arange(len(avg_signal)) / sampleRate
ax_m.plot(t_sig, mf_output, color='C2', label='Matched Filter Output')
ax_m.scatter(peaks_mf / sampleRate, mf_output[peaks_mf], color='red', marker='x', label='Detected Pulses')
# zoom around first pulse
t0 = artifactStarts[0] / sampleRate
ax_m.set_xlim(t0 - 0.05, t0 + 0.05)
ax_m.set_title("Matched-Filter Peaks (Zoomed)")
ax_m.set_xlabel("Time (s)")
ax_m.set_ylabel("Response")
ax_m.legend()

plt.show()


####### Overlay synthetic sine wave for stim-rate validation
# --- Overlay synthetic sine wave for stim-rate validation ---
# Align a sine wave to the first detected stimulation
t0 = artifactStarts[0] / sampleRate
# Time axis in seconds
times = np.arange(len(avg_signal)) / sampleRate
# Shift so the sine peaks align with the first artifact
sine = np.sin(2 * np.pi * calcStim * (times - t0))
# Normalize to the signal amplitude range (± max of avg_signal)
sine = sine * np.max(np.abs(avg_signal))
# Plot the sine wave on the matched-filter output plot for debugging
plt.figure(figsize=(12, 4))
plt.plot(times, avg_signal, label='Average Signal', alpha=0.6)
plt.plot(times, sine,     label=f'Synthetic {calcStim:.2f} Hz Sine', linestyle='--')
plt.xlim(t0, t0 + 1)  # zoom 1 second around first pulse
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Real vs. Synthetic Stim Pulses')
plt.legend()
plt.tight_layout()
plt.show()


# --- Overlay synthetic square wave for stim-rate validation ---
# Generate time axis
# (times already defined above)
# Create a square wave at the calculated stim frequency
sq = square(2 * np.pi * calcStim * (times - t0))
# Scale to same amplitude range as avg_signal
sq = sq * np.max(np.abs(avg_signal))

plt.figure(figsize=(12, 4))
plt.plot(times, avg_signal, label='Average Signal', alpha=0.6)
plt.plot(times, sq,      label=f'Synthetic {calcStim:.2f} Hz Square', linestyle='--')
plt.xlim(t0, t0 + 1)  # zoom around first pulse
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Real vs. Synthetic Square Wave')
plt.legend()
plt.tight_layout()
plt.show()


####### GPT VISSUALIZATION
# --- Visualization of template-matched artifact windows ---

# Automatically select the first channel if the hardcoded one is missing
if 'POL R CMa1-Ref' not in raw.ch_names:
    ch_name = raw.ch_names[0]
else:
    ch_name = 'POL R CMa1-Ref'
print(f"Plotting channel {ch_name} with artifact boundaries")
data_ch, times = raw.copy().pick(ch_name).get_data(return_times=True)
signal = data_ch[0]  # flatten to 1D

 # Define artifactEnds corresponding to each start using half-template length
half_len = len(template) // 2
artifactEnds = [min(len(avg_signal)-1, start + half_len) for start in artifactStarts]

# Convert artifact sample indices to time in seconds
t_starts = np.array(artifactStarts) / sampleRate
t_ends   = np.array(artifactEnds)   / sampleRate

plt.figure(figsize=(12, 4))
plt.plot(times, signal, label=ch_name, color='C0')

# Plot starts
plt.scatter(t_starts,
            signal[artifactStarts],
            c='green', marker='o', s=50,
            label='Artifact Starts')

# Plot ends
plt.scatter(t_ends,
            signal[artifactEnds],
            c='red', marker='x', s=50,
            label='Artifact Ends')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f"Channel {ch_name} with Template-Matched Artifact Windows")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


from scipy.interpolate import CubicSpline

def spline_stim_artifact(data, artifact_starts, artifact_ends):
    """
    Replace stimulation artifact segments in data via cubic spline interpolation.
    data: ndarray, shape (n_channels, n_samples)
    artifact_starts, artifact_ends: lists of sample indices
    """
    clean = data.copy()
    n_chan, _ = clean.shape
    for ch in range(n_chan):
        # skip channel if starts with NaN
        if np.isnan(clean[ch, 0]):
            continue
        for start, end in zip(artifact_starts, artifact_ends):
            # define points before and after artifact
            # ensure indices valid
            x = [start - 1, end + 1]
            y = clean[ch, x]
            # spline across artifact window
            cs = CubicSpline(x, y)
            xs = np.arange(start, end+1)
            clean[ch, xs] = cs(xs)
    return clean



# --- After plotting artifact boundaries, before spline interpolation ---

# 1) Debug print
print("First 10 artifact windows (samples):")
for s, e in zip(artifactStarts[:10], artifactEnds[:10]):
    print(f"  {s} → {e}")

# 2) Shade windows on raw signal
plt.figure(figsize=(12, 4))
plt.plot(times, signal, label='Raw Signal')
for s, e in zip(artifactStarts, artifactEnds):
    plt.axvspan(s / sampleRate, e / sampleRate, color='orange', alpha=0.3)
plt.title("Raw Signal with Artifact Windows Shaded")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# --- Now apply spline interpolation ---
print("Applying spline interpolation to remove artifacts...")
splineData = spline_stim_artifact(data, artifactStarts, artifactEnds)

# 3) Compare raw vs cleaned
cleaned = splineData[raw.ch_names.index(ch_name), :]
plt.figure(figsize=(12, 4))
plt.plot(times, signal,  label='Raw',   alpha=0.5)
plt.plot(times, cleaned, label='Cleaned', linewidth=1)
plt.title(f"Raw vs Cleaned Signal ({ch_name})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

raise SystemExit






if __name__ == "__main__":
    main()



raise SystemExit