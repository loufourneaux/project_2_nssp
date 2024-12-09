import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat 
from scipy.signal import butter
from scipy.signal import filtfilt
import pandas as pd

# Solution: load a list of dicts
def load_subject(subject=2,path='data'):
    file_name = f'{path}/S{subject}_A1_E1.mat'
    emg = loadmat(file_name)
    subject_dict = {}
    subject_dict['number'] = subject
    # The actual interesting data lives in the columns ['emg', 'restimulus', 'rerepetition']
    subject_dict['emg_rectified'] = emg["emg"].copy()
    # Restimulus and rerepetition are the stimulus and repetition that have been processed for better correspondance to real movement
    # Stimulus is the type of movement that is being performed
    subject_dict['stimulus'] = emg["restimulus"] 
    # Repetition is the number of times the movement has been repeated
    subject_dict['repetition'] = emg["rerepetition"] 

    return subject_dict

def get_attributes(subject):
    return subject['emg_rectified'], subject['stimulus'], subject['repetition']

def explore_data(subject):
    emg_rectified, stimulus, repetition = get_attributes(subject)
    # -1 because 0 is the resting condition
    n_stimuli = len(np.unique(stimulus)) - 1 
    # -1 because 0 is not a repetition
    n_repetitions = len(np.unique(repetition)) - 1 
    n_channels = emg_rectified.shape[1]
    
    channels = np.arange(1, n_channels+1)
    stimuli = np.arange(1, n_stimuli+1)

    print(f'How many types of movement are there? {n_stimuli}') 
    print(f'How many repetitions are there? {n_repetitions}')
    print(f'There are {n_channels} channels.')

    number_of_samples_per_trial = np.zeros((n_stimuli, n_repetitions))

    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            
            idx = np.logical_and(stimulus == stimuli_idx+1, repetition == repetition_idx+1)
            number_of_samples_per_trial[stimuli_idx, repetition_idx] = np.sum(idx.astype(int))

    print('Length of trials: ')
    df = pd.DataFrame(number_of_samples_per_trial, index=stimuli, columns=channels)
    df.columns = pd.MultiIndex.from_product([['Channel'], df.columns]) 
    df.rename_axis(index='Stimuli', inplace=True)
    display(df)

    return n_stimuli,n_channels, n_repetitions

def smooth_data(subject, n_stimuli, n_repetitions):
    emg_rectified, stimulus, repetition = get_attributes(subject)
    # Defining the length of the moving average window
    mov_mean_length = 25 
    mov_mean_weights = np.ones(mov_mean_length) / mov_mean_length

    # Initializing the data structure
    # EMG data for specific stimuli and repetition indices.
    emg_windows = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)]
    # moving average or smoothed signal (envelope) for each window of data.
    emg_envelopes = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)]

    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            idx = np.logical_and(stimulus == stimuli_idx + 1, repetition == repetition_idx + 1).flatten()
            emg_windows[stimuli_idx][repetition_idx] = emg_rectified[idx, :]
            emg_envelopes[stimuli_idx][repetition_idx]  = np.apply_along_axis(
                lambda x: np.convolve(x, mov_mean_weights, mode='same'),
                axis=0,
                arr=emg_windows[stimuli_idx][repetition_idx]
            )

    subject['emg_windows'] = emg_windows
    subject['emg_envelopes'] = emg_envelopes

    return subject


def plot_channel(emg_rectified, channel=9):
    # Visualize the data from one channel
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(emg_rectified[:, channel])   
    ax.set_title(f"EMG signal channel {channel}")
    ax.set_xlabel("Data points")
    ax.set_ylabel("Amplitude")
    plt.plot()


def build_dataset_from_ninapro(emg, stimulus, repetition, features=None):
    # Calculate the number of unique stimuli and repetitions, subtracting 1 to exclude the resting condition
    n_stimuli = np.unique(stimulus).size - 1
    n_repetitions = np.unique(repetition).size - 1
    # Total number of samples is the product of stimuli and repetitions
    n_samples = n_stimuli * n_repetitions
    
    # Number of channels in the EMG data
    n_channels = emg.shape[1]
    # Calculate the total number of features by summing the number of channels for each feature
    n_features = sum(n_channels for feature in features)
    
    # Initialize the dataset and labels arrays with zeros
    dataset = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)
    current_sample_index = 0
    
    # Loop over each stimulus and repetition to extract features
    for i in range(n_stimuli):
        for j in range(n_repetitions):
            # Assign the label for the current sample
            labels[current_sample_index] = i + 1
            # Calculate the current sample index based on stimulus and repetition
            current_sample_index = i * n_repetitions + j
            current_feature_index = 0
            # Select the time steps corresponding to the current stimulus and repetition
            selected_tsteps = np.logical_and(stimulus == i + 1, repetition == j + 1).squeeze()
            
            # Loop over each feature function provided
            for feature in features:
                # Determine the indices in the dataset where the current feature will be stored
                selected_features = np.arange(current_feature_index, current_feature_index + n_channels)
                # Apply the feature function to the selected EMG data and store the result
                dataset[current_sample_index, selected_features] = feature(emg[selected_tsteps, :])
                # Update the feature index for the next feature
                current_feature_index += n_channels

            # Move to the next sample
            current_sample_index += 1
            
    # Return the constructed dataset and corresponding labels
    return dataset, labels

def high_pass_filter(data, cutoff=20, fs=1000, order=4):
    """
    Applies a high-pass filter to the input data.
    
    Parameters:
        data (ndarray): The EMG data (samples x channels).
        cutoff (float): Cutoff frequency of the high-pass filter in Hz.
        fs (int): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        filtered_data (ndarray): High-pass filtered data.
    """
    # Design the Butterworth high-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # Apply the filter along the first axis (time)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


    
def process_and_plot_emg(emg_data, stimulus, repetition, n_stimuli, n_repetitions, mov_mean_length=25):
    """
    Process and plot EMG data for rectified signals and their envelopes.

    Parameters:
        emg_data (numpy array): The EMG data (time points x channels).
        stimulus (numpy array): Array indicating the stimulus indices (1-based).
        repetition (numpy array): Array indicating the repetition indices (1-based).
        n_stimuli (int): Number of stimuli.
        n_repetitions (int): Number of repetitions per stimulus.
        mov_mean_length (int): Length of the moving average window.
    """
    # Define moving average weights
    mov_mean_weights = np.ones(mov_mean_length) / mov_mean_length

    # Initialize data structures
    emg_windows = [[None for _ in range(n_repetitions)] for _ in range(n_stimuli)]
    emg_envelopes = [[None for _ in range(n_repetitions)] for _ in range(n_stimuli)]

    # Process each stimulus and repetition
    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            idx = np.logical_and(stimulus == stimuli_idx + 1, repetition == repetition_idx + 1).flatten()
            emg_windows[stimuli_idx][repetition_idx] = emg_data[idx, :]
            emg_envelopes[stimuli_idx][repetition_idx] = np.apply_along_axis(
                lambda x: np.convolve(x, mov_mean_weights, mode='same'),
                axis=0,
                arr=emg_windows[stimuli_idx][repetition_idx]
            )

    # Plot settings
    num_channels = emg_data.shape[-1]
    rows, cols = 2, (num_channels + 1) // 2  # Adjust rows and columns for subplot arrangement

    # Plot Rectified EMG signals
    fig, ax = plt.subplots(rows, cols, figsize=(12, 6), constrained_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for channel_idx in range(num_channels):
        ax[channel_idx].plot(emg_windows[0][0][:, channel_idx])
        ax[channel_idx].set_title(f"Channel {channel_idx + 1}")
    plt.suptitle("Rectified EMG Signal")

    # Plot EMG Envelopes
    fig, ax = plt.subplots(rows, cols, figsize=(12, 6), constrained_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for channel_idx in range(num_channels):
        ax[channel_idx].plot(emg_envelopes[0][0][:, channel_idx])
        ax[channel_idx].set_title(f"Channel {channel_idx + 1}")
    plt.suptitle("Envelopes of the EMG Signal")

    plt.show()
    return emg_envelopes

def filter_low_mav_channels_multifeature(dataset, labels, n_features_per_channel):
    """
    Filters out channels with low MAV values (below the first quartile) and removes
    corresponding feature values for those channels.
    
    Args:
        dataset (np.ndarray): Dataset of shape (n_samples, n_channels * n_features_per_channel).
        labels (np.ndarray): The corresponding labels of shape (n_samples,).
        n_channels (int): The number of EMG channels.
        n_features_per_channel (int): The number of features per channel.
    
    Returns:
        filtered_dataset (np.ndarray): Dataset with features of low MAV channels removed.
        labels (np.ndarray): The unchanged labels.
        retained_channels (list): Indices of retained channels.
    """
    # Step 1: Extract MAV columns (assumes MAV is the first feature set)
    mav_columns = dataset[:, :10]  # First `n_channels` columns are MAV
    
    # Step 2: Compute the average MAV for each channel
    mav_values = np.mean(mav_columns, axis=0)
    
    # Step 3: Calculate the first quartile (Q1) threshold
    q1 = np.percentile(mav_values, 25)
    
    # Step 4: Retain only channels with MAV >= Q1
    retained_channels = [i for i, mav in enumerate(mav_values) if mav >= q1]
    
    # Step 5: Identify columns to retain in the dataset
    retained_columns = []
    for channel in retained_channels:
        start = channel * n_features_per_channel
        end = start + n_features_per_channel
        retained_columns.extend(range(start, end))
    
    # Step 6: Filter the dataset to retain only the selected columns
    filtered_dataset = dataset[:, retained_columns]
    
    return filtered_dataset, labels, retained_channels

def analyze_feature(dataset, labels, n_stimuli, n_repetitions, feature):
    if feature == 'mav':
        dataset=dataset[:, :10]
    elif feature == 'std':
        dataset=dataset[:, 10:20]
    elif feature == 'maxav':
        dataset=dataset[:, 20:30]
    elif feature == 'rms':
        dataset=dataset[:, 30:40]
    elif feature == 'wl':
        dataset=dataset[:, 40:50]
    elif feature == 'ssc':
        dataset=dataset[:, 50:]
    else:
        raise ValueError("Unsupported feature")

    features_set =dataset.reshape(n_stimuli, n_repetitions, 10)
    fig, ax = plt.subplots(4, 3, figsize=(10, 6), constrained_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()

    for stimuli_idx in range(n_stimuli):
        sns.heatmap(features_set[stimuli_idx, :, :].T, ax=ax[stimuli_idx] ,xticklabels=False, yticklabels=False, cbar = True)
        ax[stimuli_idx].title.set_text("Stimulus " + str(stimuli_idx + 1))
        ax[stimuli_idx].set_xlabel("Repetition")
        ax[stimuli_idx].set_ylabel("EMG channel")

def analyze_psd(emg, fs=1000, cutoff=20):
    """
    Analyze the power spectral density (PSD) of the EMG signal to detect artifacts.

    Parameters:
        emg (ndarray): EMG data (samples x channels).
        fs (int): Sampling frequency in Hz.
        cutoff (float): Low-frequency cutoff for artifact detection.

    Returns:
        lfpr (ndarray): Low-frequency power ratio for each channel.
    """
    n_channels = emg.shape[1]
    lfpr = np.zeros(n_channels)
    
    # Loop through channels
    for ch in range(n_channels):
        freqs, psd = welch(emg[:, ch], fs=fs)
        total_power = np.trapz(psd, freqs)
        low_freq_power = np.trapz(psd[freqs <= cutoff], freqs[freqs <= cutoff])
        lfpr[ch] = low_freq_power / total_power
        
        # Plot PSD for visual inspection
        plt.figure()
        plt.semilogy(freqs, psd)
        plt.axvline(cutoff, color='r', linestyle='--', label=f"Cutoff: {cutoff} Hz")
        plt.title(f"Channel {ch + 1}: PSD Analysis")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.legend()
        plt.grid()
        plt.show()
    
    return lfpr