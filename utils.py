import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat 
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