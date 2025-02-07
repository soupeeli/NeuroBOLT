import mne
import numpy as np
import pandas as pd

def normalize_data(data, means_stds=None, mode='absmax'):
    """
    data - normalize by second.
    means_std - (means, stds )

    """
    # if mode == 'absmax':
    #     transform_data = data/(
    #         np.quantile(np.abs(data), q=0.95, method="linear", axis=-1, keepdims=True)
    #         + 1e-8)
    #     return transform_data, means_stds

    if mode == 'absmax':
        means = np.mean(data, axis=-1, keepdims=True)
        data = data - means
        transform_data = data / (
                np.quantile(np.abs(data), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8)
        return transform_data, means_stds

    if means_stds is None:
        means = np.mean(data, axis=-1, keepdims=True)
        stds = np.std(data, axis=-1, keepdims=True)
        means_stds = (means, stds)

    transform_data = (data - means_stds[0]) / means_stds[1]

    return transform_data, means_stds


def epoching_seq2one(eegraw, fmridataset,
                     tmin, tmax, event_name, ifnorm=0, crop=0,
                     dropfmriframes=0):
    """
    Function for generating paired data of EEG sequences and corresponding single time points of fMRI.

    Inputs:
       - `eegraw`:
         Raw MNE data structure containing EEG recordings.
       - `fmridataset`:
         A numpy array of shape (#roi x #time), representing the fMRI data (regions of interest over time).
       - `tmin, tmax`:
         Start and end times of the epochs in seconds, relative to the time-locked event.
         (e.g., `tmin=-5` and `tmax=2.1` mean epochs from 5 seconds before to 2.1 seconds after the event.)
       - `event_name`:
         The name of the event in the EEG annotations used for time-locking.
       - `ifnorm`:
         Whether to normalize the EEG data within each epoch window:
         - `0` (default): No normalization.
         - `1`: Normalize the data.
       - `crop`:
         Optionally crop the EEG sequence by trimming it to a fixed length (default: no cropping).
       - `dropfmriframes`:
         Number of initial fMRI frames to drop (default: 0).
         This is dataset-specific and can be used to account for magnetic stabilization during acquisition.

    Outputs:
       - `data`: A dictionary containing:
         - `"eeg"`: List of numpy arrays, where each array represents an EEG epoch of shape (#channels x #time).
         - `"fmri"`: List of numpy arrays, where each array represents a single time point of fMRI data with shape (#roi).
       - `eeg_epoch`: MNE `Epochs` object containing the processed EEG epochs.
         (Useful for retrieving additional information such as channel names, times, and sampling rates.)
    """

    # if original_fps != new_fps:
    #     eegraw.resample(new_fps)

    events, event_id = mne.events_from_annotations(eegraw)
    sync_eventname = event_name  # TODO make it more generalizable
    event_id = {sync_eventname: event_id[sync_eventname]}

    # if selected_event is None:
    #     events = mne.pick_events(events, include=event_id[sync_eventname])
    # else:
    #     events = selected_event

    events = mne.pick_events(events, include=event_id[sync_eventname])
    if dropfmriframes != 0:
    # todo: if you are adapting this to a new dataset, make sure if needs dropping first several frames for mag stability
        events = events[dropfmriframes:]
    # if there are multiple mr collection events within one scan, you could pick the first of every group of events
    # e.g.:
    # events_1st = events[::30]

    # Optional: reject epoch by setting the peak to peak threshold
    reject_criteria = None
    # reject_criteria = dict(
    #     eeg=400e-6,  # 150 µV
    # )

    eeg_epoch = mne.Epochs(eegraw, events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True,
                           baseline=(None, None), reject=reject_criteria)

    eeg_epoch_data = eeg_epoch.get_data(units='uV')

    # get the indices for selected events for epoching
    select_ind = eeg_epoch.selection
    fmri_epoch = fmridataset[:, select_ind]

    # organize the eeg and fmri data into one var with lists of samples
    eeg_epoch_sample = [np.array(sample) for sample in eeg_epoch_data]
    fmri_sample = [np.array(sample) for sample in fmri_epoch.T]  # here each time point is a sample, so transpose

    if crop > 0:
        eeg_epoch_sample = [sample[:, :crop] for sample in eeg_epoch_sample]  # TODO: currently just simply crop the end
    data = {
        "eeg": eeg_epoch_sample,
        "fmri": fmri_sample
    }

    return data, eeg_epoch  # currently output eeg_epoch structure as well
                            # for infos like channels, times, sampling rate, etc
                            # in case to use it later