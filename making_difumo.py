"""
Script for generating fMRI functional mode signal using Difumo
https://parietal-inria.github.io/DiFuMo/

If you intend to apply this script to a new dataset, ensure it is
adapted to accommodate the specific preprocessing steps required for that dataset.

"""
import glob
import os

import numpy as np
import pandas as pd
from nilearn import image, masking
from nilearn.input_data import NiftiMapsMasker, NiftiLabelsMasker

from fetcher import fetch_difumo


# functions from nilearn to generate polynomial drift matrix
def poly_drift(order, frame_times):
    """Create a polynomial drift matrix.

    Parameters
    ----------
    order : int,
        Number of polynomials in the drift model.

    frame_times : array of shape(n_scans),
        Time stamps used to sample polynomials.

    Returns
    -------
    pol : ndarray, shape(n_scans, order + 1)
         Estimated polynomial drifts plus a constant regressor.

    """
    order = int(order)
    pol = np.zeros((np.size(frame_times), order + 1))
    tmax = float(frame_times.max())
    for k in range(order + 1):
        pol[:, k] = (frame_times / tmax) ** k
    pol = orthogonalize(pol)
    pol = np.hstack((pol[:, 1:], pol[:, :1]))
    return pol


def orthogonalize(X):
    """Orthogonalize every column of design `X` w.r.t preceding columns.

    Parameters
    ----------
    X : array of shape(n, p)
       The data to be orthogonalized.

    Returns
    -------
    X : array of shape(n, p)
       The data after orthogonalization.

    Notes
    -----
    X is changed in place. The columns are not normalized.

    """
    if X.size == X.shape[0]:
        return X

    from scipy.linalg import pinv

    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))

    return X

# Configuration
remove_poly = False  # Whether to remove polynomial drift
remove_glb = False   # Unused variable; consider removing if unnecessary
t_r = 2.1            # Repetition time of the fMRI scans
N_roi = 64           # Number of ROIs for the Difumo atlas (options: 64, 128, 256, 512, 1024)
resolution_mm = 2    # Spatial resolution of the Difumo atlas

# Paths
data_path = './data/fMRI/'
motion_pathroot = './data/motions/'
save_pathroot = './data/fMRI_difumo/'
endname = '*.nii'
scanname_length = 13  # Number of characters to pick from file name for the scan name
paths = glob.glob(os.path.join(data_path, endname))

# load difumo masks
maps_img = fetch_difumo(dimension=N_roi, resolution_mm=resolution_mm).maps
maps_labels = fetch_difumo(dimension=N_roi,
                           resolution_mm=resolution_mm).labels  # this labels contain overlap info with other atlas like Yeos
Difumo_roinames = maps_labels['Difumo_names']


maps_masker = NiftiMapsMasker(maps_img=maps_img,
                              verbose=1,
                              detrend=True,
                              standardize=True,
                              standardize_confounds=True,
                              high_variance_confounds=False,
                              )

for f in paths:  # f = './fMRI.nii.gz'
    scanname = os.path.basename(f)[:scanname_length]  # e.g., vcon05-scan01, change this to the length of your scan name if new dataset comes in
    if remove_poly:
        savepath = os.path.join(save_pathroot, f'{scanname}_difumo_roi_polyclean.pkl')
    else:
        savepath = os.path.join(save_pathroot, f'{scanname}_difumo_roi.pkl')

    if os.path.exists(savepath):
        print(f"{scanname} is already processed")
        continue
    print(f"Start processing {scanname}......")

    # should also load confounds, like motion
    motion_path = os.path.join(motion_pathroot, f'{scanname}_ecr_e2.volreg_par')
    if not os.path.exists(motion_path):
        print(f"{scanname} doesn't have motion files")
        continue
    motion_confound = pd.read_csv(motion_path, sep='  ', header=None, engine='python')

    if remove_poly:
        time_seq = np.arange(1, len(motion_confound) + 1)
        poly_confound = poly_drift(4, time_seq)  # order, timeframe
        poly_df = pd.DataFrame(poly_confound[:, :4])
        confounds = pd.concat([motion_confound, poly_df], axis=1)
    else:
        confounds = motion_confound

    signals = maps_masker.fit_transform(f, confounds=confounds)  # it extracts the roi signal first and then remove confounds

    # also include global (also should consider confounds)
    mean_img = image.mean_img(f)
    mask = masking.compute_epi_mask(mean_img)
    masker_global_signal = NiftiLabelsMasker(mask, 'global_signal',
                                             detrend=False,
                                             standardize=True,
                                             standardize_confounds=True,  # default is true
                                             t_r=t_r)

    ts_global_signal_clean = masker_global_signal.fit_transform(f, confounds=confounds)
    ts_global_signal = masker_global_signal.fit_transform(f)

    # organized into pandas format with name of labels as column name
    df_fmri = pd.DataFrame(signals, columns=Difumo_roinames)
    df_fmri['global signal clean'] = ts_global_signal_clean
    df_fmri['global signal raw'] = ts_global_signal

    df_fmri.to_pickle(savepath)
    # read pickle: pd.read_pickle(filename)

