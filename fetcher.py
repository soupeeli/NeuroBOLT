"""Function to fetch DiFuMo atlases.

   Direct download links from OSF:

   dic = {64: https://osf.io/pqu9r/download,
          128: https://osf.io/wjvd5/download,
          256: https://osf.io/3vrct/download,
          512: https://osf.io/9b76y/download,
          1024: https://osf.io/34792/download,
          }
"""
import os
import pandas as pd

from sklearn.utils import Bunch

from nilearn.datasets.utils import (_fetch_files,
                                    _get_dataset_dir)


def fetch_difumo(dimension=64, resolution_mm=2, data_dir=None):
    """Fetch DiFuMo brain atlas

    Parameters
    ----------
    dimension : int
        Number of dimensions in the dictionary. Valid resolutions
        available are {64, 128, 256, 512, 1024}.

    resolution_mm : int
        The resolution in mm of the atlas to fetch. Valid options
        available are {2, 3}.

    data_dir : string, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

        - 'maps': str, 4D path to nifti file containing regions definition.
        - 'labels': string list containing the labels of the regions.

    References
    ----------
    Dadi, K., Varoquaux, G., Machlouzarides-Shalit, A., Gorgolewski, KJ.,
    Wassermann, D., Thirion, B., Mensch, A.
    Fine-grain atlases of functional modes for fMRI analysis,
    Paper in preparation
    """
    dic = {64: 'pqu9r',
           128: 'wjvd5',
           256: '3vrct',
           512: '9b76y',
           1024: '34792',
           }
    valid_dimensions = [64, 128, 256, 512, 1024]
    valid_resolution_mm = [2, 3]
    if dimension not in valid_dimensions:
        raise ValueError("Requested dimension={} is not available. Valid "
                         "options: {}".format(dimension, valid_dimensions))
    if resolution_mm not in valid_resolution_mm:
        raise ValueError("Requested resolution_mm={} is not available. Valid "
                         "options: {}".format(resolution_mm,
                                              valid_resolution_mm))
    url = 'https://osf.io/{}/download'.format(dic[dimension])
    opts = {'uncompress': True}

    csv_file = os.path.join('{0}', 'labels_{0}_dictionary.csv')
    if resolution_mm != 3:
        nifti_file = os.path.join('{0}', '2mm', 'maps.nii.gz')
    else:
        nifti_file = os.path.join('{0}', '3mm', 'maps.nii.gz')

    files = [(csv_file.format(dimension), url, opts),
             (nifti_file.format(dimension), url, opts)]

    dataset_name = 'difumo_atlases'

    data_dir = _get_dataset_dir(data_dir=data_dir, dataset_name=dataset_name,
                                verbose=1)

    # Download the zip file, first
    files = _fetch_files(data_dir, files, verbose=2)
    labels = pd.read_csv(files[0])

    # README
    readme_files = [('README.md', 'https://osf.io/4k9bf/download',
                    {'move': 'README.md'})]
    if not os.path.exists(os.path.join(data_dir, 'README.md')):
        _fetch_files(data_dir, readme_files, verbose=2)

    return Bunch(maps=files[1], labels=labels)
