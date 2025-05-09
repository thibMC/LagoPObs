import os
import numpy as np
import scipy.io.wavfile as wav
from soxr import resample


def filter_wavs(dir):
    """
    Get the files in a directory and return only the list of WAV files.

    Parameters
    ----------
    dir: str/Path-like object, the path of the directory.

    Returns
    -------
    wav_only: list of str, list of WAV file names in the directory.
    """
    files = os.listdir(dir)
    wav_check = [f.endswith(".wav") for f in files]
    list_wavs = np.array(files)[wav_check]
    return list_wavs


def import_wavs(list_wavs, dir, high_f):
    """
    Import WAV files from a list of WAV files. The files can have different sampling frequencies. The files will then be resampled to a sampling frequency equals to 2*(high_f+100) and then normalize by their RMS.

    Parameters
    ----------
    list_wavs: list of str, list of WAV file names.
    dir: str, path of the directory containing the files.
    high_f: int, the highest frequency of interest in the signal.

    Returns
    -------
    list_arr: list of 64 bits 1D arrays, each array being a signal corresponding to a file name in list_wavs.
    samp_freq: int, the new sampling frequency.

    """
    list_arr = []
    samp_freq = int(2 * (high_f + 100))
    for w in list_wavs:
        # WAV Import
        sf, sound = wav.read(dir + "/" + w)
        sound = sound.astype(np.float64)  # conversion in float 64 bits
        # Resample
        rs_sound = resample(sound, sf, samp_freq, "HQ")
        # Normalisation by RMS
        rs_sound /= np.sqrt(np.mean(rs_sound**2))
        list_arr.append(rs_sound)
    return list_arr, samp_freq


def pad_signals(list_arr):
    """
    Pad arrays with 0s at the end so that all arrays have the same length.

    Parameters
    ----------
    list_arr: list of 1D arrays.

    Returns
    -------
    arr_arr_pad: array of 1D arrays of the smae length.
    """
    max_len = max([len(a) for a in list_arr])
    list_arr_pad = [np.pad(l, (0, max_len - len(l))) for l in list_arr]
    arr_arr_pad = np.array(list_arr_pad)
    return arr_arr_pad
