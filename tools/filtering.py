# Functions for filtering
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import butter, sosfiltfilt
import pywt  # pyWavelets


def butterfilter(
    signal,
    sf,
    freq_band=[10, int(fs / 2 - 1)],
):
    """
    Denoise a 1D signal using a very strict bandpass filter (Order 10 butterworth).

    Parameters
    ----------
    signal: 1D array, signal to filter.
    sf: int, sampling frequency.
    freq_band: a length-2 list, containing the cut-ofrequencies [low,high].

    Returns
    -------
    signal_filt: 1D array, filtered signal
    """
    # Make the filter
    sos = butter(10, freq_band, btype="bandpass", fs=sf, output="sos")
    # Apply it
    signal_filt = sosfiltfilt(sos, signal)
    return signal_filt


def wlt_denoise(signal, wlt="bior3.1"):
    """
    Denoise a 1D signal using the SWT (Stationary wavelet transform) also known as "algorithme à trous", see Percival and Walden, 2000.

    DB Percival and AT Walden. Wavelet Methods for Time Series Analysis. Cambridge University Press, 2000.

    Parameters
    ----------
    signal: 1D array, signal to filter.
    wlt: the wavelet used to perform the SWT to filter the signal and then the inverse SWT, to reconstruct the filtered signal. See Pywavelets help for valid wavelets names.

    Returns
    -------
    signal_wlt_filt: 1D array, the filtered signal

    """
    max_lvl = pywt.swt_max_level(len(signal), wlt)
    if max_lvl > 14:
        max_lvl = 14
    # Decomposition
    # For swt, pad 0s to have a length propotionnate to 2**maxLvl
    signal2 = np.pad(signal, [0, 2**max_lvl - len(signal) % 2**max_lvl])
    signal_swt = pywt.swt(signal2, wlt, max_lvl, trim_approx=True)
    # For each level coefficents :
    # Get statistical kurtosis (Fisher's definition)
    # if kurtosis < 0 => replace by zeros
    # else : soft thresholding by standard deviation
    signal_swt_filt = [
        np.zeros(len(signal_swt[0]))
    ]  # The approximated coefficents are zeroed out
    for i in range(1, max_lvl):
        kurt = kurtosis(signal_swt[i])
        if kurt < 0:
            coef_filt = np.zeros(len(signal_swt[i]))
        else:
            coef_filt = pywt.threshold(
                signal_swt[i], np.std(signal_swt[i]), mode="soft", substitute=0
            )
        signal_swt_filt.append(coef_filt)
    # Reconstrustion
    signal_wlt_filt = pywt.iswt(signal_swt_filt, wlt)
    signal_wlt_filt = signal_wlt_filt[: len(signal)]
    # Return result
    return signal_wlt_filt
