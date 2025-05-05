# Function to calculate images based on the STFT (Short-time Fourier transform) and envelope spectrogram
import numpy as np
from scipy.signal import hilbert, ShortTimeFFT
from scipy.signal.windows import hamming
from PIL import Image


def draw_specs(
    signal,
    win_len,
    overlap,
    win_len_env,
    overlap_env,
    sf,
    freqs_of_interest,
):
    """
    Perform the STFT (Short-time Fourier transform) on a 1D signal and its envelope and merge the two together.
    The STFTs are performed using a hamming window.

    Parameters
    ----------
    signal: 1D array, signal of interest.
    win_len: int, window length of the stft performed on the signal.
    overlap: float, overlap (in %) of the stft performed on the signal.
    win_len_env: int, window length of the stft performed on the envelope.
    overlap_env: float, overlap (in %) of the stft performed on the envelope.
    sf: int, sampling frequency.
    freqs_of_interest: a length-2 list, containing the cut-ofrequencies [low,high] of the frequencies of interest, the frequencies outside this band will be excluded.

    Returns
    -------
    spec_comb = 2D array, with the two 2D arrays resulting from the STFTs resized and merged.

    """
    # Convert overlaps from percent to ratio
    ovlp = overlap / 100
    ovlp_env = overlap_env / 100
    # Creation of scipy ShortTimeFFT instances
    window = hamming(win_len, sym=True)
    st_ft = ShortTimeFFT(window, hop=int((1 - ovlp) * win_len), fs=sf)
    window_env = hamming(win_len_env, sym=True)
    st_ft_env = ShortTimeFFT(window_env, hop=int((1 - ovlp_env) * win_len_env), fs=fs)
    # Envelope
    env = calc_env(signal)
    # STFT on signal
    s = st_ft.stft(signal)
    # exclude frequencies outside of the target frequency range
    spec = abs(
        s[
            np.logical_and(
                freqs_of_interest[0] < st_ft.f, st_ft.f < freqs_of_interest[1]
            )
        ]
    )
    # STFT of envelope
    s_env = st_ft_env.stft(env)
    # for envelope, limit around the third harmonic of ptarmigan sound
    spec_env = abs(s_env[st_ft_env.f <= 160])
    # Combine the two 2D arrays
    spec_comb = resize_merge_spec(spec, spec_env)
    return spec_comb


def calc_env(signal):
    """
    Calculate the envelope of a 1D signal using the hilbert transform,
    see scipy.signal.hilbert.

    Parameters
    ----------
    signal: 1D array, the signal.

    Returns
    ------
    env: 1D array, the signal envelope.
    """
    env = abs(hilbert(signal))
    return env


def resize_merge_spec(spec1, spec2):
    """
    Resize and merge two 2D arrays, here the results of two stfts.

    Parameters
    ----------
    spec1: 2D array, first spectrogram.
    spec2: 2D array, second spectrogram.

    Returns
    -------
    merged_spec: 2D array, with the two arrays resized and merged.

    """
    s1 = spec1.shape
    s2 = spec2.shape
    spec1 /= np.max(spec1)
    spec2 /= np.max(spec2)
    final_shape = (max([s1[1], s2[1]]), max([s1[0], s2[0]]))
    rs_spec1 = np.array(Image.fromarray(spec1).resize(final_shape, Image.LANCZOS))
    rs_spec2 = np.array(Image.fromarray(spec2).resize(final_shape, Image.LANCZOS))
    merged_spec = np.vstack((rs_spec1, rs_spec2))
    return merged_spec
