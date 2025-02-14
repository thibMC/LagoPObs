### Libraries
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import hilbert, butter, sosfiltfilt, ShortTimeFFT, convolve, find_peaks, welch, convolve, firwin2
from scipy.signal.windows import hamming, blackmanharris
from scipy.stats import kurtosis
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
import pywt # pywavelets
import matplotlib.pyplot as plt
import pandas as pd

### Variables ne dépendant pas de la fréquence d'échantillonnage
f_filt = [950,2800] # bande de fréquences pour la filtration
chosen_wlt = "bior3.1" # ondelette pour la filtration
wlen_courte = 320 # taille de fenetre pour spectro précis
wlen_large = 1536  # taille de fenete pour spectro imprecis

ovlp = 0.75 # overlap
pulse_prom = 0.01 # prominence des pulses
spec_prom = 0.5 # Prominence of peaks in spectrum

### Récupérer les sons
wd = "/home/thibaut/Documents/LPO_Prestation/Analyses/Alpyr_2010-2013_modif/" # Adresse du dossier avec les sons
wdCC = wd + "CC/" # Repertoire des chants courts (CC)
wdCL = wd + "CL/" # Repertoire des chants longs (CL)

### Import du son à étudier
# fs, lago = wav.read(wdCC + "CC_AF1_2.wav")
# fs, lago = wav.read(wdCC + "CC_PC6_2.wav")
# fs, lago = wav.read(wdCC + "CC_PC6_1.wav")
# fs, lago = wav.read(wdCC + "CC_PC2_1.wav")
# fs, lago = wav.read(wd + "CC_PC2_4.wav")
# fs, lago = wav.read(wdCC + "CC_AF4_2.wav")
# fs, lago = wav.read(wdCC + "CC_AF3_1.wav")
# fs, lago = wav.read(wdCC + 'CC_AF9_4.wav')
# fs, lago = wav.read(wdCL + "PC8_5.wav")
# fs, lago = wav.read(wdCL + "PC8_4.wav")
# fs, lago = wav.read(wdCL + "PC8_1.wav")
# fs, lago = wav.read(wdCL + "AF4_2.wav")
# fs, lago = wav.read(wdCL + "AF4_3.wav")
fs, lago = wav.read("/home/thibaut/Documents/LPO_Prestation/Analyses/For_Test/" +
                    "SMA13707_20230618_050402-661.00-677.00-Lagopus_muta-0.14-channel_0_tresbonexempleoverlapautreespece.wav")


### Create stft instance
hamming_courte = hamming(wlen_courte, sym=True) # Construction de la fenêtre
sTFT_precise = ShortTimeFFT(hamming_courte, hop=int((1-ovlp)*wlen_courte), fs=fs)  # configuration de la STFT

### Calcul of spectral kurtosis
# See the definition in Antoni's papers
# but simplier in the matlab help: https://www.mathworks.com/help/signal/ref/spectralkurtosis.html
# Function to speed up the process
def calc_kurtosis(signal, stft_instance):
    St = stft_instance.stft(signal) # Perform stft on signals
    # Time average of different power of the stft
    S4X = np.mean(abs(St)**4, axis = 1)
    S2X = np.mean(abs(St)**2, axis = 1)
    # Kurtosis
    spectral_kurtosis = S4X / S2X**2 - 2
    return(stft_instance.f, spectral_kurtosis)

### Comparaison
# 1 : filtration via passe-bande
sos = butter(10, f_filt, btype = "bandpass", fs=fs, output="sos") # Filtre passe-bande très restrictif
lago_butter = sosfiltfilt(sos, lago)

# 2 : filtration par kurtosis spectral
kurt_lago_brut = calc_kurtosis(lago, sTFT_precise)
plt.plot(kurt_lago_brut[0], kurt_lago_brut[1])
plt.show()
# Design fir filter from spectral kurtosis
kurt_lago_brut[1][-1] = 0
taps = firwin2(int(fs/10), kurt_lago_brut[0]/(fs/2), kurt_lago_brut[1]/max(kurt_lago_brut[1]), window="hamming")
# Apply filter
lago_kurt = convolve(lago, taps, 'same')
# plt.plot(lago/max(lago))
plt.plot(lago_butter/max(lago_butter))
plt.plot(lago_kurt/max(lago_kurt))
plt.show()

# 3 : filtration avec bandpass + kurtosis
kurt_lago_butter = calc_kurtosis(lago_butter, sTFT_precise)
plt.plot(kurt_lago_brut[0], kurt_lago_brut[1])
plt.plot(kurt_lago_butter[0], kurt_lago_butter[1])
plt.show()
# Design fir filter from spectral kurtosis
kurt_lago_butter[1][-1] = 0
taps_butter = firwin2(int(fs/10), kurt_lago_butter[0]/(fs/2), kurt_lago_butter[1]/max(kurt_lago_butter[1]), window="hamming")
# Apply filter
lago_kurt_butter = convolve(lago_butter, taps_butter, 'same')
# Compare
# plt.plot(lago/max(lago))
plt.plot(lago_butter/max(lago_butter), label="BandPass")
plt.plot(lago_kurt/max(lago_kurt), label="Kurtosis")
plt.plot(lago_kurt_butter/max(lago_kurt_butter), label="Kurtosis + BandPass")
plt.legend()
plt.show()

# 4 : Test with wavelet filtering
# 4.1 : Find best wavelet
listWavelets = pywt.wavelist()
RMS = []
for wlt in listWavelets:
    try:
        wavelet = pywt.Wavelet(wlt)
    except ValueError:
        wavelet = pywt.ContinuousWavelet(wlt)
    if len(wavelet.wavefun(level=14)) >= 3:
        psi = wavelet.wavefun(level=6)[1]
        convWlt = convolve(lago, psi, "valid")
        rms = np.sqrt(np.mean(convWlt**2))
        RMS.append(rms)
    else:
        RMS.append("NA")
RMS = np.array(RMS)
listWavelets = np.array(listWavelets)
listWavelets = listWavelets[RMS != "NA"]
RMS = RMS[RMS != "NA"]
RMS = RMS.astype(np.float64)
plt.plot(listWavelets, RMS)
plt.show()
chosenWltName = listWavelets[np.argmax(RMS)]
chosenWlt = pywt.Wavelet(chosenWltName)
chosenPsi = chosenWlt.wavefun(level=5)[1]
plt.plot(chosenPsi)
plt.show()
chosenWltName == chosen_wlt  # => bior3.1

# 4.2 : Apply wavelet denoising
def wlt_denoise(signal, chosen_wlt):
    max_lvl = pywt.dwt_max_level(len(signal), chosen_wlt)
    if max_lvl > 14: max_lvl = 14
    # Decomposition
    # For swt, pad 0s to have a length propotionnate to 2**maxLvl
    signal2 = np.pad(signal, [0, 2**max_lvl - len(signal)%2**max_lvl])
    signal_swt = pywt.swt(signal2, chosen_wlt, max_lvl, trim_approx=True)
    # For each level coefficents :
    # Get statistical kurtosis (Fisher's definition)
    # if kurtosis < 0 => replace by zeros
    # else : soft thresholding by standard deviation
    signal_swt_filt = [np.zeros(len(signal_swt[0]))] # The approximated coefficents are zeroed out
    for i in range(1, max_lvl):
        kurt = kurtosis(signal_swt[i])
        if kurt < 0:
            coef_filt = np.zeros(len(signal_swt[i]))
        else:
            coef_filt = pywt.threshold(signal_swt[i], np.std(signal_swt[i]), mode="soft", substitute=0)
        signal_swt_filt.append(coef_filt)
    # Reconstrustion
    signal_wlt_filt = pywt.iswt(signal_swt_filt, chosen_wlt)
    signal_wlt_filt = signal_wlt_filt[:len(signal)]
    # Return result
    return(signal_wlt_filt)
# 4.3 Test for the 3 kind of filtering
plt.plot(wlt_denoise(lago_butter, chosenWltName), label="BandPass")
plt.plot(wlt_denoise(lago_kurt, chosenWltName), label="Kurtosis")
plt.plot(wlt_denoise(lago_kurt_butter, chosenWltName), label="Kurtosis + BandPass")
plt.legend()
plt.show()
lago_wlt = wlt_denoise(lago_butter, chosenWltName)

# 5 : Spectro
spec = sTFT_precise.spectrogram(lago_wlt)
plt.imshow(spec)
plt.show()
stft = sTFT_precise.stft(lago_wlt)
plt.imshow(abs(stft))
plt.show()
plt.plot(np.sum(abs(stft), axis=0))
plt.show()

# 6 : kurtosis spectral to identify frequencies
kurt_wlt = calc_kurtosis(lago_wlt, sTFT_precise)
plt.plot(kurt_wlt[0], kurt_wlt[1])
plt.show()

# 6 : beat tracking
plt.plot(np.sum(abs(stft), axis=0))
plt.plot(np.sum(abs(np.diff(stft, axis = 1)), axis=0))
plt.show()

# 6 : Test essentia
from essentia.standard import *
