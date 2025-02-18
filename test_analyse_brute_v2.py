### Test de Classification sans obtention de paramètres acoustiques fin
# le traitement des spectros (log-sclae et padding) et
# l'algo de reduction + clustering sont inspîrés de :
# https://janclemenslab.org/das/unsupervised/birds.html

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
from PIL import Image
import hdbscan
import umap
import colorcet as cc
from sklearn.decomposition import PCA
import librosa

### Variables ne dépendant pas de la fréquence d'échantillonnage
f_filt = [950,2800] # bande de fréquences pour la filtration
chosen_wlt = "bior3.1" # ondelette pour la filtration
wlen_courte = 320 # taille de fenetre pour spectro précis
wlen_large = 1536  # taille de fenete pour spectro imprecis
wlen_giga_large = 8192
ovlp = 0.75 # overlap spectro
ovlp_env = 0.9 # overlap spectro enveloppe

### Fenetres pour stft
hamming_courte = hamming(wlen_courte, sym=True) # Construction de la fenêtre
blackman_harris_longue = blackmanharris(wlen_large, sym=True)  # Construction de la fenêtre
hamming_giga_large = hamming(wlen_giga_large, sym=True) # Construction de la fenêtre

### Fonction wavelet denoising via swt
def wlt_denoise(signal, wlt):
    max_lvl = pywt.dwt_max_level(len(signal), wlt)
    if max_lvl > 14: max_lvl = 14
    # Decomposition
    # For swt, pad 0s to have a length propotionnate to 2**maxLvl
    signal2 = np.pad(signal, [0, 2**max_lvl - len(signal)%2**max_lvl])
    signal_swt = pywt.swt(signal2, wlt, max_lvl, trim_approx=True)
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
    signal_wlt_filt = pywt.iswt(signal_swt_filt, wlt)
    signal_wlt_filt = signal_wlt_filt[:len(signal)]
    # Return result
    return(signal_wlt_filt)

### Log resize spectrogram
# from das_unsupervised : https://github.com/janclemenslab/das_unsupervised/blob/master/src/das_unsupervised/spec_utils.py
def log_resize_spec(spec, scaling_factor=10):
    """Log resize time axis. SCALING_FACTOR determines nonlinearity of scaling."""
    #from https://github.com/timsainb/avgn_paper
    resize_shape = [int(np.log(spec.shape[0]) * scaling_factor), spec.shape[1]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.LANCZOS))
    return(resize_spec.T)

### Resize and merge together 2 spectrogram
def resize_merge_spec(spec1, spec2):
    s1 = spec1.shape
    s2 = spec2.shape
    spec1 /= np.max(spec1)
    spec2 /= np.max(spec2)
    final_shape = (max([s1[1],s2[1]]),
                max([s1[0],s2[0]]))
    rs_spec1 = np.array(Image.fromarray(spec1).resize(final_shape, Image.LANCZOS))
    rs_spec2 = np.array(Image.fromarray(spec2).resize(final_shape, Image.LANCZOS))
    merged_spec = np.vstack((rs_spec1, rs_spec2))
    return(merged_spec)


### Récupérer les sons
# wd = "Alpyr_2010-2013_modif/" # Adresse du dossier avec les sons
# wdCC = wd + "CC/" # Repertoire des chants courts (CC)
# wdCL = wd + "CL/" # Repertoire des chants longs (CL)
# sons = os.listdir(wdCC) # liste des sons
wd = "CCFlaine2017/"
sons = os.listdir(wd) # liste des sons
# Individus et region via le nom de fichier
indiv = np.array([son.split("_")[1] for son in sons])
color_ind = np.zeros(len(indiv))
for (k, ind) in enumerate(np.unique(indiv)):
    print((k,ind))
    color_ind[indiv==ind] = k

### Creation de la liste contenant les resultats de la stft
results_stft_courte = []
results_stft_longue = []

### Liste contenant les sons filtres
lago_wlt_list = []

### Boucle sur les sons pour avoir la stft
for son in sons:
# for son in [14,6,34,35,47]:

    ### Import du son à étudier
    # fs, lago = wav.read(wdCC + son)
    fs, lago = wav.read(wd + son)
    lago = lago.astype(np.float64) # conversion en float 64 bits, peut accelerer le calcul
    t = np.arange(len(lago)) / fs # Recupère le temps
    ### Etape 1 : filtration via passe-bande
    sos = butter(10, f_filt, btype = "bandpass", fs=fs, output="sos") # Filtre passe-bande très restrictif
    lago_filt = sosfiltfilt(sos, lago)
    # check passe bande
    # plt.plot(lago)
    # plt.plot(lago_filt)
    # plt.show()
    ### Etape 2 : filtration par ondelettes
    lago_wlt = wlt_denoise(lago_filt, chosen_wlt)
    # check
    # plt.plot(lago_filt)
    # plt.plot(lago_wlt)
    # plt.show()
    ### Etape 3 : calcul de l'enveloppe et stft de l'enveloppe
    env_lago = abs(hilbert(lago_wlt))
    sTFT_giga_large = ShortTimeFFT(hamming_giga_large, hop=int((1-ovlp_env)*wlen_giga_large), fs=fs)
    s_env = sTFT_giga_large.stft(env_lago)
    spec_env = abs(s_env[sTFT_giga_large.f <= 160])
    ### Etape 4 : stft
    # Test de configurations
    # Instances de la classe scipy
    sTFT_precise = ShortTimeFFT(hamming_courte, hop=int((1-ovlp)*wlen_courte), fs=fs)
    sTFT_large = ShortTimeFFT(blackman_harris_longue, hop=int((1-ovlp)*wlen_large), fs=fs)
    # Application
    s_precis = sTFT_precise.stft(lago_wlt)
    # on restreint à la bande de fréquences du filtre
    spec_precis = abs(s_precis[np.logical_and(f_filt[0] < sTFT_precise.f, sTFT_precise.f < f_filt[1])])
    f_precis = sTFT_precise.f[np.logical_and(sTFT_precise.f >= f_filt[0], sTFT_precise.f < f_filt[1])]
    # idem pour la stft à large fenêtre
    # s_large = sTFT_large.stft(lago_wlt)
    # spec_large = abs(s_large[np.logical_and(f_filt[0] < sTFT_large.f, sTFT_large.f < f_filt[1])])
    # f_large = sTFT_large.f[np.logical_and(sTFT_large.f >= f_filt[0], sTFT_large.f < f_filt[1])]
    s_large = sTFT_giga_large.stft(lago_wlt)
    spec_large = abs(s_large[np.logical_and(f_filt[0] < sTFT_giga_large.f, sTFT_giga_large.f < f_filt[1])])
    f_large = sTFT_giga_large.f[np.logical_and(sTFT_giga_large.f >= f_filt[0], sTFT_giga_large.f < f_filt[1])]
    ### Etape 5 : merge stft + stft de l'enveloppe
    spec_precis = resize_merge_spec(spec_precis, spec_env)
    spec_large = resize_merge_spec(spec_large, spec_env)
    ### Etape 6 : ajout aux listes de resultats
    results_stft_courte.append(spec_precis)
    results_stft_longue.append(spec_large)
    #
    ###  Etape 6 : ajout du sons filtre a une liste pour d'autres tests
    lago_wlt_list.append(lago_wlt)


### Preparation des stft
# Prep des spectro
# Log-scale
courte_rs = [log_resize_spec(s) for s in results_stft_courte]
longue_rs = [log_resize_spec(s) for s in results_stft_longue]
# Regarde qui est le plus long
i_max_court = np.argmax([r.shape[1] for r in results_stft_courte])
i_max_long = np.argmax([r.shape[1] for r in results_stft_longue])

# check if same
if i_max_court == i_max_long:
    print("same !")
    i_max = i_max_court
else:
    print("!!! NOT THE SAME !!!")

# Pad to have all sounds equal to the targeted duration
courte = [np.pad(s, [(0,0), (0,results_stft_courte[i_max].shape[1]-s.shape[1])]) for s in results_stft_courte]
courte_rs = [np.pad(s, [(0,0), (0,courte_rs[i_max].shape[1]-s.shape[1])]) for s in courte_rs]
longue = [np.pad(s, [(0,0), (0,results_stft_longue[i_max].shape[1]-s.shape[1])]) for s in results_stft_longue]
longue_rs = [np.pad(s, [(0,0), (0,longue_rs[i_max].shape[1]-s.shape[1])]) for s in longue_rs]
# Flatten
courte_flat = np.array([c.T.ravel() for c in courte])
courte_rs_flat = np.array([c.T.ravel() for c in courte_rs])
longue_flat = np.array([l.T.ravel() for l in longue])
longue_rs_flat = np.array([l.T.ravel() for l in longue_rs])

### Meilleure configuration d'UMAP pour la séparation des sons
# non supervise
plt.suptitle("STFT + Env Courte")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(courte_flat)
    plt.scatter(u[:,0],u[:,1], c=color_ind, cmap="hsv")
    plt.title(str(n) + "voisins")
plt.show()
plt.suptitle("STFT + Env Courte Log")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(courte_rs_flat)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
plt.suptitle("STFT + Env Longue")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(longue_flat)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
plt.suptitle("STFT + Env Longue Log")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(longue_rs_flat)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
# supervise
plt.suptitle("Supervisée STFT + Env Courte")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(courte_flat, y=color_ind)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
plt.suptitle("Supervisée STFT + Env Courte Log")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(courte_rs_flat, y=color_ind)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
plt.suptitle("Supervisée STFT + Env Longue")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(longue_flat, y=color_ind)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
plt.suptitle("Supervisée STFT + Env Longue Log")
for n in range(2,26):
    plt.subplot(3,8,n-1)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(longue_rs_flat, y=color_ind)
    plt.scatter(u[:,0],u[:,1], c=color_ind , cmap='hsv')
    plt.title(str(n) + "voisins")
plt.show()
