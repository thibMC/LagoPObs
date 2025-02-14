### Test de Classification sans obtention de paramètres acoustiques fin
### Libraries
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import hilbert, butter, sosfiltfilt, ShortTimeFFT, convolve, find_peaks, welch, convolve
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

### Fenetres pour stft
hamming_courte = hamming(wlen_courte, sym=True) # Construction de la fenêtre
blackman_harris_longue = blackmanharris(wlen_large, sym=True)  # Construction de la fenêtre

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

### Récupérer les sons
wd = "CCFlaine2017/"
sons = np.array(os.listdir(wd)) # liste des sons
# Individus via le nom de fichier
indiv = np.array([son.split("_")[1] for son in sons])
color_ind = np.zeros(len(indiv))
for (k, ind) in enumerate(np.unique(indiv)):
    print((k,ind))
    color_ind[indiv==ind] = k

### Creation de l'array contenant les résultats
# resultats = np.zeros((len(sons), 12))
resultats = np.zeros((len(sons), 8))

# Time it
import time
t1 = time.time()
# Boucle sur les sons
for son in sons:
# for son in [14,6,34,35,47]:

    ### Import du son à étudier
    fs, lago = wav.read(wd + son)
    lago = lago.astype(np.float64) # conversion en float 64 bits, peut accelerer le calcul
    t = np.arange(len(lago)) / fs # Recupère le temps

    ### Analyse
    ### Etape 1 : filtration via passe-bande
    sos = butter(10, f_filt, btype = "bandpass", fs=fs, output="sos") # Filtre passe-bande très restrictif
    lago_filt = sosfiltfilt(sos, lago)

    ### Etape 1.5 : Définition des variables dépendant de la fréquence d'échantillonnage
    win_total = int(0.01*fs) # Taille de fenêtre de lissage
    pulse_wlen = int(0.018 * fs) # Taille de fenêtre pour la détection des pulses
    pulse_width = int(0.01*fs)/2 # Largeur des pulses

    ### Etape 2 : filtration via ondelettes
    lago_wlt = wlt_denoise(lago_filt, chosen_wlt)
    plt.plot(lago_filt)
    plt.plot(lago_wlt)
    plt.title(son)
    plt.show()

    ## Étape 3 : Spectre précis en temps servant à la détection des pulses
    # Faire un spectrogram précis en temps via une fenêtre de hamming
    sTFT_precise = ShortTimeFFT(hamming_courte, hop=int((1-ovlp)*wlen_courte), fs=fs)  # configuration de la STFT
    s = sTFT_precise.stft(lago_wlt)
    spec_precis = abs(s[np.logical_and(f_filt[0] < sTFT_precise.f, sTFT_precise.f < f_filt[1])])  # on restreint à la bande de fréquences du filtre
    f_constraint_precis = sTFT_precise.f[np.logical_and(sTFT_precise.f >= f_filt[0], sTFT_precise.f < f_filt[1])]
    sum_spec = np.sum(spec_precis, axis = 0) # Somme en fréquence
    sum_spec_interp = PchipInterpolator(sTFT_precise.t(len(lago_wlt)), sum_spec)(t)

    ### Étape 4 : Détection des pulses
    peaks = find_peaks(sum_spec_interp/max(sum_spec_interp), wlen=pulse_wlen, prominence = pulse_prom, width=pulse_width)[0]
    plt.plot(sum_spec_interp)
    plt.plot(peaks, sum_spec_interp[peaks], 'or')
    plt.title(son)
    plt.show()

    ### Etape 5 : Recherche des groupes de pulses via STFT à fenêtre large
    # Search for maxima in spectrum with large hamming window => similar to sinusoidal analysis
    # see spkit sinusoidal transform :
    # https://spkit.github.io/auto_examples/signal_processing/plot_sp_sinusodal_model_analysis_synthesis.html
    hamming_longue = blackmanharris(wlen_large, sym=True)  # Construction de la fenêtre
    sTFT_large = ShortTimeFFT(hamming_longue, hop=int((1-ovlp)*wlen_large), fs=fs)  # configuration de la STFT
    s_large = sTFT_large.stft(lago_wlt)
    spec_large = abs(s_large[np.logical_and(f_filt[0] < sTFT_large.f, sTFT_large.f < f_filt[1])])  # on restreint à la bande de fréquences du filtre
    f_constraint_large = sTFT_large.f[np.logical_and(sTFT_large.f >= f_filt[0], sTFT_large.f < f_filt[1])]
    # Find the peaks in each column
    spec_loc = []
    plt.title(son)
    plt.subplot(211)
    plt.imshow(spec_large, origin='lower', aspect='auto', cmap = "gray")
    for k in range(spec_large.shape[1]):
        loc = find_peaks(spec_large[:,k], distance=spec_large.shape[0]/3, prominence=spec_prom, height=np.mean(spec_large))[0]
        # Add to array, only if at least 2 elements
        # if len(loc) >= 3:
        if len(loc) >= 2:
            spec_loc.append(loc)
            plt.plot(np.repeat(k, len(loc)), loc, "o")
        else:
            plt.plot(k, 0)
            spec_loc.append([0])
    plt.subplot(212)
    mean_ploc = [np.mean(p) for p in spec_loc]
    plt.plot(mean_ploc)
    plt.show()
    # # Interpolate and replace non nul value by 1
    mean_ploc_interp = Akima1DInterpolator(sTFT_large.t(len(lago_wlt)), mean_ploc, method="makima")(t)
    mean_ploc_interp[mean_ploc_interp != 0] = 1 # replace by np.sign ?
    plt.title(son)
    plt.plot(sum_spec_interp/sum_spec_interp.max())
    plt.plot(mean_ploc_interp/mean_ploc_interp.max())
    plt.plot(np.diff(mean_ploc_interp)) # will be used to differenciate groups of pulses
    plt.show()
    # Identify pulse groups
    # Début quand diff = 1
    groupe_debut = np.where(np.diff(mean_ploc_interp) == 1)[0]
    # Fin quand diff = -1
    groupe_fin = np.where(np.diff(mean_ploc_interp) == -1)[0]
    # Ajouter 0 au début si la voc commence directement
    if len(groupe_debut) < len(groupe_fin): groupe_debut = np.insert(groupe_debut, 0, 0)
    # Si pas de pulses à l'intérieur d'un groupe, supprimer le groupe
    filter_groupes = [k for k in range(len(groupe_debut)) if any(np.logical_and(groupe_debut[k] <= peaks, groupe_fin[k] > peaks))]
    g_deb = groupe_debut[filter_groupes]
    g_fin = groupe_fin[filter_groupes]
    # Indices des pulses pour chaque groupe
    groupes = [peaks[np.logical_and(g_deb[k] <= peaks, g_fin[k] > peaks)] for k in range(len(g_deb))]
    # Si le premier groupe a moins de 5 pulses, le supprimer
    while len(groupes[0]) < 5:
        groupes.pop(0)
    # Si le troisième groupe a moins de 5 pulses, le supprimer
    if len(groupes[2]) < 3: groupes.pop(2)
    # Verification visuelle
    plt.plot(sum_spec_interp)
    plt.plot(peaks, sum_spec_interp[peaks], 'ob')
    for k in range(len(groupes)):
        plt.plot(groupes[k], sum_spec_interp[groupes[k]], 'o', label="Groupe "+str(k+1))
    plt.legend(loc="best")
    plt.title(son)
    plt.show()

    ### Étape 6 : Calcul des paramètres temporels + nombres de pulses
    nbre_pulse_G1 = len(groupes[0]) # Nombres pulses G1
    nbre_pulse_G2 = len(groupes[1]) # Nombres pulses G2
    silence_g1g2  = (groupes[1][0] - groupes[0][-1]) / fs # Silence entre 1er et 2eme groupe
    silence_g2g3  = (groupes[2][0] - groupes[1][-1]) / fs # Silence entre 2ème et 3eme groupe

    ### Étape 7 : Calcul des paramètres du spectre d'enveloppe
    # Pour l'enveloppe, on utilisera en fait le résultats du spectre fin
    # env_wlt = np.abs(hilbert(lago_wlt))
    welch_spec1 = welch(sum_spec_interp[groupes[0][0]:groupes[0][-1]], fs, "hamming", nperseg=fs)
    welch_spec1 = np.array(welch_spec1)[:,welch_spec1[0]>15] # frequency >15
    welch_spec3 = welch(sum_spec_interp[groupes[2][0]:groupes[2][-1]], fs, "hamming", nperseg=fs)
    welch_spec3 = np.array(welch_spec3)[:,welch_spec3[0]>15]
    freq_env1 = welch_spec1[0][np.argmax(welch_spec1[1])]
    freq_env3 = welch_spec3[0][np.argmax(welch_spec3[1])]
    # plt.plot(welch_spec1[0], welch_spec1[1]/welch_spec1[1].max())
    # plt.plot(freq_env1, 1, 'ob')
    # plt.plot(welch_spec3[0], welch_spec3[1]/welch_spec3[1].max())
    # plt.plot(freq_env3, 1, 'or')
    # plt.title(son)
    # plt.xlim((0,250))
    # plt.show()
    # Rajouter les modulations du spectre d'enveloppe ?

    ### Étape 7.5 : Rajouter acceleration moyenne du pulse rate pour G1 et G3
    acc_G1 = np.mean(np.diff(np.diff(groupes[0]))) / fs
    acc_G3 = np.mean(np.diff(np.diff(groupes[2]))) / fs

    ### Étape 8 : Fréquence et modulation de fréquence
    # f_pulse = np.linspace(f_filt[0], f_filt[1], (f_filt[1]-f_filt[0])*10)
    # def calc_freq(num_groupe):
    #     F1 = []
    #     F2 = []
    #     F_mod = []
    #     for g in groupes[num_groupe]:
    #         pulse_spec = []
    #         for f in f_pulse:
    #             p = lago_wlt[max(0,g-200):min(g+200,len(lago_wlt))]
    #             conv_pulse = convolve(p, np.sin(2*np.pi*f*np.arange(len(p))/fs), "valid")
    #             rms = np.sqrt(np.mean(conv_pulse**2))
    #             pulse_spec.append(rms)
    #         pulse_spec = np.array(pulse_spec)
    #         f1 = f_pulse[f_pulse <= 1800][np.argmax(pulse_spec[f_pulse <= 1800])]
    #         f2 = f_pulse[f_pulse > 1800][np.argmax(pulse_spec[f_pulse > 1800])]
    #         F1.append(f1)
    #         F2.append(f2)
    #         peaks_spec = find_peaks(pulse_spec)[0]
    #         f_mod = np.mean(np.diff(f_pulse[peaks_spec]))
    #         F_mod.append(f_mod)
    #     return(np.mean(F1),
    #             np.mean(F2),
    #             np.mean(F_mod))
    #
    # mean_f1_g1, mean_f2_g1, mean_fmod_g1 = calc_freq(0)
    # mean_f1_g3, mean_f2_g3, mean_fmod_g3 = calc_freq(2)
    # a priori, les frequences pas si importantes au niveau de la pop

    ### Étape 9 : Sauvegarde des paramètres de chaque son
    resultats[sons == son] = np.array([
    nbre_pulse_G1, # Nombres pulses G1
    nbre_pulse_G2, # Nombres pulses G2
    silence_g1g2, # Silence entre 1er et 2eme groupe
    silence_g2g3, # Silence entre 2ème et 3eme groupe
    freq_env1, # Frequence de l'enveloppe du groupe de pulses 1'
    freq_env3, # Frequence de l'enveloppe du groupe de pulses 3'
    acc_G1, # acceleration moyenne du pulse rate pour le groupe 1
    acc_G3 # acceleration moyenne du pulse rate pour le groupe 3
    # mean_f1_g1, # Moyenne du premier formant du premier groupe de pulses
    # mean_f2_g1, # Moyenne du premier formant du premier groupe de pulses
    # mean_fmod_g1, # Moyenne de la frequence de modulation du premier groupe de pulses
    # mean_f1_g3, # Moyenne du premier formant du troisième groupe de pulses
    # mean_f2_g3, # Moyenne du deuxieme formant du troisième groupe de pulses
    # mean_fmod_g3 # Moyenne de la fréquence de modulation du troisième groupe de pulses
    ])

t2 = time.time()
dt = t2 - t1
print(dt)

# Dataset de test a mis 23 secondes pour 81 sons => 0.28 sec/son
np.where(np.isnan(resultats)) # 'MZ000022_MI_C2.wav' => probleme, voir pour rééquilibrer les niveaux? voir travail de das_unsupervised

ind = indiv[sons != 'MZ000023_MI_C2.wav']
col_ind = color_ind[sons != 'MZ000023_MI_C2.wav']
res = resultats[sons != 'MZ000023_MI_C2.wav']
res2 = res[:,2:]

# Selection de la meilleur configuration d'UMAP en non supervise
import umap
for n in range(1,11):
    plt.subplot(2,5,n)
    u = umap.UMAP(min_dist=0.1, n_neighbors=n, random_state=2).fit_transform(res)
    plt.plot(u[:,0],u[:,1], c=col_ind , cmap='hsv')
plt.show()

# umap or PCA for dimension reduction ?
# and hdbscan for clustering
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics as m
def umap_clust(spec_flat, min_size=1):
    out = umap.UMAP(min_dist=0.1, n_neighbors=5, random_state=2).fit_transform(spec_flat)
    hdbscan_labels = hdbscan.HDBSCAN(min_samples=min_size).fit_predict(out)
    return(out, hdbscan_labels)
def pca_clust(spec_flat, min_size=2):
    # out = umap.UMAP(min_dist=0.5, random_state=2).fit_transform(spec_flat)
    out = PCA(2).fit_transform(spec_flat)
    hdbscan_labels = hdbscan.HDBSCAN(min_samples=min_size).fit_predict(out)
    return(out, hdbscan_labels)
# Reduction dimension + clustering
res_pca = pca_clust(res2,2)
res_umap = umap_clust(res2,2)
plt.subplot(211)
plt.scatter(res_pca[0][:,0], res_pca[0][:,1], c=col_ind , cmap='hsv')
plt.title("PCA")
plt.subplot(212)
plt.scatter(res_umap[0][:,0], res_umap[0][:,1], c=col_ind, cmap='hsv')
plt.legend()
plt.title("UMAP")
plt.show()
# Nombre cluster
print(np.unique(res_pca[1]))
print(np.unique(res_umap[1]))
# confusion matrix
C = m.confusion_matrix(ind, res_umap[1].astype("str"))
idx = np.argmax(C, axis=0)  # re-order columns

plt.imshow(C[idx, :])
plt.colorbar()
plt.title('Confusion matrix')
plt.ylabel('Manual label')
plt.xlabel('Cluster label')
plt.show()

### Recherche de la meilleure combinaison de parametres
from itertools import combinations
comb = []
for k in range(1,res.shape[1]):
    comb.append(combinations(range(res.shape[1]), k))

comb = [c for combi in comb for c in list(combi)]
rand = []
for c in comb:
    res3 = res[:,c]
    res3_umap = umap_clust(res3,2)
    rand.append([m.adjusted_rand_score(ind,res3_umap[1].astype("str"))])
plt.plot(rand)
plt.show()
np.argmax(rand)
comb[np.argmax(rand)]

