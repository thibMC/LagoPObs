### Timing
import time

# T1 = time.time()
### Libraries
import os
from itertools import product
import numpy as np
import scipy.io.wavfile as wav
from scipy.stats import kurtosis
from scipy.signal import hilbert, butter, sosfiltfilt, ShortTimeFFT
from scipy.signal.windows import hamming, blackmanharris
import pywt  # pywavelets
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics as m
import colorcet as cc
from sklearn.mixture import GaussianMixture
import pandas as pd
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist
from sklearn.mixture import GaussianMixture
from sklearn.cluster import (
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    BisectingKMeans,
    MeanShift,
)
import keras
from keras.applications.vgg19 import preprocess_input
from multiprocessing import Pool

### Variables ne dépendant pas de la fréquence d'échantillonnage
f_filt = [950, 2800]  # bande de fréquences pour la filtration
chosen_wlt = "bior3.1"  # ondelette pour la filtration
wlen = np.array([256, 512, 1024, 2048, 4056, 8192])  # taille de fenêtres à étudier
wlen_env = 2048  # taille de fenete pour spectro imprecis
ovlp = 0.75  # overlap spectro
ovlp_env = 0.9  # overlap spectro enveloppe
max_dur = 4  # durée en secondes maximale d'un son
wd = "/home/tmc/Documents/LPO_Prestation/Analyses/CCFlaine2017"  # répertoire contenant les sons
# graine = 20250218  # graine pour UMAP
# n_features = np.arange(10, 110, 10)
# n_features = np.arange(52, 68, 2)
# n_features = [54, 55, 56]


# ### Load model
# new_input = keras.Input(shape=(shape_spec[0], shape_spec[1], 3))
# model = keras.applications.VGG19(
#     include_top=False, weights="imagenet", input_tensor=new_input, pooling="avg"
# )


### Fonctions
# Fonction wavelet denoising via swt
def wlt_denoise(signal, wlt):
    max_lvl = pywt.dwt_max_level(len(signal), wlt)
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


# Resize and merge together 2 spectrograms
def resize_merge_spec(spec1, spec2):
    s1 = spec1.shape
    s2 = spec2.shape
    spec1 /= np.max(spec1)
    spec2 /= np.max(spec2)
    final_shape = (max([s1[1], s2[1]]), max([s1[0], s2[0]]))
    rs_spec1 = np.array(Image.fromarray(spec1).resize(final_shape, Image.LANCZOS))
    rs_spec2 = np.array(Image.fromarray(spec2).resize(final_shape, Image.LANCZOS))
    merged_spec = np.vstack((rs_spec1, rs_spec2))
    return merged_spec


# Fonction globale qui prend en entrée un dossier avec des sons
# et renvoie deux éléments :
#   - un array avec les noms des sons
#   - une liste avec tous les sons fitrés
#   - un array avec les fréquences d'échantillonnage => plus de flexibilite
def filtre_sons(dossier):
    sons = np.array(os.listdir(dossier))  # liste des sons
    FS = np.zeros(len(sons))
    # Liste contenant les sons filtres
    lago_wlt_list = []
    for son in sons:
        # Import du son à étudier
        fs, lago = wav.read(dossier + "/" + son)
        FS[sons == son] = fs
        lago = lago.astype(
            np.float64
        )  # conversion en float 64 bits, peut accelerer le calcul
        # Etape 1 : filtrage via passe-bande
        sos = butter(
            10, f_filt, btype="bandpass", fs=fs, output="sos"
        )  # Filtre passe-bande très restrictif
        lago_filt = sosfiltfilt(sos, lago)
        # Etape 2 : filtrage par ondelettes
        lago_wlt = wlt_denoise(lago_filt, chosen_wlt)
        # Commence au début de la vocalise
        lago_wlt = lago_wlt[np.where(lago_wlt != 0)[0][0] :]
        # Pad à la durée max
        lago_wlt = np.pad(lago_wlt, (0, max_dur * fs - len(lago_wlt)))
        # Etape 3 : ajout à la liste
        lago_wlt_list.append(lago_wlt)
    return (sons, lago_wlt_list, FS)


# Fonction qui crée les spectros à étudier
def features_from_spec(
    signal,
    taille_fenetre,
    overlap,
    taille_env,
    overlap_env,
    fs,
    taille_max,
    freqs_interet,
):
    # Création des instances de ShortTimeFFT
    fenetre = hamming(taille_fenetre, sym=True)  # Construction de la fenêtre
    # fenetre = blackmanharris(taille_fenetre, sym=True) # Construction de la fenêtre
    sTFT = ShortTimeFFT(fenetre, hop=int((1 - overlap) * taille_fenetre), fs=fs)
    fenetre_env = hamming(taille_env, sym=True)
    sTFT_env = ShortTimeFFT(fenetre_env, hop=int((1 - overlap_env) * taille_env), fs=fs)
    # Enveloppe
    env = abs(hilbert(signal))
    # Spectro normal
    s = sTFT.stft(signal)
    spec = abs(s[np.logical_and(freqs_interet[0] < sTFT.f, sTFT.f < freqs_interet[1])])
    # Spectro de l'enveloppe
    s_env = sTFT_env.stft(env)
    spec_env = abs(s_env[sTFT_env.f <= 160])  # limite à environ la 3eme harmonique
    # Combine les deux spectros
    spec_comb = resize_merge_spec(spec, spec_env)
    # Pad à la taille max
    max_len = max(
        [len(sTFT.t(int(taille_max * fs))), len(sTFT_env.t(int(taille_max * fs)))]
    )
    spec_comb = np.pad(spec_comb, [(0, 0), (0, max_len - spec_comb.shape[1])])
    imagette = np.repeat(spec_comb[:, :, np.newaxis], 3, axis=2)
    imagette = imagette.reshape(
        (1, imagette.shape[0], imagette.shape[1], imagette.shape[2])
    )
    imagette = preprocess_input(imagette)
    new_input = keras.Input(shape=(spec_comb.shape[0], spec_comb.shape[1], 3))
    model = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_tensor=new_input, pooling="avg"
    )
    prediction = model.predict(imagette)
    return prediction.ravel()


### Test et mesure du temps d'execution
# Recuperation des sons filtrés
t1 = time.time()
noms, signaux, freq_ech = filtre_sons(wd)
t2 = time.time()
print("Filtration : " + str(t2 - t1))  # 21.33 secs
# Extractions des features
t1 = time.time()
features = []
for w in wlen:
    f = [
        features_from_spec(
            signaux[k], w, ovlp, wlen_env, ovlp_env, freq_ech[k], max_dur, f_filt
        )
        for k in range(len(signaux))
    ]
    f = np.dstack(f)
    f = np.squeeze(f, 0)
    features.append(f.T)

t2 = time.time()
print("Features : " + str(t2 - t1))  # 12.36 secs


# Récuperation des individus via le nom de fichier
indiv = np.array([n.split("_")[1] for n in noms])


def cluster_features(array_features, nom_males=indiv, n_iter=100):
    n_sons = array_features.shape[0]
    n_clust = np.arange(2, n_sons)
    cluster_number_found = np.zeros((n_iter, 6)).astype(int)
    rand_index = np.zeros((n_iter, 6))
    for i in range(n_iter):
        sil = np.zeros((4, len(n_clust)))
        for c in n_clust:
            agglo_clust = AgglomerativeClustering(n_clusters=c).fit_predict(
                array_features
            )
            gmm_clust = GaussianMixture(n_components=c).fit_predict(array_features)
            km_clust = KMeans(n_clusters=c).fit_predict(array_features)
            bkm_clust = BisectingKMeans(n_clusters=c).fit_predict(array_features)
            sil[:, c - 2] = [
                m.silhouette_score(array_features, clust)
                for clust in [agglo_clust, gmm_clust, km_clust, bkm_clust]
            ]
        cluster_number_found[i, : sil.shape[0]] = [
            n_clust[np.argmax(sil[i_s, :])] for i_s in range(sil.shape[0])
        ]
        hdbscan_clust = HDBSCAN(min_cluster_size=2).fit_predict(array_features)
        cluster_number_found[i, sil.shape[0]] = len(
            np.unique(hdbscan_clust[hdbscan_clust != -1])
        )
        ms_clust = MeanShift(n_jobs=-1).fit_predict(array_features)
        cluster_number_found[i, sil.shape[0] + 1] = max(ms_clust) + 1
        rand_index[i] = [
            m.rand_score(nom_males, clustering_best)
            for clustering_best in [
                AgglomerativeClustering(
                    n_clusters=cluster_number_found[i, 0]
                ).fit_predict(array_features),
                GaussianMixture(n_components=cluster_number_found[i, 1]).fit_predict(
                    array_features
                ),
                KMeans(n_clusters=cluster_number_found[i, 2]).fit_predict(
                    array_features
                ),
                BisectingKMeans(n_clusters=cluster_number_found[i, 3]).fit_predict(
                    array_features
                ),
                hdbscan_clust,
                ms_clust,
            ]
        ]
    cluster_number_found = pd.DataFrame(
        cluster_number_found,
        columns=[
            "agglomerative",
            "gmm",
            "kmeans",
            "bisecting_kmeans",
            "hdbscan",
            "meanshift",
        ],
    )
    rand_index = pd.DataFrame(
        rand_index,
        columns=[
            "agglomerative",
            "gmm",
            "kmeans",
            "bisecting_kmeans",
            "hdbscan",
            "meanshift",
        ],
    )
    return (cluster_number_found, rand_index)


# pool = Pool()
# t1 = time.time()
# clustering_diagnostique = pool.map(cluster_features, features)
# t2 = time.time()
# print("Multiprocessing: " + str(t2 - t1))
# pool.close()
# pool.join()


clustering_diagnostique = [cluster_features(f) for f in features]
