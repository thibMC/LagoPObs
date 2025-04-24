### Calibrage des propritées de l'obtention du sous-espace
# genere par UMAP

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
from multiprocessing import Pool

### Variables ne dépendant pas de la fréquence d'échantillonnage
f_filt = [950, 2800]  # bande de fréquences pour la filtration
chosen_wlt = "bior3.1"  # ondelette pour la filtration
wlen = np.array([256, 512, 1024, 2048, 4056, 8192])  # taille de fenêtres à étudier
wlen_test, wlen_env_test = np.meshgrid(wlen, wlen)
wlen_test, wlen_env_test = wlen_test.flatten(), wlen_env_test.flatten()
ovlp = 0.75  # overlap spectro
ovlp_env = 0.9  # overlap spectro enveloppe
max_dur = 4  # durée en secondes maximale d'un son
wd = "/home/tmc/Documents/LPO_Prestation/Analyses/CCFlaine2017"  # répertoire contenant les sons
n_features = np.arange(10, 110)
# n_features = np.arange(52, 68, 2)
# n_features = [54, 55, 56]


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
def spec(
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
    return spec_comb


### Test et mesure du temps d'execution
# Recuperation des sons filtrés
t1 = time.time()
noms, signaux, freq_ech = filtre_sons(wd)
t2 = time.time()
print("Filtration : " + str(t2 - t1))  # 21.33 secs
# Transformation en spectros
t1 = time.time()
spectrogrammes = []
for w in zip(wlen_test, wlen_env_test):
    s = [
        spec(signaux[k], w[0], ovlp, w[1], ovlp_env, freq_ech[k], max_dur, f_filt)
        for k in range(len(signaux))
    ]
    spectrogrammes.append(np.array(s))

t2 = time.time()
print("Spectro : " + str(t2 - t1))  # 12.36 secs
# Récuperation des individus via le nom de fichier
indiv = np.array([n.split("_")[1] for n in noms])
# Transformation en entier
_, int_ind = np.unique(indiv, return_inverse=True)


def transfo_8bits(spectro):
    spectro = 255 * spectro / np.max(spectro)
    return spectro.astype(np.uint8)


def distance_matches(matcher, kp_des1, kp_des2, n_closest):
    # From https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/?completed=/corner-detection-python-opencv-tutorial/
    kp1, des1 = kp_des1
    kp2, des2 = kp_des2
    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    if isinstance(matcher, cv2.BFMatcher):
        matches = matcher.match(des1, des2)
        # Sort matches by distances
        matches = sorted(matches, key=lambda x: x.distance)
    elif isinstance(matcher, cv2.FlannBasedMatcher):
        matches_flann = matcher.knnMatch(des1, des2, k=2)
        matches = []
        for m1, m2 in matches_flann:
            if m1.distance < 0.5 * m2.distance:
                matches.append(m1)
        # Sort matches by distances
        matches = sorted(matches, key=lambda x: x.distance)
    # Get mean distance of the n_clostest matches
    if matches and len(matches) <= n_closest:
        dist = np.mean([m.distance for m in matches])
    elif matches and len(matches) > n_closest:
        dist = np.mean([m.distance for m in matches[:n_closest]])
    else:
        dist = 1e6
    return dist


def cluster_spectros(
    liste_spectros,
    nom_males,
    color_labels,
    detector_methode="ORB",
    methode_linkage="ward",
    n_keys=10,
    plot=True,
):
    liste_8bits = [transfo_8bits(s) for s in liste_spectros]
    if detector_methode == "SIFT":
        detector = cv2.SIFT_create()
        # FLAN_INDEX_KDTREE = 1
        # index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        # search_params = dict (checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matcher = cv2.BFMatcher(crossCheck=True)
    elif detector_methode == "ORB":
        detector = cv2.ORB_create(edgeThreshold=1)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif detector_methode == "AKAZE":
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # FLAN_INDEX_KDTREE = 1
        # index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        # search_params = dict (checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif detector_methode == "KAZE":
        detector = cv2.KAZE_create()
        # FLAN_INDEX_KDTREE = 1
        # index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        # search_params = dict (checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matcher = cv2.BFMatcher(crossCheck=True)
    elif detector_methode == "BRISK":
        detector = cv2.BRISK_create(thresh=5)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    n_specs = len(liste_spectros)
    keypoints_descriptors = [detector.detectAndCompute(l, None) for l in liste_8bits]
    dist_images = [
        distance_matches(
            matcher, keypoints_descriptors[i], keypoints_descriptors[j], n_keys
        )
        for i in range(n_specs)
        for j in range(n_specs)
    ]
    dist_images = np.array(dist_images).reshape((n_specs, n_specs))
    # ce n'est pas symetrique, donc on fait la moyenne avec la transposée
    matrice_sym = (dist_images + dist_images.T) / 2
    # matrice_sym = dist_images
    if plot:
        matrice_sym = (dist_images + dist_images.T) / 2
        d_im = squareform(matrice_sym, checks=True)
        linkage_data = linkage(d_im, method=methode_linkage)
        dendrogram(linkage_data, labels=nom_males, leaf_font_size=10)
        color_labels = "C" + color_labels.astype(int).astype(str)
        # From: https://stackoverflow.com/questions/14802048/scipy-dendrogram-leaf-label-colours
        ax = plt.gca()
        x_labels = ax.get_xmajorticklabels()
        for i, x in enumerate(x_labels):
            x.set_color(color_labels[i])
    # Clustering
    sil = []
    n_clust = np.arange(2, n_specs)
    for c in n_clust:
        # clustering_tech = AgglomerativeClustering(n_clusters=c)
        clustering_tech = GaussianMixture(n_components=c)
        # clustering_tech = KMeans(n_clusters=c)
        #     clustering_tech = BisectingKMeans(n_clusters=c)
        cluster = clustering_tech.fit_predict(dist_images)
        sil.append(m.silhouette_score(dist_images, cluster))
    # clustering_tech_best = BisectingKMeans(n_clusters=n_clust[np.argmax(sil)])
    # clustering_tech_best = KMeans(n_clusters=n_clust[np.argmax(sil)])
    # clustering_tech_best = AgglomerativeClustering(n_clusters=n_clust[np.argmax(sil)])
    clustering_tech_best = GaussianMixture(n_components=n_clust[np.argmax(sil)])
    # clustering_tech_best = HDBSCAN(min_cluster_size=2)
    # clustering_tech_best = MeanShift(n_jobs=-1)
    rand = m.rand_score(nom_males, clustering_tech_best.fit_predict(dist_images))
    return rand, n_clust[np.argmax(sil)]


# window?
t1 = time.time()
a_rand = np.zeros((len(spectrogrammes), len(n_features)))
numb_clust = np.zeros((len(spectrogrammes), len(n_features)))
for i, s in enumerate(spectrogrammes):
    for i_n, n in enumerate(n_features):
        resclust = cluster_spectros(
            s, indiv, int_ind, detector_methode="ORB", n_keys=n, plot=False
        )
        a_rand[i, i_n] = resclust[0]
        numb_clust[i, i_n] = resclust[1]

t2 = time.time()
print(t2 - t1)
wlen_lab = [",".join([str(w[0]), str(w[1])]) for w in zip(wlen_test, wlen_env_test)]
plt.subplot(211)
plt.imshow(a_rand, cmap=cc.cm.fire)
plt.xticks(ticks=np.arange(len(n_features)), labels=n_features)
plt.xlabel("Nombre de features")
plt.yticks(ticks=np.arange(len(wlen_lab)), labels=wlen_lab)
plt.ylabel("Taille de fenêtre")
# plt.suptitle("Indice de Rand")
# plt.colorbar(fraction=0.01, pad=0.04)
plt.subplot(212)
plt.imshow(numb_clust, cmap=cc.cm.fire)
plt.xticks(ticks=np.arange(len(n_features)), labels=n_features)
plt.xlabel("Nombre de features")
plt.yticks(ticks=np.arange(len(wlen_lab)), labels=wlen_lab)
plt.ylabel("Nombre de clusters")
# plt.suptitle("Indice de Rand")
# plt.colorbar(fraction=0.01, pad=0.04)
plt.show()

print("Rand max : ", np.max(a_rand))
print(np.where(a_rand == np.max(a_rand)))


### Générer et visualiser le modèle
param_best = np.where(qualite_modele == np.max(qualite_modele))
param_best = (param_best[0][0], param_best[1][0])
wlen[param_best[0]]  # => 1024


# Qualite modèle en fonction des paramètres d'umap
parametres = product(voisins, d, n_comp)
parametres = np.array([p for p in parametres])
qualite_modele = np.zeros((len(wlen), parametres.shape[0]))
t1 = time.time()
for i, sp in enumerate(spectrogrammes):
    s_flat = np.array([s.T.ravel() for s in sp])
    qualite_UMAP = np.array(
        [eval_UMAP(s_flat, int_ind, int(p[0]), p[1], int(p[2])) for p in parametres]
    )
    qualite_modele[i] = qualite_UMAP
    print(str(i) + " / " + str(len(spectrogrammes)))

t2 = time.time()
print("Qualité modèle : " + str(t2 - t1))  # 71.16 secs
T2 = time.time()
print("Temps total d'execution" + str(T2 - T1))
# Sauve en CSV
qualite_df = pd.DataFrame(qualite_modele)
qualite_df.to_csv("resultats_calibrage_UMAP.csv", sep=",")
# Pour reprendre le travail après sauvegarde
qualite_modele = np.loadtxt("resultats_calibrage_UMAP.csv", delimiter=",", skiprows=1)
qualite_modele = qualite_modele[:, 1:]
# Resultats sous forme de matrice
plt.imshow(qualite_modele, cmap=cc.cm.fire)
plt.xticks(ticks=np.arange(len(parametres)))
plt.xlabel("Nombre de voisins pour UMAP")
plt.yticks(ticks=np.arange(len(wlen)), labels=wlen)
plt.ylabel("Taille de fenêtre")
plt.suptitle("Indice silhouette")
plt.colorbar(fraction=0.01, pad=0.04)
plt.show()

print("Rand max : ", np.max(qualite_modele))
print(np.where(qualite_modele == np.max(qualite_modele)))

### Générer et visualiser le modèle
param_best = np.where(qualite_modele == np.max(qualite_modele))
param_best = (param_best[0][0], param_best[1][0])
wlen[param_best[0]]  # => 1024
param_select = parametres[param_best[1]].astype(
    int
)  # => voisins : 12/distance : 0/nombre composantes : 52
spectro_best = spectrogrammes[param_best[0]]
spectro_best_flat = np.array([s.T.ravel() for s in spectro_best])
model_UMAP_CC = umap.UMAP(
    n_components=param_select[2],
    min_dist=param_select[1],
    n_neighbors=param_select[0],
    random_state=graine,
).fit(spectro_best_flat, y=int_ind)
sous_espace_CC = model_UMAP_CC.transform(spectro_best_flat)
plt.scatter(sous_espace_CC[:, 0], sous_espace_CC[:, 1], c=int_ind)
plt.show()

### Clustering dessus
from sklearn.mixture import GaussianMixture

silhouette_CC = []
for n in range(2, sous_espace_CC.shape[0]):
    g_alp = GaussianMixture(n).fit_predict(sous_espace_CC)
    silhouette_CC.append(m.silhouette_score(sous_espace_CC, g_alp))
plt.plot(np.arange(2, sous_espace_CC.shape[0]), silhouette_CC)
plt.show()
gmm_CC = GaussianMixture(9).fit_predict(sous_espace_CC)
cm_CC = m.confusion_matrix(gmm_CC, int_ind)

### Test sur jeu de données autres : Alpyr
# Repertoire de travail
wd_test = "Alpyr_2010-2013_modif/CC"
# Recuperation des sons filtrés
t1 = time.time()
alpyr_noms, alpyr, alpyr_fs = filtre_sons(wd_test)
t2 = time.time()
print("Filtration : " + str(t2 - t1))  # 21.33 secs
# Récupération des populations + individus
alpyr_indiv = np.array([son.split("_")[1] for son in alpyr_noms])
alpyr_int_ind = np.zeros(len(alpyr_indiv))
for k, ind in enumerate(np.unique(alpyr_indiv)):
    print((k, ind))
    alpyr_int_ind[alpyr_indiv == ind] = k
alpyr_pop = np.array([n[0] for n in alpyr_indiv])
alpyr_int_pop = np.zeros(len(alpyr_pop))
alpyr_int_pop[alpyr_pop == "P"] = 1
# Transformation en spectros
t1 = time.time()
spectro_alpyr = [
    spec(
        alpyr[k],
        wlen[param_best[0]],
        ovlp,
        wlen_env,
        ovlp_env,
        alpyr_fs[k],
        max_dur,
        f_filt,
    )
    for k in range(len(alpyr))
]
spectro_alpyr_flat = np.array([s.T.ravel() for s in spectro_alpyr])
t2 = time.time()
print("Spectro : " + str(t2 - t1))  # 12.36 secs
t1 = time.time()
sous_espace_alpyr = model_UMAP_CC.transform(spectro_alpyr_flat)
t2 = time.time()
print("UMAP : " + str(t2 - t1))  # 12.36 secs
plt.subplot(221)
plt.suptitle("Répartition des populations")
plt.scatter(sous_espace_alpyr[:, 0], sous_espace_alpyr[:, 1], c=alpyr_int_pop)
plt.subplot(222)
plt.suptitle("Répartition de tous les individus")
plt.scatter(sous_espace_alpyr[:, 0], sous_espace_alpyr[:, 1], c=alpyr_int_ind)
plt.subplot(223)
plt.suptitle("Répartition des individus des Alpes")
plt.scatter(
    sous_espace_alpyr[alpyr_pop == "A", 0],
    sous_espace_alpyr[alpyr_pop == "A", 1],
    c=alpyr_int_ind[alpyr_pop == "A"],
)
plt.subplot(224)
plt.suptitle("Répartition des individus des Pyrénées")
plt.scatter(
    sous_espace_alpyr[alpyr_pop == "P", 0],
    sous_espace_alpyr[alpyr_pop == "P", 1],
    c=alpyr_int_ind[alpyr_pop == "P"],
)
plt.show()

### Clustering
from sklearn.cluster import HDBSCAN

hdb = HDBSCAN(min_cluster_size=2)
cluster_Alp = hdb.fit_predict(sous_espace_alpyr[alpyr_pop == "A", :])
print(np.unique(cluster_Alp))
print("Nombre individus Alpes : " + str(len(np.unique(alpyr_indiv[alpyr_pop == "A"]))))
print("Nombre cluster Alpes : " + str(len(np.unique(cluster_Alp))))

cluster_Pyr = hdb.fit_predict(sous_espace_alpyr[alpyr_pop == "P", :])
print(np.unique(cluster_Pyr))
print(
    "Nombre individus Pyrénées : " + str(len(np.unique(alpyr_indiv[alpyr_pop == "P"])))
)
print("Nombre clusters Pyrénées : " + str(len(np.unique(cluster_Pyr))))

cm_alp = m.confusion_matrix(cluster_Alp, alpyr_int_ind[alpyr_pop == "A"])
cm_alp = cm_alp[~np.all(cm_alp == 0, axis=1), :]
cm_alp = cm_alp[:, ~np.all(cm_alp == 0, axis=0)]
cm_pyr = m.confusion_matrix(cluster_Pyr, alpyr_int_ind[alpyr_pop == "P"])
cm_pyr = cm_pyr[~np.all(cm_pyr == 0, axis=1), :]
cm_pyr = cm_pyr[:, ~np.all(cm_pyr == 0, axis=0)]
plt.subplot(121)
plt.imshow(cm_alp)
plt.subplot(122)
plt.imshow(cm_pyr)
plt.show()

# Gaussian mixture
from sklearn.mixture import GaussianMixture

n_maxA = len(alpyr_pop[alpyr_pop == "A"])
n_maxP = len(alpyr_pop[alpyr_pop == "P"])
sil_pyr = []
sil_alp = []
for n in range(2, n_maxA):
    g_alp = GaussianMixture(n).fit_predict(sous_espace_alpyr[alpyr_pop == "A", :])
    sil_alp.append(m.silhouette_score(sous_espace_alpyr[alpyr_pop == "A", :], g_alp))
for n in range(2, n_maxP):
    g_pyr = GaussianMixture(n).fit_predict(sous_espace_alpyr[alpyr_pop == "P", :])
    sil_pyr.append(m.silhouette_score(sous_espace_alpyr[alpyr_pop == "P", :], g_pyr))
plt.plot(np.arange(2, n_maxA), sil_alp)
plt.plot(np.arange(2, n_maxP), sil_pyr)
plt.show()

gmm_alp = GaussianMixture(6).fit_predict(sous_espace_alpyr[alpyr_pop == "A", :])
gmm_pyr = GaussianMixture(2).fit_predict(sous_espace_alpyr[alpyr_pop == "P", :])
cm_alp2 = m.confusion_matrix(gmm_alp, alpyr_int_ind[alpyr_pop == "A"])
cm_pyr2 = m.confusion_matrix(gmm_pyr, alpyr_int_ind[alpyr_pop == "P"])
plt.subplot(121)
plt.imshow(cm_alp2)
plt.subplot(122)
plt.imshow(cm_pyr2)
plt.show()
