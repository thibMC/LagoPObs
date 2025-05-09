### Timing
import time

### Libraries
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.stats import kurtosis
from scipy.signal import hilbert, butter, sosfiltfilt, ShortTimeFFT
from scipy.signal.windows import hamming, blackmanharris
import pywt  # pywavelets
import matplotlib.pyplot as plt
from PIL import Image
import sklearn.metrics as m
from sklearn.mixture import GaussianMixture
import pandas as pd
import cv2
import colorcet as cc
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist
from sklearn.mixture import GaussianMixture
from sklearn.cluster import (
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    BisectingKMeans,
    MeanShift,
    AffinityPropagation,
)
from multiprocessing import Pool
from skimage.transform import AffineTransform
from skimage.measure import ransac
from soxr import resample


### Variables ne dépendant pas de la fréquence d'échantillonnage
f_filt = [950, 2800]  # bande de fréquences pour la filtration
chosen_wlt = "bior3.1"  # ondelette pour la filtration
wlen = np.arange(0.01, 0.37, 0.015)  # => transfo en temps, plus flexible
wlen = np.geomspace(0.01, 0.37, 10)
# taille de fenêtres à étudier
wlen_test, wlen_env_test = np.meshgrid(wlen, wlen)
wlen_test, wlen_env_test = wlen_test.flatten(), wlen_env_test.flatten()
# wlen_test = np.linspace(0.02, 0.07, 10)
# wlen_env_test = np.geomspace(0.1, 0.15, 5)
wlen_lab = [",".join([str(w[0]), str(w[1])]) for w in zip(wlen_test, wlen_env_test)]
ovlp = 0.75  # overlap spectro
ovlp_env = 0.9  # overlap spectro enveloppe
wd = "/home/tmc/Documents/LPO_Prestation/Analyses/CCFlaine2017"  # répertoire contenant les sons
# n_features = np.arange(55, 65)
# n_features = np.arange(10, 500)
n_features = np.arange(10, 110)
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
    FS = int(2 * (f_filt[1] + 100))
    # Liste contenant les sons filtres
    lago_wlt_list = []
    # List of the length
    len_list = []
    for son in sons:
        # Import du son à étudier
        fs, lago = wav.read(dossier + "/" + son)
        lago = lago.astype(
            np.float64
        )  # conversion en float 64 bits, peut accelerer le calcul
        # Resample
        lago = resample(lago, fs, FS, "HQ")
        # Normalisation by RMS
        lago /= np.sqrt(np.mean(lago**2))
        # Etape 1 : filtrage via passe-bande
        sos = butter(
            10, f_filt, btype="bandpass", fs=FS, output="sos"
        )  # Filtre passe-bande très restrictif
        lago_filt = sosfiltfilt(sos, lago)
        # Etape 2 : filtrage par ondelettes
        lago_wlt = wlt_denoise(lago_filt, chosen_wlt)
        # Commence au début de la vocalise
        lago_wlt = lago_wlt[np.where(lago_wlt != 0)[0][0] :]
        # Etape 3 : ajout aux listes
        lago_wlt_list.append(lago_wlt)
        len_list.append(len(lago_wlt))
    # Etape 4 : pad à la durée max
    max_len = max(len_list)
    lago_wlt_list = [np.pad(l, (0, max_len - len(l))) for l in lago_wlt_list]
    return (sons, lago_wlt_list, FS)


# Fonction qui crée les spectros à étudier
def spec(
    signal,
    taille_fenetre,
    overlap,
    taille_env,
    overlap_env,
    fs,
    freqs_interet,
):
    # Création des instances de ShortTimeFFT
    taille_fenetre *= fs
    taille_fenetre = int(taille_fenetre)
    taille_env *= fs
    taille_env = int(taille_env)
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
        spec(signaux[k], w[0], ovlp, w[1], ovlp_env, freq_ech, f_filt)
        for k in range(len(signaux))
    ]
    spectrogrammes.append(np.array(s))

t2 = time.time()
print("Spectro : " + str(t2 - t1))  # 12.36 secs
# Récuperation des individus via le nom de fichier
indiv = np.array([n.split("_")[1] for n in noms])
# Transformation en entier
uniq_ind, int_ind = np.unique(indiv, return_inverse=True)


def transfo_8bits(spectro):
    # set max of image at 255
    spec_8bits = 255 * abs(spectro / np.max(spectro))
    return spec_8bits.astype(np.uint8)


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
            if m1.distance < 0.75 * m2.distance:
                matches.append(m1)
        # Sort matches by distances
        matches = sorted(matches, key=lambda x: x.distance)
    # Get mean distance of the n_clostest matches
    if matches and len(matches) <= n_closest:
        dist = np.mean([m.distance for m in matches])
    elif matches and len(matches) > n_closest:
        dist = np.mean([m.distance for m in matches[:n_closest]])
    else:
        dist = 1e10
    # if matches:
    #     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #     matches_mask = np.array(mask.ravel().tolist())
    #     matches = [m for m, mm in zip(matches, matches_mask) if mm == 1]
    #     if matches and len(matches) <= n_closest:
    #         dist = np.mean([m.distance for m in matches])
    #     elif matches and len(matches) > n_closest:
    #         dist = np.mean([m.distance for m in matches[:n_closest]])
    # else:
    #     dist = 1e10
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
        detector = cv2.ORB_create(edgeThreshold=1, nlevels=7)
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
    # for c in n_clust:
    #     # clustering_tech = AgglomerativeClustering(n_clusters=c)
    #     clustering_tech = GaussianMixture(n_components=c)
    #     # clustering_tech = KMeans(n_clusters=c)
    #     #     clustering_tech = BisectingKMeans(n_clusters=c)
    #     cluster = clustering_tech.fit_predict(dist_images)
    #     sil.append(m.silhouette_score(dist_images, cluster))
    # clustering_tech_best = BisectingKMeans(n_clusters=n_clust[np.argmax(sil)])
    # clustering_tech_best = KMeans(n_clusters=n_clust[np.argmax(sil)])
    # clustering_tech_best = AgglomerativeClustering(n_clusters=n_clust[np.argmax(sil)])
    # clustering_tech_best = GaussianMixture(n_components=n_clust[np.argmax(sil)])
    # clustering_tech_best = HDBSCAN(min_cluster_size=2)
    # clustering_tech_best = MeanShift(n_jobs=-1)
    clustering_tech_best = AffinityPropagation()
    clust_final = clustering_tech_best.fit_predict(dist_images)
    rand = m.rand_score(nom_males, clust_final)
    # return rand, n_clust[np.argmax(sil)]
    rand = m.completeness_score(nom_males, clust_final)
    return rand, len(np.unique(clust_final))


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

rand_df = pd.DataFrame(
    a_rand, columns=n_features.astype(str) + "_features", index=wlen_lab
)
rand_df.to_csv("ORB_affinity_completness.csv")
numb_clust_df = pd.DataFrame(
    numb_clust, columns=n_features.astype(str) + "_features", index=wlen_lab
)
numb_clust_df.to_csv("ORB_affinity_clusters.csv")
numb_clust = numb_clust_df.to_numpy()
print("Rand max : ", np.max(a_rand))
print("N clust for rand max : ", numb_clust[np.where(a_rand == np.max(a_rand))])
# Load the best config:
rand_df = pd.read_csv(
    "/home/tmc/Documents/LPO_Prestation/LPO_Env_Dev/ORB_affinity_completness.csv",
    index_col=0,
)
a_rand = rand_df.to_numpy()
numb_clust_df = pd.read_csv("ORB_affinity_clusters.csv", index_col=0)
numb_clust = numb_clust_df.to_numpy()

plt.imshow(a_rand, cmap=cc.cm.fire)
plt.xticks(ticks=np.arange(len(n_features)), labels=n_features)
plt.xlabel("Nombre de features")
plt.yticks(ticks=np.arange(len(wlen_lab)), labels=wlen_lab)
plt.ylabel("Taille de fenêtre")
# plt.suptitle("Indice de Rand")
# plt.colorbar(fraction=0.01, pad=0.04)
plt.show()

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

# Get parameters for the best clustering
param_best = np.nonzero(a_rand == np.max(a_rand))
clusters_list = []
best_detector = cv2.ORB_create(edgeThreshold=1, nlevels=7)
best_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Several places have the best rand index, test them
for k in range(len(param_best[0])):
    wlen_best = wlen_test[param_best[0][k]]
    wlen_env_best = wlen_env_test[param_best[0][k]]
    wlen_best = 0.0486
    wlen_env_best = 0.121728
    n_features_best = n_features[param_best[1][k]]
    n_features_best = np.arange(50, 56)
    # n_features_best = 39
    best_specs = [
        spec(signaux[i], wlen_best, ovlp, wlen_env_best, ovlp_env, freq_ech, f_filt)
        for i in range(len(signaux))
    ]
    # Apply Affinity propagation + ORB with best parameters
    best_spec_8bits = [transfo_8bits(s) for s in best_specs]
    n_specs = len(best_spec_8bits)
    kp_best = [best_detector.detectAndCompute(l, None) for l in best_spec_8bits]
    dist_images = [
        distance_matches(best_matcher, kp_best[i], kp_best[j], n_features_best[k])
        for i in range(n_specs)
        for j in range(n_specs)
    ]
    dist_images = np.array(dist_images).reshape((n_specs, n_specs))
    # best_spec_8bits = [transfo_8bits(s) for s in spectrogrammes[param_best[0][k]]]
    # n_specs = len(best_spec_8bits)
    # dist_images = np.zeros((n_specs, n_specs))
    # for i in range(n_specs):
    #     for j in range(n_specs):
    #         img_i = best_spec_8bits[i]
    #         img_j = best_spec_8bits[j]
    #         shift, _, _ = phase_cross_correlation(img_i, img_j)
    #         shift = int(shift[1])
    #         if shift > 0:
    #             img_j = img_j[:, shift:]
    #         else:
    #             img_i = img_i[:, abs(shift) :]
    #         kp_best = [best_detector.detectAndCompute(l, None) for l in [img_i, img_j]]
    #         if len(kp_best[0][0]) > 0 and len(kp_best[1][0]) > 0:
    #             dist_images[i, j] = distance_matches(
    #                 best_matcher, kp_best[0], kp_best[1], n_features_best
    #             )
    #         else:
    #             dist = 1e10
    clustering_tech = AffinityPropagation()
    clusters = clustering_tech.fit_predict(dist_images)
    clusters_list.append(clusters)
    print(m.completeness_score(indiv, clusters))
    print(m.rand_score(indiv, clusters))

for c in clusters_list:
    print(c == clusters_list[0])

for c in clusters_list:
    print(c)

for c in np.unique(clusters_list[-1]):
    print(noms[clusters_list[-1] == c])

# => best features: dernier élément
wlen_best = 0.0486
wlen_best = 281 / freq_ech
wlen_env_best = 0.121728
wlen_env_best = 706 / freq_ech
n_features_best = 53
best_specs = [
    spec(signaux[k], wlen_best, ovlp, wlen_env_best, ovlp_env, freq_ech, f_filt)
    for k in range(len(signaux))
]
best_spec_8bits = [transfo_8bits(s) for s in best_specs]
n_specs = len(best_spec_8bits)
best_detector = cv2.ORB_create(edgeThreshold=1, nlevels=7)
best_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
kp_best = [best_detector.detectAndCompute(l, None) for l in best_spec_8bits]
dist_images = [
    distance_matches(best_matcher, kp_best[i], kp_best[j], n_features_best)
    for i in range(n_specs)
    for j in range(n_specs)
]
dist_images = np.array(dist_images).reshape((n_specs, n_specs))
clustering_tech = AffinityPropagation()
clusters = clustering_tech.fit_predict(dist_images)
print(m.completeness_score(indiv, clusters))
print(m.rand_score(indiv, clusters))

# Associate images with keypoints
for k in range(n_specs):
    img = cv2.drawKeypoints(
        best_spec_8bits[k],
        kp_best[k][0],
        None,
        # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite("Spectro_keypoints/" + noms[k].split(".wav")[0] + ".jpg", img)

# Get filenames of each cluster
for c in np.unique(clusters):
    print(noms[clusters == c])

np.column_stack((noms, clusters))

# Mean distance between males
# Distance between each male
np.mean(dist_images)
mean_dist = []
for i in uniq_ind:
    ind_i = np.nonzero(indiv == i)[0]
    for j in uniq_ind:
        ind_j = np.nonzero(indiv == j)[0]
        mm = dist_images[ind_i, :][:, ind_j]
        mean_dist.append(np.mean(mm))

mean_dist = np.array(mean_dist).reshape(len(uniq_ind), len(uniq_ind))
plt.imshow(mean_dist)
for (j, i), label in np.ndenumerate(mean_dist):
    plt.text(i, j, int(label), ha="center", va="center")

plt.xticks(ticks=np.arange(len(uniq_ind)), labels=uniq_ind)
plt.xlabel("Nom des mâles")
plt.yticks(ticks=np.arange(len(uniq_ind)), labels=uniq_ind)
plt.ylabel("Nom des mâles")
plt.suptitle("Distance moyenne inter-mâles calculé sur les features")
plt.show()

# Confusion matrix
uniq_clust = np.unique(clusters)
conf_mat = np.zeros((len(uniq_ind), len(uniq_clust)))
for i_u, u in enumerate(uniq_ind):
    conf_mat[i_u] = [
        np.sum(np.array([indiv == u]) * np.array([clusters == c])) for c in uniq_clust
    ]

plt.imshow(conf_mat, origin="lower")
for (j, i), label in np.ndenumerate(conf_mat):
    plt.text(i, j, int(label), ha="center", va="center")

plt.xticks(ticks=np.arange(len(uniq_clust)), labels=uniq_clust)
plt.xlabel("Numéro des clusters")
plt.yticks(ticks=np.arange(len(uniq_ind)), labels=uniq_ind)
plt.ylabel("Nom des mâles")
plt.title("Répartition des mâles au sein des clusters")
plt.show()
