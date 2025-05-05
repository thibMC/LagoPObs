# Image matching using features extractions
import numpy as np
import os
import cv2  # opencv-python
from sklearn.mixture import GaussianMixture
from sklearn.cluster import (
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    BisectingKMeans,
    MeanShift,
    AffinityPropagation,
)
from sklearn.metrics import silhouette_score


def cluster_spectro(
    list_spectros,
    n_matches=10,
    detector_methode="ORB custom",
    clustering="Affinity Propagation",
):
    """
    Cluster the combined spectrograms.

    Parameters
    ----------
    list_spectros: list of arrays, each array being the combinaison of both STFTs on each filtered sound.
    n_matches: int, number of the closest matches to keep when calculating the distance between two arrays.

    Returns
    -------
    cluster_labels: 1D array, of same length as list_spectros, with the cluter label of each array.
    keypoints_descriptors: length-n list, each element being a tupple, with the keypoints and descriptors of each array of list_spectros.
    """
    liste_8bits = [transfo_8bits(s) for s in liste_spectros]
    detector, matcher = feature_detector_matcher(name=detector_methode)
    keypoints_descriptors = [detector.detectAndCompute(l, None) for l in liste_8bits]
    n_specs = len(liste_8bits)
    dist_images = [
        distance_matches(
            matcher, keypoints_descriptors[i][1], keypoints_descriptors[j][1], n_matches
        )
        for i in range(n_specs)
        for j in range(n_specs)
    ]
    dist_images = np.array(dist_images).reshape((n_specs, n_specs))
    cluster_labels = clustering_matches(dist_images, clustering_name=clustering)
    return cluster_labels, keypoints_descriptors


def save_spectros_keypoints(
    list_spectros, keypoints_descriptors, names, dir, n_keypoints
):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    n_specs = length(list_spectros)
    for k in range(n_specs):
        img = cv2.drawKeypoints(
            list_spectros[k],
            keypoints_descriptors[k][0][:n_keypoints],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(dir + "/" + names[k].split(".wav")[0] + ".jpg", img)


def transfo_8bits(arr):
    """
    Convert a 2D array in 8-bit unsigned integers, for compatibility with OpenCV.

    Parameters
    ----------
    arr: 2D array, array to convert.

    Returns
    -------
    arr_8bits: 2D array, 8-bit unsigned array.
    """
    arr[arr < 0] = 0
    arr_8bits = 255 * abs(arr / np.max(arr))
    return arr_8bits.astype(np.uint8)


def feature_detector_matcher(name="ORB custom"):
    """
    Return a keypoint detector and descriptor extractor based on its name and the matcher, used to match descriptors between images.

    Parameters
    ---------
    name: str, name of the feature detector, choose from: "ORB", "ORB custom", "AKAZE", "KAZE", "SIFT".
    "ORB custom" is an ORB instance created with parameters tunned to separate ptarmigans.

    Returns
    -------
    detector: class instance of the feature detector.
    matcher: class instance of the corresponding matcher of featres between images.
    """
    if name == "SIFT":
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(crossCheck=True)
    elif name == "ORB":
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif name == "ORB custom":
        detector = cv2.ORB_create(edgeThreshold=1, nlevels=7)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif name == "AKAZE":
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif name == "KAZE":
        detector = cv2.KAZE_create()
        matcher = cv2.BFMatcher(crossCheck=True)
    return detector, matcher


def distance_matches(matcher, des1, des2, n_closest):
    """
    Get a matching distance between two images based on their descriptors. The shorter the distance, the more similar the images are.

    Parameters
    ----------
    matcher: the matcher that will be used to match the descriptors. A Brute-force descriptor matcher created using cv2.BFMatcher is used here.
    des1: a list, containing the descriptors of the first image resulting from the application of a feature extractor.
    des2: a list, containing the descriptors of the second image resulting from the application of a feature extractor.
    n_closest: int, number of matches with the shortest distance to consider.

    Returns
    -------
    dist: float, the distance between the two images.

    """
    # Match the descriptors
    matches = matcher.match(des1, des2)
    # Sort matches by distances
    matches = sorted(matches, key=lambda x: x.distance)
    # Get mean distance of the n_clostest matches
    if matches and len(matches) <= n_closest:
        dist = np.mean([m.distance for m in matches])
    elif matches and len(matches) > n_closest:
        dist = np.mean([m.distance for m in matches[:n_closest]])
    else:
        dist = 1e10
    return dist


def clustering_matches(dist_images, clustering_name="Affinity Propagation"):
    """
    Cluster images based on their matching distances. As the matching distance matrix is not symetrical, i.e. matching distance (im1,im2) != matching distance (im2,im1), the matching distance matrix will be considered as a normal data array and the euclidean distance will be performed by the clustering algorithm. For clustering algorithms where the number of clusters needs to be selected, the silhouette score is used.

    Parameters
    ----------
    dist_images: 2D array, a n by n array, with n the number of images. d_match[i,j] contains the matching distance of image i and image j.
    clustering_name: str, name of the clustering. Choose from: "Affinity Propagation", "Agglomerative", "Bisecting K-Means", "Gaussian Mixture Model", "HDBSCAN", "K-Means", "Mean Shift".

    Returns
    -------
    clust_label: a n-length array, with the cluter label for each image.
    """
    # Dict with clustering techniques
    dict_clust = {
        "Affinity Propagation": AffinityPropagation(),
        "Agglomerative": AgglomerativeClustering(),
        "Bisecting K-Means": BisectingKMeans(),
        "Gaussian Mixture Model": GaussianMixture(),
        "HDBSCAN": HDBSCAN(min_cluster_size=2),
        "K-Means": KMeans(),
        "Mean Shift": MeanShift(n_jobs=-1),
    }
    clust_tech = dict_clust[clustering_name]
    if clustering_name in [
        "Agglomerative",
        "Gaussian Mixture Model",
        "K-Means",
        "Bisecting K-Means",
    ]:
        sil = []
        n_image = dist_images.shape[0]
        n_clust = np.arange(2, n_image)
        for c in n_clust:
            if clustering_name == "Gaussian Mixture Model":
                clust_tech.set_params(n_components=c)
            else:
                clust_tech.set_params(n_clusters=c)
            cluster = clust_tech.fit_predict(dist_images)
            sil.append(silhouette_score(dist_images, cluster))
        if clustering_name == "Gaussian Mixture Model":
            clust_tech.set_params(n_components=n_clust[np.argmax(sil)])
        else:
            clust_tech.set_params(n_clusters=n_clust[np.argmax(sil)])
        clustering_final = clust_tech.fit_predict(dist_images)
    else:
        clustering_final = clust_tech.fit_predict(dist_images)
    return clustering_final
