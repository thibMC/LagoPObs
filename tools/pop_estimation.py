# Librairies
from datetime import datetime, date
import numpy as np
import pandas as pd


def estimate_number_of_individuals(pi_pop):
    """
    Some individual may be split into several clusters because their sounds are modified through time or that some individuals are only present really briefly or even because of noises. As such, the total number of clusters found during the analysis might be an overestimation of the population.

    This function tries to estimate the number of individuals in the recorded population based on an information criterion inspired by the AIC (Akaike Information Criterion, [1]) and a generalization  of the Presence Index (see population_presence_index) to a population.

    We can assimilate the PPI to a likelihood that the n clusters in the population represents the data and, ultimately, the real number of individuals in the sampled population.

    We can use the formula of the AIC [1-2] as inspiration to calculate a cost function, the Population Information Criterion (PIC), that strikes a balance between the explanation of the population and the overstimation of the number of individuals in the population. The PIC of a population of n clusters is equal to :

    PIC(n) = 2x(n/N) - 2xlog(1+PPI(n))

    with n the number of clusters in the population, N the total number of clusters found, log the natural logarithm and PPI(n) the Population Presence Index for the n clusters.

    The estimated number of individuals according to the clustering is thus the number that minimizes the PIC.

    [1] Akaike, H. (1998). Information theory and an extension of the maximum likelihood principle. In Selected papers of hirotugu akaike (pp. 199-213). New York, NY: Springer New York.

    [2] https://en.wikipedia.org/wiki/Akaike_information_criterion


    Parameters
    ----------
    pi_pop: a 1D array, result from population_presence_index, containing the Population Presence Index for a number of cluster ranging from 1 to the total number of cluster.

    Returns
    -------
    n_indiv: int, the estimated number of individuals based on the PIC.
    df_pic: a pandas DataFrame containing 3 columns: the number of clusters in "Number_of_clusters", the corresponding PPI in "Population_Presence_Index" and
    """
    pic = np.array(
        [
            2 * (k + 1) / len(pi_pop) - 2 * np.log(1 + pi_pop[k])
            for k in range(len(pi_pop))
        ]
    )
    df_pic = pd.DataFrame(
        {
            "Number_of_clusters": np.arange(1, len(pi_pop) + 1),
            "Population_Presence_Index": pi_pop,
            "Population_Information_Criterion": pic,
        }
    )
    # Number of individuals
    n_indiv = 1 + np.argmin(pic)
    return (int(n_indiv), df_pic)


def population_presence_index(pi_arr, presence):
    """
    We can generalize the Presence Index (see presence_index_arr) to a population as the Population Presence Index (PPI) of n clusters as:

    PPI(n) = (Sum of the number of sounds of clusters in n / Total number of sounds) x (Number of days with at least the presence of one cluster in n / Total number of days)

    PPI is calculated for an increasing number of clusters. The order of inclusion of a cluster in the simulated population is made according to their presence in the area, i.e. in decreasing order of PI. The more clusters are added, the more the PPI will increase and be closer to 1.

    Parameters
    ----------
    pi_arr: a numpy array of shape (number of clusters, 4), the result of the function presence_index_arr.
    presence: a pandas DataFrame of shape (number of clusters, number of dates), the result of the function presence_clusters.

    Returns
    -------
    pi_pop: a 1D array, containing the Population Presence Index for a number of cluster ranging from 1 to the total number of cluster

    """
    # Get the indices of the decreasing order of PI
    clusters_ordered = np.argsort(pi_arr[:, 3])[::-1]
    # Create the list with the clusters in the growing population
    pops = [clusters_ordered[:k] for k in range(1, len(clusters_ordered) + 1)]
    # Number of sounds covered by the population
    n_sounds_pop = np.array([np.sum(pi_arr[p, 2]) for p in pops])
    # Number of days where we have at least a cluster of the population
    sounds_pop_per_day = [np.sum(presence[p, :], axis=0) for p in pops]
    n_days_pop = np.array([len(np.nonzero(s)[0]) for s in sounds_pop_per_day])
    # Population presence index
    pi_pop = (n_sounds_pop / np.sum(presence)) * (n_days_pop / presence.shape[1])
    return pi_pop


def get_date_from_filename(name):
    """
    Get the date from a filename.

    Parameters
    ----------
    name: str, the filename, must be split in different part, separated by undescores, with the date in the second position and with the following format: yearmonthday.
    For example:  "xxxxx_20230619_xxxxxx.wav" means that the following file was recorded in June 13, 2023.

    Returns
    -------
    date: a date in the datetime.date format (see datetime for further details)

    """
    d = name.split("_")[1]
    date = datetime.strptime(d, "%Y%m%d")
    date = datetime.date(date)
    return date


def daily_vocalize_clusters(df):
    """
    Parameters
    ----------
    df: a pandas DataFrame with at least 2 columns. One nammed "Date", composed of dates extracted from filenames, using get_date_from_filename. The other, nammed "Cluster", should contains the results of image_matching.cluster_spectro.

    Returns
    -------
    n_clust_voc_per_day: a pandas DataFrame of shape (number of different dates, 3). The first column, "Date", contains the different days of the dataset. The second, "Number_Clusters" contains the number of clusters present per day. The third, "Number_Sounds" represents the number of sounds recorded for each date.

    """
    date_uniq = np.unique(df_cc.Date)
    nb_clusts = []
    nb_sounds = []
    for d in date_uniq:
        clust = df.Cluster[df.Date == d]
        nb_clusts.append(len(np.unique(clust)))
        nb_sounds.append(len(clust))
    n_clusts_sounds_per_day = np.column_stack(
        (date_uniq, np.array(nb_clusts), np.array(nb_vocs))
    )
    n_clusts_sounds_per_day = pd.DataFrame(
        n_clusts_sounds_per_day, columns=["Date", "Number_Clusters", "Number_Sounds"]
    )
    return n_clusts_sounds_per_day


def presence_index_arr(df):
    """
    Calculate the number of days of presence, number of sounds and the presence index of each cluster.
    The Presence Index (PI), defined in [1], is a simple way of measuring if a particular cluster is regularly present in the area. For a cluster k, it is calculated using the formula:

    PI(k) =  (Number of days of presence of k / Total number of days) x (Number of sounds attributed to k / Total number of sounds)

    According to [1], a PI >= 0.01 allows to identify clusters that could represent rock ptarmigan males that would be residential males in the area and that are more likely to have a female.

    [1] Marin-Cudraz, T. (2019). Bioacoustics potential as a tool for counting diffcult-to-access species: The case of the rock ptarmigan (Lagopus muta). PhD thesis. https://theses.hal.science/tel-02894049/file/These-Thibaut-Marin-Cudraz-2019.pdf

    Parameters
    ----------
    df: a pandas DataFrame with at least 2 columns. One nammed "Date", composed of dates extracted from filenames, using get_date_from_filename. The other, nammed "Cluster", should contain the results of image_matching.cluster_spectro.

    Returns
    -------
    pi_arr: a numpy array of shape (number of clusters, 4).
    The first column, contains the id of the cluster.
    The second column contains the number of days of presence.
    The third, the total number of sounds attributed to a particular cluster.
    The forth, the presence index of each cluster.

    """
    uniq_cluster = np.unique(df.Cluster)
    presence = presence_clusters(df)
    # Number of days where each cluster is present
    nb_days = np.count_nonzero(presence, axis=1)
    # Number of sounds of each cluster
    nb_sounds = np.sum(presence_cc, axis=1)
    # Presence Index
    presence_index = nb_days * nb_sounds / (np.sum(nb_sounds) * presence.shape[1])
    # Array
    pi_arr = np.column_stack(
        (
            uniq_cluster,
            nb_days,
            nb_sounds,
            presence_index,
        )
    )
    return pi_arr


def presence_clusters(df):
    """
    Presence of each cluster per day.

    Parameters
    ----------
    df: a pandas DataFrame with at least 2 columns. One nammed "Date", composed of dates extracted from filenames, using get_date_from_filename. The other, nammed "Cluster", should contains the results of image_matching.cluster_spectro.

    Returns
    -------
    presence: a pandas DataFrame of shape (number of clusters, number of dates), with each line representing the number of sounds assigned to a particular cluster per day.

    """
    clusters = np.unique(df.Cluster)
    nb_clusters = len(clusters)
    days = np.unique(df.Date)
    n_days = len(days)
    presence = np.zeros((nb_clusters, n_days))
    for c in clusters:
        for d in days:
            n = len(df.Cluster[np.logical_and(df.Cluster == c, df.Date == d)])
            presence[clusters == c, days == d] = n
    cluster_index = ["Cluster_" + str(int(c)) for c in clusters]
    presence = pd.DataFrame(presence, columns=days, index=cluster_index)
    return presence
