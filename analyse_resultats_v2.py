# Libraries
from datetime import datetime, date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

# Import the results CSV
df_cc = pd.read_csv(
    "/home/tmc/Documents/LPO_Prestation/Resultats/Clustering/CC/clustering_results.csv"
)
df_cl = pd.read_csv(
    "/home/tmc/Documents/LPO_Prestation/Resultats/Clustering/CL/clustering_results.csv"
)

# Get additional infos for filename:
# - Recorder
df_cc["Recorder"] = df_cc.File.apply(lambda x: x.split("_")[0])
df_cl["Recorder"] = df_cl.File.apply(lambda x: x.split("_")[0])


# - Date
def get_date_from_filename(name):
    d = name.split("_")[1]
    date = datetime.strptime(d, "%Y%m%d")
    date = datetime.date(date)
    return date


df_cc["Date"] = df_cc.File.apply(get_date_from_filename)
df_cl["Date"] = df_cl.File.apply(get_date_from_filename)


# - Start time
def get_time_from_filename(name):
    d = name.split("_")[1]
    t = name.split("_")[2]
    t = t.split("-")[0]
    date = datetime.strptime(d + "_" + t, "%Y%m%d_%H%M%S")
    return date


df_cc["Start_recording_time"] = df_cc.File.apply(get_time_from_filename)
df_cl["Start_recording_time"] = df_cl.File.apply(get_time_from_filename)


# Get the number of different clusters per date
def daily_vocalize_clusters(df):
    date_uniq = np.unique(df.Date)
    nb_clust = []
    nb_voc = []
    for d in date_uniq:
        clust = df.Cluster[df.Date == d]
        nb_clust.append(len(np.unique(clust)))
        nb_voc.append(len(clust))
    diff_clust = np.column_stack((date_uniq, np.array(nb_clust), np.array(nb_voc)))
    return diff_clust


# Daily number of clusters and number of vocalisations
daily_cc = daily_vocalize_clusters(df_cc)
daily_cl = daily_vocalize_clusters(df_cl)

# count days
count_days = [date(2023, 5, 25), date(2023, 6, 2), date(2023, 6, 17)]

plt.plot(
    daily_cc[:, 0],
    daily_cc[:, 2],
    "g",
    linestyle="-",
    marker="o",
    label="Chants courts",
)
plt.plot(
    daily_cl[:, 0], daily_cl[:, 2], "b", linestyle="-", marker="o", label="Chants longs"
)
plt.xticks(daily_cc[:, 0], daily_cc[:, 0], rotation="vertical")
plt.yticks(np.arange(26), labels=np.arange(26))
plt.xlabel("Date")
plt.ylabel("Nombre de vocalises utilisées pour le clustering")
plt.legend()
plt.show()


plt.plot(
    daily_cc[:, 0],
    daily_cc[:, 1],
    "g",
    linestyle="-",
    marker="o",
    label="Chants courts",
)
# Add the count dates in cc plot
for c_d in count_days:
    plt.plot(c_d, daily_cc[daily_cc[:, 0] == c_d, 1], "or", markersize=7)
plt.plot(
    daily_cl[:, 0], daily_cl[:, 1], "b", linestyle="-", marker="o", label="Chants longs"
)
# Add the count dates in cl plot
for c_d in count_days:
    plt.plot(c_d, daily_cl[daily_cl[:, 0] == c_d, 1], "or", markersize=7)
plt.xticks(daily_cc[:, 0], daily_cc[:, 0], rotation="vertical")
plt.yticks(np.arange(13))
plt.xlabel("Date")
plt.ylabel("Nombre de clusters présents")
plt.legend()
plt.show()


# Presence of each cluster per day
def presence_clusters(df):
    clusters = np.unique(df.Cluster)
    nb_clusters = len(clusters)
    days = np.unique(df.Date)
    n_days = len(days)
    presence = np.zeros((nb_clusters, n_days))
    for c in clusters:
        for d in days:
            n = len(df.Cluster[np.logical_and(df.Cluster == c, df.Date == d)])
            presence[clusters == c, days == d] = n
    return presence


presence_cc = presence_clusters(df_cc)
presence_cl = presence_clusters(df_cl)
uniq_cluster_cc = np.unique(df_cc.Cluster)
uniq_cluster_cl = np.unique(df_cl.Cluster)

# Number of days where each cluster is present
# and number of vocs per cluster
nb_days_presence_cc = np.count_nonzero(presence_cc, axis=1)
nb_vocs_cc = np.sum(presence_cc, axis=1)
presence_index_cc = (
    nb_days_presence_cc * nb_vocs_cc / (np.sum(nb_vocs_cc) * presence_cc.shape[1])
)

df_presence_cc = pd.DataFrame(
    np.column_stack(
        (
            uniq_cluster_cc,
            nb_days_presence_cc,
            nb_vocs_cc,
            presence_index_cc,
        )
    ),
    columns=["Cluster", "Nb_days", "Nb_vocs", "Presence_Index"],
)
nb_days_presence_cl = np.count_nonzero(presence_cl, axis=1)
nb_vocs_cl = np.sum(presence_cl, axis=1)
presence_index_cl = (
    nb_days_presence_cl * nb_vocs_cl / (np.sum(nb_vocs_cl) * presence_cl.shape[1])
)
df_presence_cl = pd.DataFrame(
    np.column_stack(
        (
            uniq_cluster_cl,
            nb_days_presence_cl,
            nb_vocs_cl,
            presence_index_cl,
        )
    ),
    columns=["Cluster", "Nb_days", "Nb_vocs", "PI"],
)

# Plots
plt.subplot(211)
plt.bar(
    np.argsort(nb_days_presence_cc).astype(str),
    np.sort(nb_days_presence_cc),
    color="g",
    label="Chants courts",
)
plt.legend()
plt.ylabel("Présence (jours)")
plt.xlabel("Cluster")
plt.subplot(212)
plt.bar(
    np.argsort(nb_days_presence_cl).astype(str),
    np.sort(nb_days_presence_cl),
    color="b",
    label="Chants longs",
)
plt.legend()
plt.ylabel("Présence (jours)")
plt.xlabel("Cluster")
plt.show()

plt.subplot(211)
plt.bar(
    np.argsort(nb_vocs_cc).astype(str),
    np.sort(nb_vocs_cc),
    color="g",
    label="Chants courts",
)
plt.legend()
plt.ylabel("Nombre de vocalises")
plt.xlabel("Cluster")
plt.subplot(212)
plt.bar(
    np.argsort(nb_vocs_cl).astype(str),
    np.sort(nb_vocs_cl),
    color="b",
    label="Chants longs",
)
plt.legend()
plt.ylabel("Nombre de vocalises")
plt.xlabel("Cluster")
plt.show()

plt.subplot(211)
plt.bar(
    np.argsort(presence_index_cc).astype(str),
    np.sort(presence_index_cc),
    color="g",
    label="Chants courts",
)
plt.plot(
    np.argsort(presence_index_cc).astype(str), np.repeat(0.01, len(presence_cc)), "--r"
)
plt.legend()
plt.ylabel("Indice de Présence")
plt.xlabel("Cluster")
plt.subplot(212)
plt.bar(
    np.argsort(presence_index_cl).astype(str),
    np.sort(presence_index_cl),
    color="b",
    label="Chants longs",
)
plt.plot(
    np.argsort(presence_index_cl).astype(str), np.repeat(0.01, len(presence_cl)), "--r"
)
plt.legend()
plt.ylabel("Indice de Présence")
plt.xlabel("Cluster")
plt.show()

# plt.subplot(211)
# plt.plot(
#     np.sort(presence_index_cc)[1:] / np.sort(presence_index_cc)[:-1],
#     color="g",
#     label="Chants courts",
# )
# plt.legend()
# plt.ylabel("Indice de Présence")
# plt.xlabel("Cluster")
# plt.subplot(212)
# plt.plot(
#     np.sort(presence_index_cl)[1:] / np.sort(presence_index_cl)[:-1],
#     color="b",
#     label="Chants longs",
# )
# plt.legend()
# plt.ylabel("Indice de Présence")
# plt.xlabel("Cluster")
# plt.show()


# Number of resident VS sattelites based on cost of adding a male
# Sort the clusters by decreasing PI values
# As the clusters are ints ranging from 0 to the total number of clusters -1,
# they can be associated with the indices of the array
cc_clusters_pi_decrease = np.argsort(presence_index_cc)[::-1]
cl_clusters_pi_decrease = np.argsort(presence_index_cl)[::-1]


# Function to calculate the PI for a population of N clusters
def presence_index_pop(ordered_clusters, df, presence):
    # Create the list with the clusters in the growing population
    pops = [ordered_clusters[:k] for k in range(1, len(ordered_clusters) + 1)]
    # Number of vocalisation covered by the population
    n_vocs_pop = np.array([np.sum(df.Nb_vocs[p]) for p in pops])
    # Number of days where we have at least a cluster of the population
    vocs_pop_per_day = [np.sum(presence[p, :], axis=0) for p in pops]
    n_days_pop = np.array([len(np.nonzero(v)[0]) for v in vocs_pop_per_day])
    pi_pop = n_vocs_pop * n_days_pop / (np.sum(presence) * presence.shape[1])
    return pi_pop


pi_population_cc = presence_index_pop(
    cc_clusters_pi_decrease, df_presence_cc, presence_cc
)
pi_population_cl = presence_index_pop(
    cl_clusters_pi_decrease, df_presence_cl, presence_cl
)
plt.plot(pi_population_cc, "g", label="Chants courts")
plt.plot(pi_population_cl, "b", label="Chants longs")
plt.xlabel("Nombre de mâles dans la population")
plt.ylabel("Indice de présence de la population")
plt.legend()
plt.show()

# cost of adding a male into the population = Critère d'information de population
# => Population Information Criterion
pic_cc = np.array(
    [
        2 * (k + 1) / len(pi_population_cc) - 2 * np.log(1 + pi_population_cc[k])
        for k in range(len(pi_population_cc))
    ]
)

pic_cl = np.array(
    [
        2 * (k + 1) / len(pi_population_cl) - 2 * np.log(1 + pi_population_cl[k])
        for k in range(len(pi_population_cl))
    ]
)

plt.plot(np.arange(1, len(pic_cc) + 1), pic_cc, "g", label="Chants courts")
plt.plot(np.argmin(pic_cc) + 1, np.min(pic_cc), "or", markersize=7)
plt.plot(np.arange(1, len(pic_cl) + 1), pic_cl, "b", label="Chants longs")
plt.plot(np.argmin(pic_cl) + 1, np.min(pic_cl), "or", markersize=7)
plt.legend()
plt.xlabel("Nombre de clusters")
plt.ylabel("Critère d'information de population")
plt.show()
