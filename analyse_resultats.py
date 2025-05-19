# Libraries
from datetime import datetime
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
    date_uniq = np.unique(df_cc.Date)
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
plt.plot(
    daily_cl[:, 0], daily_cl[:, 1], "b", linestyle="-", marker="o", label="Chants longs"
)
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

for c in np.unique(df_cc.Cluster):
    plt.plot(daily_cc[:, 0], presence_cc[c, :], label="Cluster n°" + str(c))
    plt.legend()
    plt.show()

# Number of days where each cluster is present
# and number of vocs per cluster
# and spread of the presence of each cluster
nb_days_presence_cc = np.count_nonzero(presence_cc, axis=1)
nb_vocs_cc = np.sum(presence_cc, axis=1)
dt_clust_cc = []
# for k in range(presence_cc.shape[0]):
#     presence = presence_cc[k, :].nonzero()[0]
#     dt = presence[-1] - presence[0]
#     dt_clust_cc.append(dt)
for k in range(presence_cc.shape[0]):
    presence = presence_cc[k, :].nonzero()[0]
    if len(presence) > 1:
        mean_dt = np.mean(np.diff(presence))
    else:
        mean_dt = presence_cc.shape[0]
    dt_clust_cc.append(mean_dt)
dt_clust_cc = np.array(dt_clust_cc)
# presence_index_cc = (
#     nb_days_presence_cc * nb_vocs_cc / (np.sum(nb_vocs_cc) * dt_clust_cc)
# )
presence_index_cc = (
    nb_days_presence_cc * nb_vocs_cc / (np.sum(nb_vocs_cc) * presence_cc.shape[1])
)

df_presence_cc = pd.DataFrame(
    np.column_stack(
        (
            uniq_cluster_cc,
            nb_days_presence_cc,
            nb_vocs_cc,
            dt_clust_cc,
            presence_index_cc,
        )
    ),
    columns=["Cluster", "Nb_days", "Nb_vocs", "Mean_dt", "PI"],
)
nb_days_presence_cl = np.count_nonzero(presence_cl, axis=1)
nb_vocs_cl = np.sum(presence_cl, axis=1)
dt_clust_cl = []
# for k in range(presence_cl.shape[0]):
#     presence = presence_cl[k, :].nonzero()[0]
#     dt = presence[-1] - presence[0]
#     dt_clust_cl.append(dt)
for k in range(presence_cl.shape[0]):
    presence = presence_cl[k, :].nonzero()[0]
    if len(presence) > 1:
        mean_dt = np.mean(np.diff(presence))
    else:
        mean_dt = presence_cl.shape[0]
    dt_clust_cl.append(mean_dt)
dt_clust_cl = np.array(dt_clust_cl)
presence_index_cl = (
    nb_days_presence_cl * nb_vocs_cl / (np.sum(nb_vocs_cl) * presence_cl.shape[1])
)
df_presence_cl = pd.DataFrame(
    np.column_stack(
        (
            uniq_cluster_cl,
            nb_days_presence_cl,
            nb_vocs_cl,
            dt_clust_cl,
            presence_index_cl,
        )
    ),
    columns=["Cluster", "Nb_days", "Nb_vocs", "Mean_dt", "PI"],
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
    np.argsort(dt_clust_cc).astype(str),
    np.sort(dt_clust_cc),
    color="g",
    label="Chants courts",
)
plt.legend()
plt.ylabel("Nombre de jours moyens entre deux présences")
plt.xlabel("Cluster")
plt.subplot(212)
plt.bar(
    np.argsort(dt_clust_cl).astype(str),
    np.sort(dt_clust_cl),
    color="b",
    label="Chants longs",
)
plt.legend()
plt.ylabel("Nombre de jours moyens entre deux présences")
plt.show()

plt.subplot(211)
plt.bar(
    np.argsort(presence_index_cc).astype(str),
    np.sort(presence_index_cc),
    color="g",
    label="Chants courts",
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
plt.legend()
plt.ylabel("Indice de Présence")
plt.xlabel("Cluster")
plt.show()
