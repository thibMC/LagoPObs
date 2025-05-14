# Libraries
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
presence_cl = presence_clusters(df_cc)

for c in np.unique(df_cc.Cluster):
    plt.plot(daily_cc[:, 0], presence_cc[c, :], label="Cluster n°" + str(c))
plt.legend()
plt.show()
