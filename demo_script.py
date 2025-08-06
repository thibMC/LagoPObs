### Demo script
# Import
import numpy as np
import pandas as pd

try:
    from tools import utils, filtering, spectro, image_matching, pop_estimation
except:
    from LagoPObs.tools import utils, filtering, spectro, image_matching, pop_estimation

# Variables: same as the default in the GUI
input_dir = ""  # directory with sounds
output_dir = ""  # directory where the results will be saved
wlt_filt = "Yes"  # Wavelet filtering?
f_filt = [950, 2800]  # frequency bandwidth
wlen = 281  # Window length
ovlp = 75  # overlap spectro
wlen_env = 706
ovlp_env = 90  # overlap spectro enveloppe
n_matches = 53  # number of matches
detector_methode = "ORB custom"  # Feature extraction algorithm
clustering = "Affinity Propagation"  # Clustering algorithm
estim_pop = "Yes"

# Perform analysis
# filter the WAV filenames in the input directory
list_wavs = utils.filter_wavs(input_dir)
# Import the sounds, transform them in float64 arrays
# and normalize them by their RMS
list_arr, sf = utils.import_wavs(list_wavs, input_dir, f_filt[1])
# Bandpass filtering
list_arr_filt = [filtering.butterfilter(a, sf, f_filt) for a in list_arr]
# Wavelet filtering if wlt_fit == "Yes"
if wlt_filt == "Yes":
    list_arr_filt = [filtering.wlt_denoise(a) for a in list_arr_filt]
arr_filt = utils.pad_signals(list_arr_filt)
# Drawing the spectrograms
spectros = [
    spectro.draw_specs(
        a,
        int(wlen),
        int(ovlp),
        int(wlen_env),
        int(ovlp_env),
        sf,
        f_filt,
    )
    for a in arr_filt
]
# clustering the spectrograms
clusters, kp_desc = image_matching.cluster_spectro(
    spectros, n_matches, detector_methode, clustering
)
# Save the spectrograms with the keypoints in it
image_matching.save_spectros_keypoints(spectros, kp_desc, list_wavs, output_dir)
# Save the clustering results
# It is really important to create the DataFrame with a dict here,
# otherwise, it can impede the cluster order and thus the results.
df_res = pd.DataFrame({"File": list_wavs, "Cluster": clusters})
df_res.to_csv(output_dir + "/clustering_results.csv", index=False)

# Population estimation
if estim_pop == "Yes":
    try:
        df_res_with_date = pop_estimation.add_date_to_df(df_res)
    except ValueError:
        print(
            "Wrong filename format! The filenames must be split in different part, separated by undescores, with the date in the second position and with the following format: yearmonthday. For example: 'xxxxx_20230619_xxxxxx.wav' means that the following file was recorded in June 13, 2023."
        )
    else:
        n_clusts_per_day = pop_estimation.daily_vocalize_clusters(df_res_with_date)
        n_clusts_per_day.to_csv(
            output_dir + "/Number_of_clusters_per_day.csv", index=False
        )
        pres = pop_estimation.presence_clusters(df_res_with_date)
        pres.to_csv(output_dir + "/number_of_sounds_per_cluster_per_date.csv")
        pi_arr = pop_estimation.presence_index_arr(df_res_with_date)
        pi_df = pd.DataFrame(
            pi_arr,
            columns=[
                "Cluster",
                "Days_of_presence",
                "Number_of_sounds",
                "Presence_index",
            ],
        )
        sorted_pi_df = pi_df.sort_values(by="Presence_index", ascending=False)
        sorted_pi_df.to_csv(output_dir + "/presence_index.csv", index=False)
        # Estimation of resident individuals according to Presence Index
        n_indiv_pi = np.count_nonzero(pi_df.Presence_index >= 0.01)
        # Estimation of the whole population using Population Information Criterion
        pi_pop = pop_estimation.population_presence_index(pi_arr, pres)
        n_indiv_pic, df_pic = pop_estimation.estimate_number_of_individuals(pi_pop)
        df_pic.to_csv(output_dir + "/PPI_PIC.csv", index=False)
        res_print = f"Estimated number of individuals using PI: {n_indiv_pi}\nEstimated number of individuals using PIC: {n_indiv_pic}"
        print(res_print)
        print(res_print, open(output_dir + "/results.txt", "w"))
