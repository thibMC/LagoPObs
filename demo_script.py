### Demo script
# Import
from LagoPObs.tools import utils, filtering, spectro, image_matching

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

# Perform analysis
# filter the WAV filenames in the input directory
list_wavs = utils.filter_wavs(wd)
# Import the sounds, transform them in float64 arrays
# and normalize them by their RMS
list_arr, sf = utils.import_wavs(list_wavs, wd, f_filt[1])
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
        int(wlen_best * freq_ech),
        int(ovlp * 100),
        int(wlen_env_best * freq_ech),
        int(ovlp_env * 100),
        freq_ech,
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
df_res = pd.DataFrame(
    np.column_stack((list_wavs, clusters)), columns=["File", "Cluster"]
)
df_res.to_csv(param_values[1] + "/clustering_results.csv", index=False)
