# LagoPObs: Lagopède Population Observator

This code is © T. Marin-Cudraz, REEF PULSE S.A.S, LPO AuRA, 2025, and it is made available under the MIT license enclosed with the software.  As this license is not very restrictive, we invite you to use, share and modify the software and its codes.

Over and above the legal restrictions imposed by this license, if you use this software for an academic publication then please provide proper attribution. This can be to this code directly:

T. Marin-Cudraz, B. Drillat. LagoPObs: Lagopède Population Observator, a software to separate male ptarmigans indivuals from their sounds (2025). https://github.com/lpoaura/LagoPObs.

# Purpose of the software

The purpose of this software and its code is to separate and group animal sounds. It was initially made to estimate populations of rock ptarmigan (*Lagopus muta*) via long-term recordings, but can potentially be generalized to:
- Test the separation of a bioacoustics dataset into several types of sounds/songs/vocalizations.
- Be a first separation step in an automatic signal separation/recognition pipeline.
- Test the separation of animal populations via their sounds.
- Test the individual signature carried by signals.
- Estimate a number of individuals in a population.

This software has several advantages: it's relatively fast (around ten seconds for 100 files), it doesn't need a learning phase compared to machine learning, and it is presented as stand-alone executables (for Windows 10/11 and Linux), with a simple interface. A demonstration script is given for those wishing to play and modify the underlying code. However, a calibration phase is required to enhance the software performance, depending on the type of signal and the results expected.

# Technical description

Sound separation is based on the results of a STFT (Short-Time Fourier Transform) and an envelope STFT. Each STFT / envelope STFT  combination is compared with the others using computer vision algorithms: feature matching (in the exemples below, an ORB [1-2]). They search for keypoints in these spectrograms and extract descriptors of local spectrogram features around these keypoints. Two examples are given below, a vocalization of a rock ptarmigan (Fig.1) and a song of a hazel grouse (*Tetrastes bonasia*, Fig.2). In both cases, the images analyzed by feature matching consist of two sound spectrograms, cut at the frequencies of interest: at the top, the STFTs of the filtered and cleaned sound (see below for filtering) and at the bottom, the spectrogram of the envelope of the filtered and cleaned sound. Note that the origin of each STFT is at the top left, so the frequency axis is inverted compared with a conventional STFT display. The colored dots are the keypoints identified by the feature matching algorithm. We can see that the algorithm identify the characteristics of the image well, whether for species with highly rhythmic, pulse-based vocal production like the ptarmigan, or based on frequency modulation like the grouse.

|![example_rock_ptarmigan.png](Readme/example_rock_ptarmigan.png)|
|:--:|
|Figure 1: Image corresponding to a rock ptarmigan vocalization (*rock_ptarmigan_1.wav* from the *Examples* folder) in the top part. The same image with its keypoints as colored dots in the bottom part. Each image is divided in two parts. On top, the STFT of the filtered and resampled sound (new sampling frequency: 5800 Hz), for frequencies between 950 and 2800 Hz. STFT characteristics: Hamming window, size: 281, overlap: 75%. Below, STFT of the envelope, for frequencies between 0 and 160 Hz. STFT envelope characteristics: Hamming window, size 706, overlap: 90%. Sound source: Thibaut Marin-Cudraz, Frédéric Sèbe, [3-4].|

|![example_hazel_grouse.png](Readme/example_hazel_grouse.png)|
|:--:|
|Figure 2: Image corresponding to a hazel grouse song (*XC361691_Gelinotte_des_bois_1.wav* from the *Examples* folder) in the top part. The bottom part is the same image with its keypoints as colored dots. Each image is divided in two parts. On top, the STFT of the filtered and resampled sound (new sampling frequency: 32200 Hz), for frequencies between 6000 and 16000 Hz. STFT characteristics: Hamming window, size: 512, overlap: 75%. Below, STFT of the envelope, for frequencies between 0 and 160 Hz. STFT envelope characteristics: Hamming window, size 4096, overlap: 90%. Sound source: https://xeno-canto.org/361691, Benjamin Drillat.|


Once the keypoints and descriptors have been extracted, a matcher (in this case a Brute force matcher [5-6]) is used to match the descriptors of two images, identifying descriptors with similar characteristics (Fig.3). Based on the matches, an average distance is calculated between two images. The smaller the distance, the more similar the descriptors of the two images. A matrix composed of the distances calculated between each pair of images is then computed and injected into a clustering algorithm to separate the sounds into homogeneous groups. Some types of sounds, such as ptarmigan vocalizations, have subtle inter-individual differences. To increase inter-individual differences and facilitate their separation, the software lets the clustering algorithms preprocess the data and calculate the Euclidean distance between each pair of sounds based on the inter-image distance matrix. The clustering results are saved in a CSV file (*clustering_results.csv*) and the images with their keypoints (as shown in Figure 1 and 2) are exported and saved.

|![match_example.png](Readme/match_example.png)|
|:--:|
|Figure 3: Example of matching between two images corresponding to two different rock ptarmigan vocalizations (*rock_ptarmigan_1.wav* and *rock_ptarmigan_2.wav* from the *Examples* folder supplied with the software and generated with the *draw_matches_example.py* script from the same folder). We can see that the matcher has made a few approximations, for example the match on the far right in green connects two points that have similar characteristics locally but are not really identical. To avoid including these approximations, the software filter the matches (see below for details).|

If the user wishes to obtain additional information and wants to estimate a population based on the clustering results, then further analysis are performed. The following calculations are made assuming that:

- The sounds are taken from long-term acoustic monitoring, with several days of recordings.
- The name of each file is composed of different parts separated by an underscore (“_”) and contains its recording date in the format “yearmonthday” in the second position. For example, "xxxxx_20230619_xxxxxx.wav" indicates that the file was recorded on June 19, 2023.

Based on the recording date and cluster assignment of each file, the software calculates the number of sounds and clusters per day (saved in the file *number_of_clusters_per_day.csv*), the number of sounds per cluster per day (saved in the file *number_of_sounds_per_cluster_per_date.csv*). Using these data, a presence index (PI, saved in *presence_index.csv*), defined in [4], is calculated for each cluster $k$ (1) :

$$PI_k = \frac{N_{sounds}(k)}{N_{sounds}(total)} \times \frac{N_{days}(k)}{N_{days}(total)}  \qquad(1)$$

With $N_{sounds}(k)$, the number of sounds assigned to the cluster $k$ ; $N_{sounds}(total)$, the total number of sounds in the dataset; $N_{days}(k)$, the number of days on which $k$ is present, $N_{days}(total)$, the total number of days with at least one sound, i.e. the duration of the study with acoustic activity in days.

A presence index of 0.01 was used in [4] to estimate the number of male ptarmigan regularly present in the study area. This threshold has been retained here. For the moment it is not changeable using the interface, but it is possible to modify it in the code, see *demo_script.py*). Using this threshold allows to estimate the number of clusters regularly present in the study.

Some individuals or groups may be divided into several clusters because their vocalizations or songs evolve over the course of the study. This is particularly true for stereotyped sound emissions and birds learning constantly. In the first example, the variability of which is based mainly on genetics and physiological condition of the individual transmitter. The sounds emitted by an individual can thus vary according to weather or food availability. In the second example, individuals incorporate constantly ew patterns in their repertoire. In addition, some individuals may be present only sporadically or are just passing by. Finally, the same sounds may be split in different clusters due to noise: if the software is configured to detect very fine differences, then the slightest noise present may interfere with correct grouping. As a result, the total number of clusters found in the analysis may be an overestimate of the population. The total size of the population can then be estimated via a generalization of the presence index and an information criterion inspired by the AIC (Akaike Information Criterion, [7-8]).

We can generalize the cluster population presence index to the PPI (Presence Population Index) (2) :

$$PPI_n = \frac{\sum_{k}^{n} N_{sounds}(k)}{N_{sounds}(total)} \times \frac{\bigcup_{k}^{n}N_{days}(k)}{N_{days}(total)}  \qquad(2)$$

$PPI_n$ is the PPI for a population of $n$ clusters; $\sum_{k}^{n} N_{sounds}(k)$, the sum of the number of sounds from the $n$ clusters in the population; $N_{sounds}(total)$, the total number of sounds in the dataset; $\bigcup_{k}^{n}N_{days}(k)$, the number of days on which at least one of the clusters in the population is present; $N_{days}(total)$, the total number of days with at least one sound, i.e. the duration of the study with acoustic activity in days. The PPI is calculated for populations ranging from 1 to the total number of clusters found.  Clusters are gradually included in populations according to their presence in the dataset, in descending order of PI: the population containing $n$ individuals will contain the $n$ individuals with the highest PI.

The PPI can be seen as an approximation of the probability that the $n$ clusters in the population represent the data, and is therefore an estimate of the real number of individuals: the more clusters are added to the population, the closer the PPI is to 1. This PPI must therefore be balanced against the risk of overestimating the number of individuals present in the real population. The AIC (3), or Akaike Information Criterion [7-8], measures the quality of a statistical model by balancing the explanatory power of the model, the natural logarithm of its likelihood $\log(L)$ and $n$, the number of parameters in a statistical model:

$$AIC = 2 \times n - 2 \times \log(L) \qquad (3)$$

Based on this equation, we define the Population Information Criterion (4) or PIC for an estimated population of $n$ clusters as a function of $N$, the total number of clusters found by clustering and $PPI_n$, the PPI of an estimated population of $n$ clusters :

$$PIC_n = 2 \times \frac{n}{N} - 2 \times \log(1+ PPI_n) \qquad (4)$$

1 is added to the PPI to make the logarithm positive, and $n$ divided by $N$ to give a ratio ranging from 0 to 1, just like the PPI. Thus, PPI and PIC will have roughly the same influence on the PIC value. The PIC therefore balances the number of clusters in the population with the probability that these $n$ clusters are representative of the real population. The ideal number of clusters is therefore estimated by minimizing the PIC: the $n$ with the smallest PIC represents the estimate of the total number of individuals in the population. The PIC and PPI values are stored and saved in the CSV file *PPI_PIC.csv*.

Therefore, the software gives two estimates: the number of resident individuals based on the Presence Index, and the total population based on the Population Information Criterion. These estimates are based on relatively simple calculations, but are sufficient for the purpose needed here. If users want to perform further, more complex analyzes of population dynamics (capture-mark-recapture, ...), the results saved by the software can be used if necessary.


# Installation

The software is available as executables (generated using pyinstaller [9]) for Windows (only tested on Windows 11) and Linux that require no dependencies. macOS users can still execute the python code (see below) but will need the installation of Python and the libraries.

If you want to execute the python code, you need to install these libraries:

- With pip:

```
pip install numpy scipy scikit-learn matplotlib soxr pywavelets pandas opencv-python
```
- With conda:
```
conda install numpy scipy scikit-learn matplotlib soxr pywavelets pandas opencv-python
```
To get the software and the python codes, you can clone the repo with git by typing in the terminal:

```
git clone https://github.com/lpoaura/LagoPObs
```

You can also download the repo as a ZIP file using the *Code* button and clicking on *Download ZIP* (Fig.4). The software and the codes will be available in the downloaded ZIP file.


|![download_repo.png](Readme/download_repo.png)|
|:--:|
|Figure 4: Download the GitHub repo in ZIP format.|


# Starting the software

## Using executables

No need to install the libraries if you are using the executables.

- For Windows users: clone/download this repo, unzip and go to the folder and double-click on *LagoPObs_windows.exe*. The software may take a few seconds to open.

- For Linux users: clone/download, unzip this repo and go to the folder. Double-click on *LagoPObs_linux*. If the application does not start, then make it executable using the terminal:
    ```
    chmod +x LagoPobs_linux
    ```
    Then you should be able to execute it by double-clicking or typing in a terminal:
    ```
    .\LagoPobs_linux
    ```

## Running the python code

The code was tested for Python 3.9, 3.11, 3.12 and 3.13. There should be no problems with 3.10. The only strict requirement is to have a version of scikit-learn that is 1.3 or higher (inclusion of HDBSCAN).
Install the required libraries with *pip* or *conda*.
You will need to clone/download this repo, unzip and go to the folder.
If you want to use the GUI (Graphical User Interface), open a terminal and type:
```
python LagPObs.py
```

If you just want to study and use the underlying code of the analysis without the GUI, e.g. to use it in a pipeline or to test for different configurations at the same time, open and use the *demo_script.py* file.


# Using the software

The GUI was made using tkinter [10] and is simple with only a few buttons and fields (Fig.5). The software is automatically setup to estimate populations of rock ptarmigans.

|![GUI_LagoPObs.png](Readme/GUI_LagoPObs.png)|
|:--:|
|Figure 5: The user interface.|

## Best practices

Before describing the interface in detail, here's a summary of the best practices and the ideal order of operations:

1) Select the folder with the WAV files.

2) Select the folder where the results will be stored.

3) If you don't want to estimate a population or a number of groups, deactivate *Population estimation*.

4) If there is noise in the sounds studied, try to improve sound quality by activating wavelet filtering via *Wavelet filtering*. Check that denoising does not alter the signals of interest.

5) Set the appropriate frequency band for the signals with *Lowest frequency* and *Highest frequency*. Ideally, the setup should contain only the signal of interest.

6) Set the STFT parameters, and try out the software a few times to see if the settings looks fine. Window sizes can be fine-tuned precisely. For some species, a change of a few units can drastically increase the quality of the results.

7) Test different feature extraction algorithms to see how well keypoints are recognized. Don't hesitate to run the software a few times and check visually.

8) Test the different types of clustering with a dataset where group membership is known and/or easily verifiable (e.g. you want to separate two different types of signal), to check the quality of the clustering results. Some algorithms tend to subdivide sounds into smaller clusters, while others will be more conservative and minimize the number of clusters.

9) In parallel with 8, set the number of matches required to calculate distances. Increasing the number of matches used to calculate distances between images enables the software to separate signals according to finer differences, but increases the risk of overestimating the number of clusters and increases the influence of noise (if still present after filtration) on the results. The ideal compromise must therefore be found for each use of the software. The number of matches should not exceed 500.

## Description of the interface

To change the folders where the sounds are located, press *Select input folder*, to change the directory where the results will be saved, press *Select output folder*. By default, the location of the program is used. The input folder can have other types of files than WAV inside, it will automatically filter out non-WAV files. It accepts multiple sample rates and bitrates as the sounds will be automatically converted to 64 bits arrays and will be resampled.

Below, you have access to the different analysis parameters. By default, all parameters are set so that the software can discriminate between rock ptarmigan males.

The *Wavelet filtering* option allow to activate/disable the option for the software to perform a wavelet denoising scheme based on the stationary wavelet transform (i.e. the "algorithme à trous", see PyWavelets [11] and [12]) using a biorthogonal wavelet. The wavelet cannot be modified at the moment without going into the code, but it should work for a wide range of signals. The sound is first resampled (see below) and bandpass filtered between the *Lowest frequency* and the *Highest frequency*, then it is decomposed using a multilevel stationary wavelet transform in several levels. For each level, Fisher's kurtosis is calculated and the level is set to 0s if the kurtosis is negative. Then, a soft thresholding is applied using the standard deviation of each particular level as the threshold. The signal is then reconstructed using the modified levels. The bandpass filter is applied even if *Wavelet filtering* is disabled.

The *Lowest frequency* and *Highest frequency* set the frequency bandwidth of the bandpass filtering (an order 10 butterworth filter). The *Highest frequency* parameter is really important as it will also condition the new sampling rate $FS$ at which the sounds will be resampled using the formula (5):

$$FS = 2 \times (Highest frequency + 100) \qquad(5)$$

100 is added to avoid frequency aliasing at some frequencies of interest. The choice of the *Highest frequency* is thus critical. It also implies that the temporal resolution of your spectrograms will be modified so be sure to check that the window lengths are still appropriate after modifying this parameter.

The characteristics of the spectrograms of the filtered signal and its envelope can be configured using the next 4 parameters: *Window length* and *Overlap of STFT of sound* for the STFT performed on the filtered signal and *Window length* and *Overlap* of the STFT for the spectrogram of the envelope. A hamming window is used for both spectrograms and cannot be modified using the interface.

The *Feature extraction algorithm* parameter is used to select the algorithm responsible for extracting keypoints and descriptors. The aim of these algorithms is to detect the critical keypoints in the image that best describe its characteristics. Each keypoint is accompanied by a descriptor that represents the local features of the image around the keypoint (Fig.1,2). A list of algorithms from OpenCV [13] is proposed: ORB [1-2], Custom ORB is an ORB but with parameters set specifically for rock ptarmigan spectrograms, SIFT [14-15], AKAZE [16-17], KAZE [18-19]. ORB or Custom ORB should be suitable for most applications, and are also the fastest. After extracting key points and descriptors, a Brute force matcher [5] will match the descriptors of each image pair (Fig.3). FLANN-based matcher were tested but not included here as it performed equally or worst and were slower. A distance for each match is calculated, then the matches are ranked by increasing distance. The algorithm keeps a number of matches equal to the *Number of matches* smallest distances. The average distance of these matches is calculated and corresponds to the distance between the two images. The distance is calculated for each pair of images and used to construct the inter-image distance matrix.

The *Clustering algorithm* parameter controls the type of algorithm used for clustering. The choice is made from a list of algorithms proposed by scikit-learn [20-21]: Affinity propagation [22], agglomerative hierarchical clustering [23], Bisecting K-Means [24], Gaussian mixture model [25], HDBSCAN [26], K-Means [27], Mean Shift [28]. Scikit-learn provides a very good description of each algorithm, so no further details will be given here, and readers wishing to find out more can consult the link in [29]. Some algorithms tend to subdivide sounds into small clusters, while others are more conservative and minimize the number of clusters.

Finally, you can activate/deactivate the estimation of the population using *Population estimation*. See the technical description of the software for further informations.

Once the setup is made to your liking, click on *Validate and proceed to analysis*.

The software will perform a check-up for errors in the parameters and display an error message containing the detected errors if some are encountered. Figure 6 is an example of a configuration with problems, and will give the error message shown on the left of Figure 7. After closing the error message, the user is sent back to the main window to correct the problems. If the only errors encountered are decimal values when integer values are expected, then a simple warning is returned and the analysis continues once the warning window is closed (right part of Fig.7).

|![example_bad_configuration.png](Readme/example_bad_configuration.png)|
|:--:|
|Figure 6: Example of incorrect software settings: there is no WAV file in the input folder, Lowest frequency is not an integer, Highest frequency and Windows length of STFT is not a number.|

|![error_message_GUI.png](Readme/error_message_GUI.png)![warning_message_GUI.png](Readme/warning_message_GUI.png)|
|:--:|
|Figure 7: On the left, error message caused by the configuration in Figure 6. Right, warning given when there are only decimal problems.|

Once this verification step has been completed, the software will redisplay the configuration with the various parameters, requesting validation from the user (Figure 8).

|![configuration_validation.png](Readme/configuration_validation.png)|
|:--:|
|Figure 8: Software configuration validation window. The language of the buttons will be automatically setup on the language of the operating system (in this case, French).|

Once the configuration has been validated, another window opens, showing the progress of the analysis, with a bar that gradually fills up as the analysis progresses (Fig.9). If the user did not request to estimate a population, the window stops at "Analysis finished!" If the user asked to estimate a population, then the window will display estimates of resident individuals and the total number of individuals (see the software description for more information). A button allow to close the window and return to the main software window. If the user activates population estimation but the file names do not correspond to the expected format (see the technical description for more information), an error window will be displayed (Fig.10). The analysis then stops and the results are saved as if the user did not activate the population estimation.

|![state_of_analysis.png](Readme/state_of_analysis.png)|
|:--:|
|Figure 9: Analysis progress window, displaying the progress of the various stages. Here, the population estimation was activated.|

|![error_population_estimation.png](Readme/error_population_estimation.png)|
|:--:|
|Figure 10: Error window generated when file name format does not match software expectations.|

## Generated files

Once the analysis is complete, the files are saved in the folder indicated by the *Output folder*. Inside, images are generated for each sound, with the keypoints on them as shown in Figures 1 and 2. The name of each image is the name of the corresponding sound. A CSV file is also saved (*clustering_results.csv*) and contains a table with the first column containing the file names and the second with the cluster to which the sound belongs, with cluster numbers starting at 0.

If the user activated population estimation, 4 supplementary CSV files are generated:

- *number_of_clusters_per_day.csv* contains the number of sounds and clusters per day.

- *number_of_sounds_per_cluster_per_date.csv* contains the daily number of sounds per cluster for each day.

- *presence_index.csv* contains the number of days of presence, sounds and presence index for each cluster. Clusters are sorted by decreasing presence index.

- *PPI_PIC.csv* contains the population presence index (PPI) and the population information criterion (PIC) as a function of the number of clusters in the estimated population.

- *results.txt* contains the number of individuals estimated using the Presence Index (PI) and the population information criterion (PIC)

# References

[1] Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011, November). ORB: An efficient alternative to SIFT or SURF. In 2011 International conference on computer vision (pp. 2564-2571). Ieee.

[2] https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

[3] Marin-Cudraz, T. (2019). Potentialité de la bioacoustique comme outil de dénombrement d'espèces difficiles d'accès: cas du Lagopède alpin (Lagopus muta) (Doctoral dissertation, Université de Lyon).

[4] Marin-Cudraz, T., Muffat-Joly, B., Novoa, C., Aubry, P., Desmet, J. F., Mahamoud-Issa, M., ... & Sèbe, F. (2019). Acoustic monitoring of rock ptarmigan: a multi-year comparison with point-count protocol. Ecological Indicators, 101, 710-719.

[5] Jakubović, A., & Velagić, J. (2018, September). Image feature matching and object detection using brute-force matchers. In 2018 International Symposium ELMAR (pp. 83-86). IEEE.

[6] https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

[7] Akaike, H. (1998). Information theory and an extension of the maximum likelihood principle. In Selected papers of hirotugu akaike (pp. 199-213). New York, NY: Springer New York.

[8] https://en.wikipedia.org/wiki/Akaike_information_criterion

[9] https://pyinstaller.org/en/stable/

[10] Lundh, F. (1999). An introduction to tkinter. URL: www. pythonware. com/library/tkinter/introduction/index. htm, 539, 540.

[11] Lee, G., Gommers, R., Waselewski, F., Wohlfahrt, K., & O'Leary, A. (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237.

[12] https://en.wikipedia.org/wiki/Stationary_wavelet_transform

[13] Bradski, G., & Kaehler, A. (2000). OpenCV. Dr. Dobb’s journal of software tools, 3(2).

[14] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60, 91-110.

[15] https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

[16] Alcantarilla, P. F., & Solutions, T. (2011). Fast explicit diffusion for accelerated features in nonlinear scale spaces. IEEE Trans. Patt. Anal. Mach. Intell, 34(7), 1281-1298.

[17] https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html

[18] Alcantarilla, P. F., Bartoli, A., & Davison, A. J. (2012). KAZE features. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12 (pp. 214-227). Springer Berlin Heidelberg.

[19] https://docs.opencv.org/4.x/d3/d61/classcv_1_1KAZE.html

[20] Kramer, O., & Kramer, O. (2016). Scikit-learn. Machine learning for evolution strategies, 45-53.

[21] https://scikit-learn.org/stable/

[22] McInnes, L., & Healy, J. (2017, November). Accelerated hierarchical density based clustering. In 2017 IEEE international conference on data mining workshops (ICDMW) (pp. 33-42). IEEE.

[23] https://en.wikipedia.org/wiki/Hierarchical_clustering

[24] Di, J., & Gou, X. (2018). Bisecting K-means Algorithm Based on K-valued Selfdetermining and Clustering Center Optimization. J. Comput., 13(6), 588-595.

[25] Bouman, C. A., Shapiro, M., Cook, G. W., Atkins, C. B., & Cheng, H. (1997, April). Cluster: An unsupervised algorithm for modeling Gaussian mixtures.

[26] L. McInnes and J. Healy, (2017). Accelerated Hierarchical Density Based Clustering. In: IEEE International Conference on Data Mining Workshops (ICDMW), 2017, pp. 33-42. Accelerated Hierarchical Density Based Clustering

[27] Ahmed, M., Seraj, R., & Islam, S. M. S. (2020). The k-means algorithm: A comprehensive survey and performance evaluation. Electronics, 9(8), 1295.

[28] Comaniciu, D., & Meer, P. (2002). Mean shift: A robust approach toward feature space analysis. IEEE Transactions on pattern analysis and machine intelligence, 24(5), 603-619.

[29] https://scikit-learn.org/stable/modules/clustering.html
